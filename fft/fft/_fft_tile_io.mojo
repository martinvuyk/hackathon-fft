"""FFT TileIO — transpose between layout dimensions."""

from max.algorithm import parallelize
from std.math import ceildiv
from std.bit import prev_power_of_two
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.sys.info import simd_width_of
from std.utils.index import IndexList
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from ._utils import _calc_batches_M_N, _dims, _dims_from_tail


def _complex_transpose_mask[TILE: Int]() -> IndexList[TILE * TILE * 2]:
    var mask = IndexList[TILE * TILE * 2](fill=0)
    for i in range(TILE):
        for j in range(TILE):
            var src_re = i * (2 * TILE) + j * 2
            var src_im = src_re + 1

            var dst_re = j * (2 * TILE) + i * 2
            var dst_im = dst_re + 1

            mask[dst_re] = src_re
            mask[dst_im] = src_im
    return mask


@always_inline
def _transpose_cpu[
    *, into_: Int, from_: Int
](
    dst: TileTensor[mut=True, ...],
    src: TileTensor[mut=False, dst.dtype, dst.LayoutType, ...],
    num_workers: Int,
):
    comptime sizes = _calc_batches_M_N[
        _dims_from_tail[src.LayoutType], into_=into_, from_=from_
    ]()
    comptime intra_fft_batches = Int(sizes[0])
    comptime M = Int(sizes[1])
    comptime N = Int(sizes[2])
    comptime TILE = min(
        prev_power_of_two(min(M, N)), simd_width_of[src.dtype]()
    )
    comptime mask = _complex_transpose_mask[TILE]()

    var src_ptr = src.ptr
    var dst_ptr = dst.ptr

    def _transpose_batch(b: Int) {imm src_ptr, imm dst_ptr}:
        var src_base = src_ptr + b * M * N * 2
        var dst_base = dst_ptr + b * M * N * 2

        for i in range(0, M, TILE):
            for j in range(0, N, TILE):
                if i + TILE <= M and j + TILE <= N:
                    var tile_in = SIMD[src.dtype, TILE * TILE * 2]()

                    comptime for row in range(TILE):
                        var val = src_base.load[TILE * 2](
                            ((i + row) * N + j) * 2
                        )
                        UnsafePointer(to=tile_in).unsafe_bitcast[
                            Scalar[src.dtype]
                        ]().store(row * TILE * 2, val)

                    var tile_out = tile_in.shuffle[mask]()

                    comptime for col in range(TILE):
                        var val = (
                            UnsafePointer(to=tile_out)
                            .unsafe_bitcast[Scalar[src.dtype]]()
                            .load[TILE * 2](col * TILE * 2)
                        )
                        dst_base.store(((j + col) * M + i) * 2, val)
                else:
                    for ii in range(i, min(i + TILE, M)):
                        for jj in range(j, min(j + TILE, N)):
                            var val = src_base.load[2]((ii * N + jj) * 2)
                            dst_base.store((jj * M + ii) * 2, val)

    parallelize(
        _transpose_batch,
        intra_fft_batches,
        min(num_workers, intra_fft_batches, TILE),
    )


def _transpose_gpu[
    dst_dtype: DType,
    dst_layout_type: TensorLayout,
    dst_origin: MutOrigin,
    src_origin: ImmOrigin,
    *,
    into_: Int,
    from_: Int,
    scheduled_batches: UInt,
](
    dst: TileTensor[dst_dtype, dst_layout_type, dst_origin],
    src: TileTensor[dst_dtype, dst_layout_type, src_origin],
):
    comptime sizes = _calc_batches_M_N[
        _dims[dst_layout_type], into_=into_, from_=from_
    ]()
    comptime intra_fft_batches = sizes[0]
    comptime M = sizes[1]
    comptime N = sizes[2]

    comptime TILE = 32

    var tx = UInt(thread_idx.x)
    var ty = UInt(thread_idx.y)
    var bx = UInt(block_idx.x) * TILE
    var by = UInt(block_idx.y) * TILE
    var b = UInt(block_idx.z)

    comptime tile_layout = row_major[TILE, TILE, 2]()
    var shared_tile = stack_allocation[
        src.dtype, address_space=AddressSpace.SHARED
    ](tile_layout)
    var tile_ptr = shared_tile.ptr

    comptime extra_fft_batches = UInt(dst_layout_type.static_shape[0])
    comptime total_batches = intra_fft_batches * extra_fft_batches
    comptime scheduled_runs = ceildiv(total_batches, scheduled_batches)
    for batch in range(scheduled_runs):
        var current_batch = b + batch * scheduled_batches
        if current_batch >= total_batches:
            return

        var src_base = src.ptr + current_batch * M * N * 2
        var dst_base = dst.ptr + current_batch * M * N * 2

        var x_in = bx + tx
        var y_in = by + ty

        if x_in < N and y_in < M:
            var val = src_base.load[2]((y_in * N + x_in) * 2)
            tile_ptr.store((ty * TILE + (tx ^ ty)) * 2, val)

        barrier()

        var x_out = by + tx
        var y_out = bx + ty

        if x_out < M and y_out < N:
            var val = tile_ptr.load[2]((tx * TILE + (tx ^ ty)) * 2)
            dst_base.store((y_out * M + x_out) * 2, val)

        barrier()


@always_inline
def _scatter_dim_result[
    out_dtype: DType, write_global_lhs: Bool, last_write_lhs: Bool
](
    global_i: UInt,
    shared_lhs: TileTensor[mut=True, out_dtype, ...],
    shared_rhs: TileTensor[mut=True, out_dtype, ...],
    base_out: TileTensor[mut=True, out_dtype, ...],
    base_calc: TileTensor[mut=True, out_dtype, ...],
):
    var c_num: SIMD[out_dtype, 2]
    comptime if last_write_lhs:
        c_num = shared_lhs.raw_load[2](Int(global_i) * 2)
    else:
        c_num = shared_rhs.raw_load[2](Int(global_i) * 2)

    comptime if write_global_lhs:
        base_out.raw_store(Int(global_i) * 2, c_num)
    else:
        base_calc.raw_store(Int(global_i) * 2, c_num)
