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

from ._utils import (
    _calc_batches_M_N,
    _dims,
    _dims_from_tail,
    _product_of_dims_slice,
)


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
    tile_x: Int = 32,
    tile_y: Int = 32,
    y_rep: Int = 1,
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

    comptime TILE = tile_x
    comptime TY = tile_y
    comptime Y_REP = y_rep
    comptime assert TILE % TY == 0
    comptime assert Y_REP >= 1
    comptime ELEMS = TILE // TY
    # Full tiles only: skip per-element bounds (640×480 with TILE=32).
    comptime full_tiles = (Int(M) % TILE == 0) and (Int(N) % TILE == 0)
    # Diagonal block map: cuts DRAM partition camping on Pascal (GTX 10xx).
    # Grid.y covers ceil(tiles_y / Y_REP); each block owns Y_REP tiles along M.
    comptime tiles_y = ceildiv(Int(M), TILE)
    comptime tiles_y_groups = ceildiv(tiles_y, Y_REP)

    var tx = UInt(thread_idx.x)
    var ty = UInt(thread_idx.y)
    var tile_x_idx = UInt(block_idx.x)
    var tile_y_group = (UInt(block_idx.x) + UInt(block_idx.y)) % UInt(
        tiles_y_groups
    )
    var bx = tile_x_idx * UInt(TILE)
    var b = UInt(block_idx.z)

    comptime tile_layout = row_major[TILE, TILE + 1, 2]()
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

        # Full tiles: stage global→regs→SM then SM→regs→global for better ILP.
        # TILE+1 column pad avoids bank conflicts (vs xor swizzle).
        comptime TPAD = TILE + 1
        comptime for yr in range(Y_REP):
            var tile_y_idx = tile_y_group * UInt(Y_REP) + UInt(yr)
            if tile_y_idx >= UInt(tiles_y):
                continue
            var by = tile_y_idx * UInt(TILE)

            comptime if full_tiles:
                comptime reg_layout = row_major[ELEMS, 2]()
                var regs = stack_allocation[src.dtype](reg_layout)
                comptime for i in range(ELEMS):
                    var x_in = bx + tx
                    var y_in = by + ty + UInt(i * TY)
                    regs.ptr.store(
                        i * 2, src_base.load[2]((y_in * N + x_in) * 2)
                    )
                comptime for i in range(ELEMS):
                    var row = ty + UInt(i * TY)
                    tile_ptr.store(
                        (row * UInt(TPAD) + tx) * 2,
                        regs.ptr.load[2](i * 2),
                    )
            else:
                comptime for i in range(ELEMS):
                    var x_in = bx + tx
                    var y_in = by + ty + UInt(i * TY)
                    var row = ty + UInt(i * TY)
                    if x_in < N and y_in < M:
                        var val = src_base.load[2]((y_in * N + x_in) * 2)
                        tile_ptr.store((row * UInt(TPAD) + tx) * 2, val)

            barrier()

            comptime if full_tiles:
                comptime reg_layout = row_major[ELEMS, 2]()
                var regs = stack_allocation[src.dtype](reg_layout)
                comptime for i in range(ELEMS):
                    var row = ty + UInt(i * TY)
                    regs.ptr.store(
                        i * 2,
                        tile_ptr.load[2]((tx * UInt(TPAD) + row) * 2),
                    )
                comptime for i in range(ELEMS):
                    var x_out = by + tx
                    var y_out = bx + ty + UInt(i * TY)
                    dst_base.store(
                        (y_out * M + x_out) * 2, regs.ptr.load[2](i * 2)
                    )
            else:
                comptime for i in range(ELEMS):
                    var x_out = by + tx
                    var y_out = bx + ty + UInt(i * TY)
                    var row = ty + UInt(i * TY)
                    if x_out < M and y_out < N:
                        var val = tile_ptr.load[2]((tx * UInt(TPAD) + row) * 2)
                        dst_base.store((y_out * M + x_out) * 2, val)

            # SM reused for next Y_REP tile and/or next batch.
            if yr + 1 < Y_REP or batch + 1 < scheduled_runs:
                barrier()


def _restore_reversed_nd_gpu[
    dst_dtype: DType,
    dst_layout_type: TensorLayout,
    dst_origin: MutOrigin,
    src_origin: ImmOrigin,
    *,
    scheduled_batches: UInt,
    tile_x: Int = 32,
    tile_y: Int = 32,
    y_rep: Int = 1,
](
    dst: TileTensor[dst_dtype, dst_layout_type, dst_origin],
    src: TileTensor[dst_dtype, dst_layout_type, src_origin],
):
    """Reversed layout `[D_{r-1},…,D0]` → native `[D0,…,D_{r-1}]` (tiled)."""
    comptime dims = _dims[dst_layout_type]
    comptime assert dims.rank >= 2, "restore needs ≥ 2 spatial axes"
    comptime rank = dims.rank
    comptime D0 = UInt(dims.static_shape[0])
    comptime Dlast = UInt(dims.static_shape[rank - 1])
    comptime middle_prod = UInt(
        _product_of_dims_slice[dims, 1, rank - 1]()
    )
    comptime volume = UInt(_product_of_dims_slice[dims, 0, rank]())
    comptime src_stride_last = UInt(
        _product_of_dims_slice[dims, 0, rank - 1]()
    )
    comptime dst_stride_0 = UInt(
        _product_of_dims_slice[dims, 1, rank]()
    )
    comptime TILE = tile_x
    comptime TY = tile_y
    comptime Y_REP = y_rep
    comptime assert TILE % TY == 0
    comptime assert Y_REP >= 1
    comptime ELEMS = TILE // TY
    comptime full_tiles = (Int(D0) % TILE == 0) and (Int(Dlast) % TILE == 0)
    comptime tiles_y = ceildiv(Int(Dlast), TILE)
    comptime tiles_y_groups = ceildiv(tiles_y, Y_REP)

    var tx = UInt(thread_idx.x)
    var ty = UInt(thread_idx.y)
    var tile_x_idx = UInt(block_idx.x)
    var tile_y_group = (UInt(block_idx.x) + UInt(block_idx.y)) % UInt(
        tiles_y_groups
    )
    var bx = tile_x_idx * UInt(TILE)
    var b = UInt(block_idx.z)

    comptime TPAD = TILE + 1
    comptime tile_layout = row_major[TILE, TPAD, 2]()
    var shared_tile = stack_allocation[
        src.dtype, address_space=AddressSpace.SHARED
    ](tile_layout)
    var tile_ptr = shared_tile.ptr

    comptime extra_fft_batches = UInt(dst_layout_type.static_shape[0])
    comptime total_batches = extra_fft_batches * middle_prod
    comptime scheduled_runs = ceildiv(total_batches, scheduled_batches)
    for batch in range(scheduled_runs):
        var current_batch = b + batch * scheduled_batches
        if current_batch >= total_batches:
            return

        var outer = current_batch // middle_prod
        var mid = current_batch % middle_prod
        var vol_base = outer * volume

        var rem = mid
        var dst_middle: UInt = 0
        comptime for i in range(1, rank - 1):
            comptime Di = UInt(dims.static_shape[i])
            comptime native_stride_i = UInt(
                _product_of_dims_slice[dims, i + 1, rank]()
            )
            var di = rem % Di
            rem = rem // Di
            dst_middle += di * native_stride_i

        comptime for yr in range(Y_REP):
            var tile_y_idx = tile_y_group * UInt(Y_REP) + UInt(yr)
            if tile_y_idx >= UInt(tiles_y):
                continue
            var by = tile_y_idx * UInt(TILE)

            comptime if full_tiles:
                comptime reg_layout = row_major[ELEMS, 2]()
                var regs = stack_allocation[src.dtype](reg_layout)
                comptime for e in range(ELEMS):
                    var d0_in = bx + tx
                    var dlast_in = by + ty + UInt(e * TY)
                    var src_off = (
                        vol_base
                        + dlast_in * src_stride_last
                        + mid * D0
                        + d0_in
                    ) * 2
                    regs.ptr.store(e * 2, src.ptr.load[2](src_off))
                comptime for e in range(ELEMS):
                    var row = ty + UInt(e * TY)
                    tile_ptr.store(
                        (row * UInt(TPAD) + tx) * 2,
                        regs.ptr.load[2](e * 2),
                    )
            else:
                comptime for e in range(ELEMS):
                    var d0_in = bx + tx
                    var dlast_in = by + ty + UInt(e * TY)
                    var row = ty + UInt(e * TY)
                    if d0_in < D0 and dlast_in < Dlast:
                        var src_off = (
                            vol_base
                            + dlast_in * src_stride_last
                            + mid * D0
                            + d0_in
                        ) * 2
                        var val = src.ptr.load[2](src_off)
                        tile_ptr.store((row * UInt(TPAD) + tx) * 2, val)

            barrier()

            comptime if full_tiles:
                comptime reg_layout = row_major[ELEMS, 2]()
                var regs = stack_allocation[src.dtype](reg_layout)
                comptime for e in range(ELEMS):
                    var row = ty + UInt(e * TY)
                    regs.ptr.store(
                        e * 2,
                        tile_ptr.load[2]((tx * UInt(TPAD) + row) * 2),
                    )
                comptime for e in range(ELEMS):
                    var dlast_out = by + tx
                    var d0_out = bx + ty + UInt(e * TY)
                    var dst_off = (
                        vol_base
                        + d0_out * dst_stride_0
                        + dst_middle
                        + dlast_out
                    ) * 2
                    dst.ptr.store(dst_off, regs.ptr.load[2](e * 2))
            else:
                comptime for e in range(ELEMS):
                    var dlast_out = by + tx
                    var d0_out = bx + ty + UInt(e * TY)
                    var row = ty + UInt(e * TY)
                    if dlast_out < Dlast and d0_out < D0:
                        var val = tile_ptr.load[2]((tx * UInt(TPAD) + row) * 2)
                        var dst_off = (
                            vol_base
                            + d0_out * dst_stride_0
                            + dst_middle
                            + dlast_out
                        ) * 2
                        dst.ptr.store(dst_off, val)

            if yr + 1 < Y_REP or batch + 1 < scheduled_runs:
                barrier()


@always_inline
def _scatter_dim_result[
    out_dtype: DType,
    write_global_lhs: Bool,
    last_write_lhs: Bool,
    sm_complex_stride: Int = 2,
    global_complex_stride: Int = 2,
](
    global_i: UInt,
    shared_lhs: TileTensor[mut=True, out_dtype, ...],
    shared_rhs: TileTensor[mut=True, out_dtype, ...],
    base_out: TileTensor[mut=True, out_dtype, ...],
    base_calc: TileTensor[mut=True, out_dtype, ...],
):
    var c_num: SIMD[out_dtype, 2]
    comptime if last_write_lhs:
        c_num = shared_lhs.raw_load[2](Int(global_i) * sm_complex_stride)
    else:
        c_num = shared_rhs.raw_load[2](Int(global_i) * sm_complex_stride)

    comptime if write_global_lhs:
        base_out.raw_store(Int(global_i) * global_complex_stride, c_num)
    else:
        base_calc.raw_store(Int(global_i) * global_complex_stride, c_num)
