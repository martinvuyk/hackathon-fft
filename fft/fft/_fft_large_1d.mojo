"""Two-upload 1D path for axes that miss one thread block (N > max threads)."""

from std.math import pi, sin, cos, ceildiv
from std.gpu import thread_idx, block_idx, block_dim
from layout import TileTensor, TensorLayout
from max.gpu.host import DeviceContext

from ._fft_tile_io import _transpose_gpu
from ._utils import _large_1d_can_split, _large_1d_factor_pair


@always_inline
def _fft_axis_uses_four_step[
    dim: Int,
    max_threads_per_block: UInt,
    amnt_dims: Int,
]() -> Bool:
    """1D only, and only when the axis cannot occupy one thread block."""
    return (
        amnt_dims == 1
        and dim > Int(max_threads_per_block)
        and _large_1d_can_split[dim, Int(max_threads_per_block)]()
    )


@always_inline
def _fft_axis_four_step_factors[
    dim: Int, max_threads_per_block: UInt
]() -> Tuple[Int, Int]:
    return _large_1d_factor_pair[dim, Int(max_threads_per_block)]()


def _four_step_twiddle_kernel[
    dtype: DType,
    layout_type: TensorLayout,
    origin: MutOrigin,
    *,
    inverse: Bool,
    N1: Int,
    N2: Int,
](buf: TileTensor[dtype, layout_type, origin]):
    """`buf[b, j, k] *= exp(±2πi j k / (N1 N2))` on a `[B, N1, N2, 2]` view."""
    comptime assert dtype.is_floating_point()
    comptime N = N1 * N2
    comptime batches = Int(layout_type.static_shape[0])
    comptime total = batches * N

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= total:
        return
    var local = i % N
    var j = local // N2
    var k = local % N2
    if j == 0 or k == 0:
        return
    comptime sign = 1.0 if inverse else -1.0
    var theta_f32 = (
        sign * 2.0 * pi * Float64(j * k) / Float64(N)
    ).cast[DType.float32]()
    var wr = cos(theta_f32).cast[dtype]()
    var wi = sin(theta_f32).cast[dtype]()
    var re = buf.ptr.load(i * 2)
    var im = buf.ptr.load(i * 2 + 1)
    buf.ptr.store(i * 2, re * wr - im * wi)
    buf.ptr.store(i * 2 + 1, re * wi + im * wr)


def _enqueue_four_step_twiddle[
    dtype: DType,
    layout_type: TensorLayout,
    *,
    inverse: Bool,
    N1: Int,
    N2: Int,
](ctx: DeviceContext, buf: TileTensor[mut=True, dtype, layout_type, ...]) raises:
    comptime batches = Int(layout_type.static_shape[0])
    comptime total = batches * N1 * N2
    comptime block = 256
    comptime grid = max(ceildiv(total, block), 1)
    comptime func = _four_step_twiddle_kernel[
        dtype,
        layout_type,
        buf.origin,
        inverse=inverse,
        N1=N1,
        N2=N2,
    ]
    ctx.enqueue_function[func](buf, grid_dim=grid, block_dim=block)


def _enqueue_matrix_transpose[
    dtype: DType,
    layout_type: TensorLayout,
    *,
    scheduled_batches: UInt,
](
    ctx: DeviceContext,
    dst: TileTensor[mut=True, dtype, layout_type, ...],
    src: TileTensor[mut=False, dtype, layout_type, ...],
) raises:
    """Adjacent transpose on `[B, M, N, 2]` (memory becomes `[B, N, M]`)."""
    comptime dims_rank = layout_type.rank - 2
    comptime assert dims_rank == 2, "two-upload transpose expects rank-2 spatial"
    comptime M = Int(layout_type.static_shape[1])
    comptime N = Int(layout_type.static_shape[2])
    comptime TILE = 32
    comptime M_ = ceildiv(M, TILE)
    comptime N_ = ceildiv(N, TILE)
    comptime func = _transpose_gpu[
        dst_dtype=dtype,
        dst_layout_type=layout_type,
        dst_origin=dst.origin,
        src_origin=src.origin,
        into_=0,
        from_=1,
        scheduled_batches=scheduled_batches,
    ]
    ctx.enqueue_function[func](
        dst,
        src,
        grid_dim=(N_, M_, Int(scheduled_batches)),
        block_dim=(TILE, TILE, 1),
    )


def _copy_tile_kernel[
    dst_dtype: DType,
    src_dtype: DType,
    dst_layout_type: TensorLayout,
    src_layout_type: TensorLayout,
    dst_origin: MutOrigin,
    src_origin: ImmOrigin,
](
    dst: TileTensor[dst_dtype, dst_layout_type, dst_origin],
    src: TileTensor[src_dtype, src_layout_type, src_origin],
):
    comptime n = dst_layout_type.static_cosize
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dst.ptr.store(i, src.ptr.load(i).cast[dst_dtype]())


def _enqueue_buffer_copy_n[
    dst_dtype: DType,
    src_dtype: DType,
    dst_layout_type: TensorLayout,
    src_layout_type: TensorLayout,
](
    ctx: DeviceContext,
    dst: TileTensor[mut=True, dst_dtype, dst_layout_type, ...],
    src: TileTensor[mut=False, src_dtype, src_layout_type, ...],
) raises:
    comptime n = dst_layout_type.static_cosize
    comptime block = 256
    comptime grid = max(ceildiv(n, block), 1)
    comptime func = _copy_tile_kernel[
        dst_dtype,
        src_dtype,
        dst_layout_type,
        src_layout_type,
        dst.origin,
        src.origin,
    ]
    ctx.enqueue_function[func](dst, src, grid_dim=grid, block_dim=block)
