"""GPU Stockham layout helpers — values that actually change codegen per dim."""

from layout import TensorLayout
from std.math import ceildiv


@always_inline
def _sm_complex_stride[
    dim: Int,
    amnt_dims: Int,
    *,
    warp_size: Int = 32,
    block_threads: Int = 0,
]() -> Int:
    """Shared-memory scalars per complex.

    Bank-pad 1D when the block has at least four warps on a packed line.
    One-warp intra blocks do not need the extra lane.
    """
    comptime four_warps = 4 * warp_size
    comptime nthr = block_threads if block_threads > 0 else dim
    return 3 if amnt_dims == 1 and nthr >= four_warps else 2


@always_inline
def _fft_nd_fuse_forward_restore[start_dim_idx: Int]() -> Bool:
    """Replace forward transpose chain with restore (2D+ saves one global pass)."""
    return start_dim_idx >= 1


@always_inline
def _fft_nd_fuse_back_transpose[
    dims: TensorLayout,
    dim: Int,
    *,
    shared_mem_per_block_bytes: Int,
    warp_size: Int = 32,
    max_thread_block_size: Int = 1024,
]() -> Bool:
    """2D last-axis pack+transpose-store to skip the standalone back-T.

    Needs room for ≥2 packed lines of ping-pong and a thread count that fits
    a block with `tpt ≈ dim/4` (same ballpark as SBRC).
    """
    comptime rank = dims.rank
    comptime if rank != 2:
        return False
    comptime d0 = Int(dims.static_shape[0])
    comptime d1 = Int(dims.static_shape[1])
    comptime if dim != d1:
        return False
    # Skip tiny / unit-test shapes; keep 64×48 as a fuse-back correctness check.
    comptime if d0 < 48 or d1 < 48:
        return False
    comptime if _fft_nd_use_column_tiles[
        dims, warp_size=warp_size, max_thread_block_size=max_thread_block_size
    ]():
        return False
    comptime line_bytes = 2 * dim * 2 * 4  # ping-pong scalars, fp32
    comptime max_L = shared_mem_per_block_bytes // max(line_bytes, 1)
    # Cap tpt and L so Pascal register file fits (480×heavy bfly → OOR).
    comptime tpt = min(ceildiv(dim, 4), 64)
    comptime L = min(2, min(max_L, max_thread_block_size // max(tpt, 1)))
    return L >= 2 and tpt * L <= max_thread_block_size


@always_inline
def _fft_nd_use_column_tiles[
    dims: TensorLayout, *, warp_size: Int = 32, max_thread_block_size: Int = 1024
]() -> Bool:
    """Column-tile ND when every spatial axis is the same pow2 that fits a block.

    Lower bound two warps: SBRC `tpt = dim/4` stays at least half a warp.
    """
    comptime d0 = Int(dims.static_shape[0])
    comptime pow2_fits = (
        (d0 & (d0 - 1)) == 0
        and d0 >= 2 * warp_size
        and d0 <= max_thread_block_size
    )
    comptime rank = dims.rank
    comptime if rank == 2:
        return pow2_fits and Int(dims.static_shape[1]) == d0
    elif rank == 3:
        return (
            pow2_fits
            and Int(dims.static_shape[1]) == d0
            and Int(dims.static_shape[2]) == d0
        )
    else:
        return False


@always_inline
def _fft_nd_use_contig_four_step[
    dim: Int,
    *,
    warp_size: Int = 32,
    max_n2: Int = 16,
]() -> Bool:
    """Contiguous mixed axis as Bailey `warp × n2` (e.g. 480 = 32×15)."""
    comptime n2 = dim // warp_size if (warp_size > 0 and dim % warp_size == 0) else 0
    comptime pow2 = (dim & (dim - 1)) == 0
    return not pow2 and n2 >= 3 and n2 <= max_n2


@always_inline
def _fft_nd_use_warp_four_step_column[
    dim: Int,
    ortho_len: Int,
    *,
    warp_size: Int = 32,
    max_n1: Int = 64,
    max_n2: Int = 64,
]() -> Bool:
    """2D column FFT via Bailey four-step with power-of-two `n1` tile width."""
    comptime n1 = _fft_nd_warp_fs_n1[dim, warp_size=warp_size, max_n1=max_n1]()
    comptime n2 = dim // n1 if (n1 > 0 and dim % n1 == 0) else 0
    return (
        ortho_len > 0
        and n1 >= warp_size
        and n2 >= 2
        and n2 <= max_n2
    )


@always_inline
def _fft_nd_warp_fs_n1[
    dim: Int, *, warp_size: Int = 32, max_n1: Int = 64
]() -> Int:
    """Largest power-of-two `n1` in `[warp, max_n1]` that divides `dim`."""
    comptime if max_n1 >= 64 and dim % 64 == 0:
        return 64
    elif max_n1 >= 32 and dim % 32 == 0:
        return 32
    elif dim % warp_size == 0:
        return warp_size
    else:
        return 0


@always_inline
def _complex_scalar_offset(complex_i: Int, *, stride: Int) -> Int:
    """Linear scalar offset of complex index `complex_i` for a given stride."""
    return complex_i * stride
