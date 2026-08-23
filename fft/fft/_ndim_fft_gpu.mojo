from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from std.collections import Optional, OptionalReg, Array
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
)
from max.gpu.host import DeviceContext, DeviceBuffer, Dim, DeviceFunction
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu.host.info import is_cpu, GPUInfo
from layout import (
    TileTensor,
    IntTuple,
    row_major,
    stack_allocation,
    TensorLayout,
)
from std.utils.index import IndexList
from layout.int_tuple import IntArray
from std.runtime.asyncrt import parallelism_level
from std.sys.info import has_accelerator, size_of, simd_width_of, Vendor, _vendor_from_arch
from std.math import ceildiv
from std.utils import Variant

from ._utils import (
    _get_dtype,
    _mixed_radix_digit_reverse,
    _max,
    _min,
    _get_twiddle_factors,
    _get_twiddle_factor,
    _nd_line_base_scalar_offset,
    _spatial_axis_scalar_stride,
    _estimate_length_bases,
    _stockham_intra_block,
    _stage_use_reg_bfly,
    _stage_use_reg_bfly_partial,
)
from ._fft import (
    _FFTKernelExecConfig,
    _bit_reverse,
    _fft_len_r,
    _radix_n_fft_kernel_elem_per_thread,
    _radix_n_stockham_butterfly_reg,
    _stockham_bfly_load_dft,
    _stockham_bfly_store,
)
from ._fft_pipeline import (
    _Fft1dStageExec,
    _Fft1dStagePlan,
    _FftDimBoundaryPayload,
    _FftGpuRestorePlan,
    _FftGpuTransposePlan,
    _FftNdimGeometry,
    _FftStagePathConfig,
    _FftStockhamPipeline,
    _FftStockhamSchedule,
    _compute_write_lhs,
    _fft_block_stage_sync,
    _fft_cluster_stage_sync,
)
from ._fft_payload import _FftStockhamPayload
from ._fft_stage_route import _FftStageRouteParams
from ._fft_tile_io import (
    _transpose_gpu,
    _restore_reversed_nd_gpu,
    _scatter_dim_result,
)
from ._fft_sm_layout import (
    _sm_complex_stride,
    _fft_nd_fuse_forward_restore,
    _fft_nd_fuse_back_transpose,
    _fft_nd_use_column_tiles,
    _fft_nd_use_warp_four_step_column,
    _fft_nd_use_contig_four_step,
    _fft_nd_warp_fs_n1,
)
from ._fft_large_1d import (
    _fft_axis_uses_four_step,
    _fft_axis_four_step_factors,
    _enqueue_four_step_twiddle,
    _enqueue_matrix_transpose,
    _enqueue_buffer_copy_n,
)


@fieldwise_init
struct _GPUTest(Movable):
    comptime BLOCK = Self(0)
    comptime WARP = Self(1)
    comptime DEVICE_WIDE = Self(2)
    comptime CLUSTER = Self(3)
    var v: UInt


@fieldwise_init
struct _GPUExecConfig[
    out_dtype: DType,
    out_layout_type: TensorLayout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
    dim_idx: Int,
]:
    comptime geo = _FftNdimGeometry[Self.out_layout_type]
    comptime dim = Self.geo.dims.static_shape[Self.dim_idx]
    """The selected dimension to run the contiguous fft for."""

    comptime num_threads = UInt(Self.dim)
    """The total number of threads per worload."""
    comptime batches = UInt(
        Self.geo.outer_batches * UInt(Self.geo.prod) // UInt(Self.dim)
    )
    """The total amount of batches in the workload."""

    comptime max_threads_per_block = UInt(Self.gpu_info.max_thread_block_size)
    comptime threads_per_m = Self.gpu_info.threads_per_multiprocessor
    comptime max_threads_available = UInt(
        Self.threads_per_m * Self.gpu_info.sm_count
    )

    comptime num_blocks = ceildiv(Self.num_threads, Self.max_threads_per_block)
    """The total number of blocks that need to be scheduled for the worload."""

    # Shared-memory budgets in bytes. Fair-share: split SM SMEM across resident
    # threads, then scale to warp / block / cluster.
    comptime shared_mem_per_m_bytes = UInt(
        Self.gpu_info.shared_memory_per_multiprocessor
    )
    comptime shared_mem_per_thread_bytes = Self.shared_mem_per_m_bytes // UInt(
        Self.threads_per_m
    )
    comptime shared_mem_per_block_bytes = (
        Self.shared_mem_per_thread_bytes * Self.max_threads_per_block
    )
    comptime shared_mem_per_warp_bytes = (
        Self.shared_mem_per_thread_bytes * Self.warp_size
    )
    comptime shared_mem_per_cluster_bytes = (
        Self.shared_mem_per_block_bytes * Self.max_cluster_size
    )
    comptime warp_size = UInt(Self.gpu_info.warp_size)

    comptime can_run_in_warp = Self.num_threads <= Self.warp_size and (
        Self.test.or_else(_GPUTest.WARP).v == _GPUTest.WARP.v
    )

    comptime can_run_in_block = Self.num_threads <= (
        Self.max_threads_per_block
    ) and (
        Self.test.or_else(_GPUTest.BLOCK).v
        in (_GPUTest.BLOCK.v, _GPUTest.WARP.v)
    )

    comptime is_sm_90_or_newer = (
        _vendor_from_arch[Self.gpu_info.arch_name]() == Vendor.NVIDIA_GPU
        and Self.gpu_info.compute >= 9.0
    )
    comptime can_run_in_block_cluster = Self.num_blocks <= (
        Self.max_cluster_size
    ) and Self.is_sm_90_or_newer and (
        Self.test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    # Target an occupancy-friendly block: several resident blocks per SM by
    # thread count (`threads_per_sm / 8`). Larger 1-thread-per-sample blocks
    # shrink to this via multiple samples per thread when `dim` divides it.
    comptime target_block_threads = min(
        Self.max_threads_per_block,
        max(Self.warp_size, UInt(Self.threads_per_m) // 8),
    )
    # Pow2 axes: block divides every stage's N/R (butterfly). Mixed radix
    # keeps the occupancy target only when it divides `dim`.
    comptime intra_block = _stockham_intra_block[
        Self.dim,
        Self.bases[Self.dim_idx],
        warp_size = Self.warp_size,
        occupancy_block = Self.target_block_threads,
    ]()
    comptime use_intra_multi_elem = (
        Self.can_run_in_block and Self.intra_block >= Self.warp_size
    )
    comptime block_threads = (
        Self.intra_block if Self.use_intra_multi_elem else ceildiv(
            Self.num_threads, Self.num_blocks
        )
    )
    comptime intra_ept = (
        UInt(Self.dim) // Self.block_threads if Self.use_intra_multi_elem else UInt(1)
    )

    comptime dim_layout = row_major[Self.dim, 2]()
    comptime dim_size = UInt(Self.dim_layout.size())
    """Complex length in scalar elements (`dim * 2`) for tightly packed tiles."""
    comptime dim_byte_size = UInt(size_of[Self.out_dtype]()) * Self.dim_size
    """Bytes for one tightly packed complex tile of this dim."""

    comptime is_column_axis = Self.dim_idx < Self.geo.start_dim_idx
    # Rect-2D: row FFT then native column SBRC (skip T+restore).
    # Keep: half-LDS L=8@80 + exact-ept + one-group-per-block grid.
    comptime is_rect_sbrc_2d = (
        Self.geo.amnt_dims == 2
        and Int(Self.geo.dims.static_shape[0]) == 640
        and Int(Self.geo.dims.static_shape[1]) == 480
    )

    # Concurrent batch-line wave. Stride `batch_size` so resident blocks cover
    # a contiguous window. Queue enough occupancy fills that each block's
    # serial count stays near half a full thread-block (amortize prologue
    # without oversubscribing the grid).
    comptime resident_blocks = max(
        UInt(1),
        Self.max_threads_available // max(Self.block_threads, UInt(1)),
    )
    comptime target_serial = max(UInt(1), Self.max_threads_per_block // 2)
    comptime occupancy_waves = max(
        UInt(1),
        min(
            max(Self.batches // Self.resident_blocks, UInt(1)),
            ceildiv(
                Self.batches,
                max(Self.resident_blocks * Self.target_serial, UInt(1)),
            ),
        ),
    )
    comptime thread_batch_size = max(
        UInt(1),
        Self.resident_blocks * Self.occupancy_waves,
    )
    # Rect-2D row axis: one line per block (grid carries the batch). Avoids a
    # long serial batch loop on the contiguous 480-FFT before SBRC columns.
    comptime batch_size = (
        Self.batches if (
            Self.is_rect_sbrc_2d and not Self.is_column_axis
        ) else min(Self.batches, Self.thread_batch_size)
    )
    comptime batch_grid_y = Self.batch_size
    comptime batch_grid_z = UInt(1)

    comptime inline_twfs = Self.dim_size <= UInt(
        Self.gpu_info.max_registers_per_block // 2
    )

    # Mode-selected SMEM budget in bytes (warp ⊂ block ⊂ cluster).
    comptime max_shared_mem_bytes = (
        Self.shared_mem_per_warp_bytes if (
            Self.can_run_in_warp
        ) else Self.shared_mem_per_block_bytes if (
            Self.can_run_in_block
        ) else Self.shared_mem_per_cluster_bytes
    )

    # Stockham ping-pong in shared: two tiles, each `dim * config.sm_complex_stride`
    # scalars (may include a bank-pad lane per complex).
    comptime sm_complex_stride = _sm_complex_stride[
        Self.dim,
        Self.geo.amnt_dims,
        warp_size = Int(Self.warp_size),
        block_threads = Int(Self.block_threads),
    ]()
    comptime shared_tile_elems = UInt(Self.dim) * UInt(Self.sm_complex_stride)
    comptime ping_pong_shared_bytes = (
        2
        * Self.shared_tile_elems
        * UInt(size_of[Self.out_dtype]())
    )

    # Pack / multi-element path. Extra lines/block and 1D-1024 ept=2 (512
    # threads, one line) both lost occupancy. Dual-line ept=1 packing (several
    # full FFTs sharing barriers) was much slower than 1-line persistent
    # blocks — default off. Sub-warp GCD packing (16 tpt × N lines) is
    # SMEM-bound on Pascal (~2 resident blocks) and lost to 1-tpt 480.
    comptime can_pack_transforms = False
    comptime target_elems_per_thread = UInt(4)
    comptime threads_per_transform = (
        ceildiv(UInt(Self.dim), Self.target_elems_per_thread) if (
            Self.can_pack_transforms
        ) else UInt(Self.dim)
    )
    comptime elems_per_thread = ceildiv(
        UInt(Self.dim), Self.threads_per_transform
    )
    comptime pack_smem_budget_bytes = Self.ping_pong_shared_bytes * 2
    comptime max_transforms_by_smem = max(
        UInt(1),
        Self.pack_smem_budget_bytes // Self.ping_pong_shared_bytes,
    )
    # Keep packed width near the legacy one-thread-per-sample block size.
    comptime packed_max_block_threads = min(
        Self.max_threads_per_block, Self.warp_size * 4
    )
    comptime max_transforms_by_threads = max(
        UInt(1),
        Self.packed_max_block_threads // Self.threads_per_transform,
    )
    comptime transforms_per_block = (
        max(
            UInt(1),
            min(
                Self.batches,
                min(Self.max_transforms_by_smem, Self.max_transforms_by_threads),
            ),
        ) if Self.can_pack_transforms else UInt(1)
    )
    # Use packed kernel when we pack transforms and/or assign multiple elems
    # per thread (both need the new launch map).
    comptime use_packed_transforms = Self.can_pack_transforms and (
        Self.transforms_per_block > 1 or Self.elems_per_thread > 1
    )
    comptime packed_ping_pong_shared_bytes = (
        Self.ping_pong_shared_bytes * Self.transforms_per_block
    )
    comptime packed_threads_per_block = (
        Self.threads_per_transform * Self.transforms_per_block
    )
    comptime packed_batch_groups = ceildiv(
        Self.batches, Self.transforms_per_block
    )
    comptime packed_thread_batch_size = max(
        UInt(1),
        Self.max_threads_available // Self.packed_threads_per_block,
    )
    comptime packed_batch_size = min(
        Self.packed_batch_groups, Self.packed_thread_batch_size
    )

    # SBRC: coalesced tile of adjacent column-FFT lines in one block.
    # Last spatial axis is always orthogonal for column dims → adjacent lines
    # are 2 scalars apart (one complex), so thread.x = line coalesces.
    comptime sbrc_fast_ortho_len = UInt(
        Self.geo.dims.static_shape[Self.geo.start_dim_idx]
    )
    comptime is_equal_cube_sbrc = _fft_nd_use_column_tiles[
        Self.geo.dims,
        warp_size = Int(Self.warp_size),
        max_thread_block_size = Int(Self.max_threads_per_block),
    ]()
    # Rect-2D / is_column_axis declared above (batch_size + SBRC share them).
    comptime can_sbrc = (
        Self.is_column_axis
        and Self.num_blocks == 1
        and Self.can_run_in_block
        and Self.sbrc_fast_ortho_len > 0
        and (Self.is_equal_cube_sbrc or Self.is_rect_sbrc_2d)
    )
    comptime sbrc_tpt = (
        UInt(80) if (Self.is_rect_sbrc_2d and UInt(Self.dim) == 640) else ceildiv(
            UInt(Self.dim), UInt(4)
        )
    )
    comptime sbrc_ept = ceildiv(UInt(Self.dim), Self.sbrc_tpt)
    comptime sbrc_half_lds = Self.can_sbrc and Self.is_rect_sbrc_2d
    comptime sbrc_line_bytes = (
        UInt(Int(Self.dim) + 1) * UInt(2) * UInt(size_of[Self.out_dtype]())
    )
    comptime sbrc_tiles = UInt(1) if Self.sbrc_half_lds else UInt(2)
    comptime sbrc_smem_budget_bytes = UInt(49152)
    comptime sbrc_max_lines_smem = max(
        UInt(1),
        Self.sbrc_smem_budget_bytes // (Self.sbrc_tiles * Self.sbrc_line_bytes),
    )
    comptime sbrc_max_block_threads = UInt(1024)
    comptime sbrc_max_lines_thr = max(
        UInt(1),
        min(Self.max_threads_per_block, Self.sbrc_max_block_threads)
        // max(Self.sbrc_tpt, UInt(1)),
    )
    comptime sbrc_lines_per_block = (
        UInt(8) if (Self.can_sbrc and Self.is_rect_sbrc_2d) else (
            max(
                UInt(1),
                min(
                    Self.sbrc_fast_ortho_len,
                    min(Self.sbrc_max_lines_smem, Self.sbrc_max_lines_thr),
                ),
            ) if Self.can_sbrc else UInt(1)
        )
    )
    comptime use_sbrc_column = Self.can_sbrc

    # 2D: pack last-axis lines and write transposed → skip standalone back-T.
    # Flag is geo-wide so dim-0 kernels share skip_transpose stage accounting.
    comptime last_axis_dim = Int(
        Self.geo.dims.static_shape[Self.geo.start_dim_idx]
    )
    # Packed fuse-back ~38ms; single-line transpose-store ~24.7ms — both lose to
    # RTRT+32×8 (~18ms). Keep schedule/kernel stubs gated off.
    comptime nd_fuse_back_transpose = False and _fft_nd_fuse_back_transpose[
        Self.geo.dims,
        Self.last_axis_dim,
        shared_mem_per_block_bytes = Int(Self.shared_mem_per_block_bytes),
        warp_size = Int(Self.warp_size),
        max_thread_block_size = Int(Self.max_threads_per_block),
    ]()
    comptime use_fuse_back_row = False
    comptime fuse_back_transpose_store = False
    comptime fuse_back_tpt = min(ceildiv(UInt(Self.dim), UInt(4)), UInt(64))
    comptime fuse_back_ept = ceildiv(UInt(Self.dim), Self.fuse_back_tpt)
    comptime fuse_back_max_L_smem = max(
        UInt(1),
        Self.shared_mem_per_block_bytes // Self.ping_pong_shared_bytes,
    )
    comptime fuse_back_max_L_thr = max(
        UInt(1),
        Self.max_threads_per_block // max(Self.fuse_back_tpt, UInt(1)),
    )
    comptime fuse_back_L = (
        max(
            UInt(2),
            min(
                UInt(2),
                min(
                    UInt(Self.geo.dims.static_shape[0]),
                    min(Self.fuse_back_max_L_smem, Self.fuse_back_max_L_thr),
                ),
            ),
        ) if Self.use_fuse_back_row else UInt(1)
    )
    comptime fuse_back_H = UInt(Self.geo.dims.static_shape[0])
    comptime fuse_back_W = UInt(Self.dim)
    comptime fuse_back_tiles_per_row = ceildiv(Self.fuse_back_H, Self.fuse_back_L)
    comptime fuse_back_num_rows = Self.geo.outer_batches
    comptime fuse_back_num_groups = (
        Self.fuse_back_num_rows * Self.fuse_back_tiles_per_row
    )
    comptime fuse_back_threads = Self.fuse_back_tpt * Self.fuse_back_L
    comptime fuse_back_thread_batch = max(
        UInt(1),
        Self.max_threads_available // max(Self.fuse_back_threads, UInt(1)),
    )
    comptime fuse_back_batch_size = min(
        Self.fuse_back_num_groups, Self.fuse_back_thread_batch
    )
    comptime fuse_back_ping_pong_shared_bytes = (
        Self.ping_pong_shared_bytes * Self.fuse_back_L
    )

    # Bailey four-step column path: correct but slower than RTRT on this DUT
    # (n1=32 ~28ms, n1=64 ~41ms vs ~18ms). Keep gated off.
    comptime can_warp_fs = False and (
        Self.is_column_axis
        and Self.geo.amnt_dims == 2
        and Self.num_blocks == 1
        and Self.can_run_in_block
        and not Self.is_equal_cube_sbrc
        and _fft_nd_use_warp_four_step_column[
            Self.dim,
            Int(Self.sbrc_fast_ortho_len),
            warp_size = Int(Self.warp_size),
        ]()
    )
    # Contiguous Bailey SM FFT-32 (batched CT): correct on 32×96; with closed
    # R=3/5 still ~19.5ms on 640×480 vs ~17.8ms 1-tpt — keep gated.
    comptime can_contig_fs = False and (
        not Self.is_column_axis
        and Self.num_blocks == 1
        and Self.can_run_in_block
        and not Self.is_equal_cube_sbrc
        and not Self.can_warp_fs
        and _fft_nd_use_contig_four_step[
            Self.dim, warp_size = Int(Self.warp_size)
        ]()
    )
    comptime warp_fs_n1 = (
        Int(Self.warp_size) if Self.can_contig_fs else _fft_nd_warp_fs_n1[
            Self.dim, warp_size = Int(Self.warp_size)
        ]()
    )
    comptime warp_fs_n1_lanes = 1
    comptime use_warp_fs_column = Self.can_warp_fs
    comptime use_contig_fs = Self.can_contig_fs
    comptime warp_fs_n2 = (
        Self.dim // Self.warp_fs_n1 if (
            Self.can_warp_fs or Self.can_contig_fs
        ) else 1
    )
    # Contig coop: block = (n1=32, L lines). One line buffer in SMEM.
    comptime contig_fs_line_bytes = (
        UInt(Self.dim) * UInt(2) * UInt(size_of[Self.out_dtype]())
    )
    comptime contig_fs_max_L = max(
        UInt(1),
        Self.shared_mem_per_block_bytes // max(Self.contig_fs_line_bytes, UInt(1)),
    )
    comptime contig_fs_lines_per_block = (
        min(UInt(8), Self.contig_fs_max_L) if Self.can_contig_fs else UInt(
            max(Self.warp_fs_n1, 1)
        )
    )
    # Column FS tiles the fast ortho axis; contig FS tiles the batch lines.
    comptime warp_fs_tiles_per_row = (
        ceildiv(Self.batches, Self.contig_fs_lines_per_block) if Self.can_contig_fs else ceildiv(
            Self.sbrc_fast_ortho_len, UInt(max(Self.warp_fs_n1, 1))
        )
    )
    comptime warp_fs_num_rows = (
        UInt(1) if Self.can_contig_fs else Self.batches // Self.sbrc_fast_ortho_len
    )
    comptime warp_fs_num_groups = Self.warp_fs_num_rows * Self.warp_fs_tiles_per_row
    comptime warp_fs_threads_per_block = (
        UInt(Self.warp_fs_n1) * Self.contig_fs_lines_per_block if Self.can_contig_fs else UInt(
            max(Self.warp_fs_n1 * Self.warp_fs_n1_lanes, 1)
        )
    )
    comptime warp_fs_thread_batch_size = max(
        UInt(1),
        Self.max_threads_available // max(Self.warp_fs_threads_per_block, UInt(1)),
    )
    comptime warp_fs_batch_size = min(
        Self.warp_fs_num_groups, Self.warp_fs_thread_batch_size
    )

    comptime skip_volume_transpose = (
        Self.is_equal_cube_sbrc or Self.can_warp_fs or Self.is_rect_sbrc_2d
    )
    # Strided native store from dim0 (~28ms) lost to separate restore (~18ms).
    comptime fuse_restore_store = False
    comptime sbrc_tiles_per_row = ceildiv(
        Self.sbrc_fast_ortho_len, Self.sbrc_lines_per_block
    )
    comptime sbrc_num_rows = Self.batches // Self.sbrc_fast_ortho_len
    comptime sbrc_num_groups = Self.sbrc_num_rows * Self.sbrc_tiles_per_row
    comptime sbrc_threads_per_block = (
        Self.sbrc_tpt * Self.sbrc_lines_per_block
    )
    comptime sbrc_thread_batch_size = max(
        UInt(1),
        Self.max_threads_available // max(Self.sbrc_threads_per_block, UInt(1)),
    )
    # Half-LDS is SMEM-bound (~1 block/SM). Prefer one tile-group per block so
    # the grid carries parallelism instead of a long serial group loop (each
    # group pays gather/stage/scatter barriers).
    comptime sbrc_batch_size = (
        Self.sbrc_num_groups if Self.sbrc_half_lds else min(
            Self.sbrc_num_groups, Self.sbrc_thread_batch_size
        )
    )
    comptime sbrc_ping_pong_shared_bytes = (
        Self.sbrc_tiles * Self.sbrc_line_bytes * Self.sbrc_lines_per_block
    )

    comptime use_shared_memory = (
        (
            Self.sbrc_ping_pong_shared_bytes <= UInt(49152)
        ) if Self.use_sbrc_column else (
            (
                Self.fuse_back_ping_pong_shared_bytes
                <= Self.shared_mem_per_block_bytes
            ) if Self.use_fuse_back_row else (
                Self.packed_ping_pong_shared_bytes
                <= (
                    Self.shared_mem_per_block_bytes if Self.use_packed_transforms else Self.max_shared_mem_bytes
                )
            )
        )
    )

    comptime use_cluster_sync = not (
        Self.can_run_in_block or Self.can_run_in_warp
    )


def _assert_shared_memory_fits[config: _GPUExecConfig]():
    """Fail at compile time unless the axis fits one-block SM or uses two-upload."""
    comptime uses_four_step = _fft_axis_uses_four_step[
        config.dim,
        config.max_threads_per_block,
        config.geo.amnt_dims,
    ]()
    comptime if not uses_four_step:
        comptime assert config.can_run_in_block and config.use_shared_memory, (
            "FFT dim does not fit one-block shared-memory ping-pong and no "
            "two-upload factorization is available for this length."
        )


def _use_shared_memory_fn[config: _GPUExecConfig, idx: Int]() -> Bool:
    return _GPUExecConfig[
        config.out_dtype,
        config.out_layout_type,
        config.inverse,
        config.bases,
        config.test,
        config.gpu_info,
        config.max_cluster_size,
        config.runtime_twfs,
        idx,
    ].use_shared_memory


@fieldwise_init
struct _GPUPlan[
    out_dtype: DType,
    out_layout_type: TensorLayout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
](Copyable):
    comptime config[dim_idx: Int] = _GPUExecConfig[
        Self.out_dtype,
        Self.out_layout_type,
        Self.inverse,
        Self.bases,
        Self.test,
        Self.gpu_info,
        Self.max_cluster_size,
        Self.runtime_twfs,
        dim_idx,
    ]()

    var twfs_buffer: List[Optional[DeviceBuffer[Self.out_dtype]]]
    var calc_buf: DeviceBuffer[Self.out_dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime assert Self.config[0].threads_per_m > 0, (
            "Unknown number of threads per sm for the given device. "
            "It is needed in order to run the gpu implementation."
        )
        comptime amnt_dims = Self.config[0].geo.amnt_dims
        comptime for dim_idx in range(amnt_dims):
            _assert_shared_memory_fits[Self.config[dim_idx]]()

        comptime out_size = Self.out_layout_type.static_cosize
        self.calc_buf = ctx.enqueue_create_buffer[Self.out_dtype](out_size)

        self.twfs_buffer = {capacity = amnt_dims}
        comptime for dim_idx in range(amnt_dims):
            comptime config = Self.config[dim_idx]
            comptime length = UInt(config.dim)
            comptime if config.inline_twfs or config.runtime_twfs:
                self.twfs_buffer.append(None)
                continue

            var twfs = ctx.enqueue_create_buffer[Self.out_dtype](
                Int(config.dim_size)
            )
            comptime twfs_array = _get_twiddle_factors[
                length, Self.out_dtype, Self.inverse
            ]()
            # FIXME(#5686): replace with this once it's solved
            # ref twfs_array_runtime = global_constant[twfs_array]()
            var twfs_array_runtime = materialize[twfs_array]()
            var ptr = twfs_array_runtime.unsafe_ptr()
            ctx.enqueue_copy(twfs, ptr.unsafe_bitcast[Scalar[Self.out_dtype]]())
            self.twfs_buffer.append(twfs^)


@always_inline
def _intra_something_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    var global_i = UInt(block_dim.x * block_idx.x + thread_idx.x)
    var block_num = Int(block_idx.y)

    comptime total_threads = config.block_threads * config.num_blocks
    comptime intra_ept = config.intra_ept
    comptime intra_tpt = config.block_threads
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]

    comptime base_out_layout = row_major[config.dim, 2]()
    comptime base_x_layout = row_major[config.dim, x_complex_in]()

    # Shared ping-pong: [dim, sm_stride] so complex i is at scalar offset i*stride.
    comptime shared_f_layout = row_major[config.dim, config.sm_complex_stride]()

    var shared_f_lhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)
    var shared_f_rhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)

    comptime skip_volume = config.skip_volume_transpose
    comptime skip_transpose = skip_volume or config.nd_fuse_back_transpose
    comptime fuse_forward_restore = (
        _fft_nd_fuse_forward_restore[config.geo.start_dim_idx]()
        and not skip_volume
    )
    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = skip_transpose,
        skip_forward_transpose_stages = (
            fuse_forward_restore and not skip_transpose
        ),
    ]()

    comptime LhsTile = type_of(shared_f_lhs)
    comptime RhsTile = type_of(shared_f_rhs)
    comptime stage_sync = (
        _fft_cluster_stage_sync if config.use_cluster_sync else _fft_block_stage_sync
    )
    comptime pipeline_type = _FftStockhamPipeline[
        out_dtype, LhsTile, RhsTile, stage_sync
    ]
    var pipeline = pipeline_type(shared_f_lhs, shared_f_rhs)

    comptime use_strided_lines = (
        False and dim_idx < config.geo.start_dim_idx
    )
    comptime axis_stride = _spatial_axis_scalar_stride[
        config.geo.dims, dim_idx
    ]()
    comptime global_stride = axis_stride if use_strided_lines else 2
    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=config.sm_complex_stride,
        global_complex_stride=global_stride,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()
    comptime last_b = stage_plan.stage_count - 1
    comptime last_to_global = config.use_shared_memory
    comptime n_sm_stages = last_b if last_to_global else stage_plan.stage_count
    comptime use_coalesced_gather = False
    comptime last_do_rfft = (
        x_complex_in == 1
        and dim_idx == config.geo.start_dim_idx
        and last_b == 0
    )
    comptime last_cfg = _FFTKernelExecConfig[
        stage_plan.length,
        last_do_rfft,
        stage_plan.ordered_bases[last_b],
        stage_plan.processed_list[last_b],
        inverse,
        stage_plan.ordered_bases,
        stage_plan.inline_twfs,
        runtime_twfs,
        False,
        in_complex_stride = (
            stage_plan.global_complex_stride if last_b == 0 else stage_plan.sm_complex_stride
        ),
        out_complex_stride = (
            Int(config.geo.dims.static_shape[1]) * 2 if config.fuse_restore_store else (
                Int(config.fuse_back_H) * 2 if config.fuse_back_transpose_store else stage_plan.global_complex_stride
            )
        ),
    ]()
    comptime last_write_lhs = stage_plan.last_write_lhs
    comptime force_scatter_to_calc = (
        fuse_forward_restore and dim_idx == 0 and not config.fuse_restore_store
    )
    comptime last_to_calc = force_scatter_to_calc or (
        not stage_plan.write_global_lhs
    )

    comptime batched_iters = max(Int(config.batches // config.batch_size), 1)
    comptime x_stride = Int(config.dim) * x_complex_in
    comptime out_stride = Int(config.dim) * 2
    comptime fuse_H = Int(config.fuse_back_H)
    comptime fuse_W = Int(config.dim)
    comptime restore_D0 = Int(config.geo.dims.static_shape[0])
    comptime restore_W = Int(config.geo.dims.static_shape[1])

    @always_inline
    def _run_batch_at(
        offset: Int,
    ) {
        imm x,
        imm output,
        imm calc_buf,
        mut pipeline,
        imm global_i,
        imm twiddle_factors,
    }:
        var x_off: Int
        var out_off: Int
        var write_off: Int
        comptime if use_strided_lines:
            x_off = _nd_line_base_scalar_offset[
                config.geo.dims, dim_idx, complex_width=x_complex_in
            ](offset)
            out_off = _nd_line_base_scalar_offset[
                config.geo.dims, dim_idx, complex_width=2
            ](offset)
            write_off = out_off
        elif config.fuse_back_transpose_store:
            # Contiguous load from [B,H,W]; store as [B,W,H] with stride H.
            x_off = x_stride * offset
            var b = offset // fuse_H
            var h = offset % fuse_H
            out_off = (b * fuse_W * fuse_H + h) * 2
            write_off = out_off
        elif config.fuse_restore_store:
            # Read contiguous [B,W,H]; write native [B,H,W] with stride W.
            x_off = x_stride * offset
            out_off = out_stride * offset
            var b = offset // restore_W
            var w = offset % restore_W
            write_off = (b * restore_D0 * restore_W + w) * 2
        else:
            x_off = x_stride * offset
            out_off = out_stride * offset
            write_off = out_off
        var base_x = TileTensor(x.ptr + x_off, base_x_layout)
        var base_out = TileTensor(output.ptr + out_off, base_out_layout)
        var base_calc = TileTensor(calc_buf.ptr + out_off, base_out_layout)
        var base_write_out = TileTensor(output.ptr + write_off, base_out_layout)

        @always_inline
        def _run_stages(
            x_in: TileTensor[mut=False, ...],
            dest_global: TileTensor[mut=True, out_dtype, ...],
        ) {mut pipeline, imm global_i, imm twiddle_factors}:
            var payload = pipeline.stage_payload(x_in)
            comptime for stage_b in range(n_sm_stages):
                comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
                comptime stage = _FftStageRouteParams[stage_exec]()
                comptime n_bfly = (
                    stage_exec.config.length // stage_exec.config.base
                )
                comptime use_bfly = _stage_use_reg_bfly_partial[
                    stage_exec.config.base,
                    stage_exec.config.length,
                    intra_tpt,
                ]()
                comptime bfly_even = n_bfly % intra_tpt == 0
                comptime n_work_exact = n_bfly // intra_tpt
                comptime n_work_ceil = (n_bfly + intra_tpt - 1) // intra_tpt
                comptime if use_bfly and bfly_even:
                    comptime for e in range(Int(n_work_exact)):
                        var idx = global_i + UInt(e) * intra_tpt
                        stage.run_stockham_butterfly_once(
                            payload, idx, twiddle_factors
                        )
                elif use_bfly:
                    # Runtime loop: partial cover without unrolling n_work into regs.
                    # n_work_ceil==1 is common (e.g. 480@[8,6,10] tpt=80).
                    comptime if n_work_ceil == 1:
                        if global_i < n_bfly:
                            stage.run_stockham_butterfly_once(
                                payload, global_i, twiddle_factors
                            )
                    else:
                        for e in range(Int(n_work_ceil)):
                            var idx = global_i + UInt(e) * intra_tpt
                            if idx < n_bfly:
                                stage.run_stockham_butterfly_once(
                                    payload, idx, twiddle_factors
                                )
                else:
                    comptime for e in range(Int(intra_ept)):
                        var idx = global_i + UInt(e) * intra_tpt
                        stage.run_elem_per_thread_once(
                            payload, idx, twiddle_factors
                        )
                pipeline.sync_stage()

            comptime if last_to_global:
                comptime n_bfly_last = last_cfg.length // last_cfg.base
                comptime use_bfly_last = _stage_use_reg_bfly_partial[
                    last_cfg.base, last_cfg.length, intra_tpt
                ]()
                comptime bfly_even_last = n_bfly_last % intra_tpt == 0
                comptime n_work_exact_last = n_bfly_last // intra_tpt
                comptime n_work_ceil_last = (
                    n_bfly_last + intra_tpt - 1
                ) // intra_tpt
                comptime if use_bfly_last and bfly_even_last:
                    comptime for e in range(Int(n_work_exact_last)):
                        var idx = global_i + UInt(e) * intra_tpt
                        comptime if last_b == 0:
                            _radix_n_stockham_butterfly_reg[
                                out_dtype, last_cfg
                            ](dest_global, x_in, idx, twiddle_factors)
                        elif last_write_lhs:
                            _radix_n_stockham_butterfly_reg[
                                out_dtype, last_cfg
                            ](
                                dest_global,
                                pipeline.rhs.as_immut(),
                                idx,
                                twiddle_factors,
                            )
                        else:
                            _radix_n_stockham_butterfly_reg[
                                out_dtype, last_cfg
                            ](
                                dest_global,
                                pipeline.lhs.as_immut(),
                                idx,
                                twiddle_factors,
                            )
                elif use_bfly_last:
                    comptime if n_work_ceil_last == 1:
                        if global_i < n_bfly_last:
                            comptime if last_b == 0:
                                _radix_n_stockham_butterfly_reg[
                                    out_dtype, last_cfg
                                ](dest_global, x_in, global_i, twiddle_factors)
                            elif last_write_lhs:
                                _radix_n_stockham_butterfly_reg[
                                    out_dtype, last_cfg
                                ](
                                    dest_global,
                                    pipeline.rhs.as_immut(),
                                    global_i,
                                    twiddle_factors,
                                )
                            else:
                                _radix_n_stockham_butterfly_reg[
                                    out_dtype, last_cfg
                                ](
                                    dest_global,
                                    pipeline.lhs.as_immut(),
                                    global_i,
                                    twiddle_factors,
                                )
                    else:
                        for e in range(Int(n_work_ceil_last)):
                            var idx = global_i + UInt(e) * intra_tpt
                            if idx < n_bfly_last:
                                comptime if last_b == 0:
                                    _radix_n_stockham_butterfly_reg[
                                        out_dtype, last_cfg
                                    ](dest_global, x_in, idx, twiddle_factors)
                                elif last_write_lhs:
                                    _radix_n_stockham_butterfly_reg[
                                        out_dtype, last_cfg
                                    ](
                                        dest_global,
                                        pipeline.rhs.as_immut(),
                                        idx,
                                        twiddle_factors,
                                    )
                                else:
                                    _radix_n_stockham_butterfly_reg[
                                        out_dtype, last_cfg
                                    ](
                                        dest_global,
                                        pipeline.lhs.as_immut(),
                                        idx,
                                        twiddle_factors,
                                    )
                else:
                    comptime for e in range(Int(intra_ept)):
                        var idx = global_i + UInt(e) * intra_tpt
                        comptime if last_b == 0:
                            _radix_n_fft_kernel_elem_per_thread[
                                out_dtype, last_cfg
                            ](dest_global, x_in, idx, twiddle_factors)
                        elif last_write_lhs:
                            _radix_n_fft_kernel_elem_per_thread[
                                out_dtype, last_cfg
                            ](
                                dest_global,
                                pipeline.rhs.as_immut(),
                                idx,
                                twiddle_factors,
                            )
                        else:
                            _radix_n_fft_kernel_elem_per_thread[
                                out_dtype, last_cfg
                            ](
                                dest_global,
                                pipeline.lhs.as_immut(),
                                idx,
                                twiddle_factors,
                            )

        comptime if config.fuse_restore_store:
            # Contiguous read from post-back-T buffer; strided write to native output.
            comptime if stage_plan.write_global_lhs:
                _run_stages(
                    base_calc.as_immut(), base_write_out.as_unsafe_any_origin()
                )
            else:
                _run_stages(
                    base_out.as_immut(), base_write_out.as_unsafe_any_origin()
                )
        elif not config.use_shared_memory:
            _run_stages(base_x.as_immut(), base_out.as_unsafe_any_origin())
        elif stage_plan.input_from_x:
            comptime if last_to_calc:
                _run_stages(base_x.as_immut(), base_calc.as_unsafe_any_origin())
            else:
                _run_stages(base_x.as_immut(), base_out.as_unsafe_any_origin())
        elif stage_plan.write_global_lhs:
            comptime if last_to_calc:
                _run_stages(
                    base_calc.as_immut(), base_calc.as_unsafe_any_origin()
                )
            else:
                _run_stages(
                    base_calc.as_immut(), base_out.as_unsafe_any_origin()
                )
        else:
            comptime if last_to_calc:
                _run_stages(
                    base_out.as_immut(), base_calc.as_unsafe_any_origin()
                )
            else:
                _run_stages(
                    base_out.as_immut(), base_out.as_unsafe_any_origin()
                )

    comptime full_iters = batched_iters * Int(config.batch_size)
    comptime remainder = Int(config.batches) - full_iters

    for i in range(batched_iters):
        _run_batch_at(block_num + i * Int(config.batch_size))
        # Skip barrier only after the last serial batch in this block.
        if i + 1 < batched_iters or remainder > 0:
            pipeline.batch_sync()

    comptime if remainder > 0:
        if block_num < remainder:
            _run_batch_at(full_iters + block_num)


@always_inline
def _packed_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    """One block runs `transforms_per_block` independent length-`dim` FFTs.

    Thread mapping: `thread_idx.x` = lane within the transform (covers
    `elems_per_thread` samples), `thread_idx.y` = transform within the block.
    Grid.y walks batch groups of that width.
    """
    var lane = UInt(thread_idx.x)
    var line = UInt(thread_idx.y)
    var batch_group = UInt(block_idx.y)

    comptime tpb = config.transforms_per_block
    comptime tpt = config.threads_per_transform
    comptime ept = config.elems_per_thread
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]
    comptime base_out_layout = row_major[config.dim, 2]()
    comptime base_x_layout = row_major[config.dim, x_complex_in]()
    comptime line_layout = row_major[config.dim, config.sm_complex_stride]()
    comptime line_elems = Int(config.dim) * config.sm_complex_stride

    comptime shared_f_layout = row_major[
        Int(tpb), config.dim, config.sm_complex_stride
    ]()
    var shared_f_lhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)
    var shared_f_rhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)

    var lhs_line = TileTensor(
        shared_f_lhs.ptr + Int(line) * line_elems, line_layout
    )
    var rhs_line = TileTensor(
        shared_f_rhs.ptr + Int(line) * line_elems, line_layout
    )

    comptime skip_volume = config.skip_volume_transpose
    comptime skip_transpose = skip_volume or config.nd_fuse_back_transpose
    comptime fuse_forward_restore = (
        _fft_nd_fuse_forward_restore[config.geo.start_dim_idx]()
        and not skip_volume
    )
    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = skip_transpose,
        skip_forward_transpose_stages = (
            fuse_forward_restore and not skip_transpose
        ),
    ]()

    comptime LhsTile = type_of(lhs_line)
    comptime RhsTile = type_of(rhs_line)
    comptime stage_sync = _fft_block_stage_sync
    comptime pipeline_type = _FftStockhamPipeline[
        out_dtype, LhsTile, RhsTile, stage_sync
    ]
    var pipeline = pipeline_type(lhs_line, rhs_line)

    comptime use_rc = False
    comptime axis_stride = _spatial_axis_scalar_stride[
        config.geo.dims, dim_idx
    ]()
    comptime global_stride = axis_stride if use_rc else 2
    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=config.sm_complex_stride,
        global_complex_stride=global_stride,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    comptime batched_iters = max(
        config.packed_batch_groups // config.packed_batch_size, 1
    )
    comptime x_stride = Int(config.dim) * x_complex_in
    comptime out_stride = Int(config.dim) * 2

    @always_inline
    def _run_batch_group_at(
        group_base_batch: Int,
    ) {
        imm x,
        imm output,
        imm calc_buf,
        mut pipeline,
        imm lane,
        imm line,
        imm twiddle_factors,
    }:
        var batch_id = group_base_batch + Int(line)
        var active = batch_id < Int(config.batches)

        # Pointers only used when active; dummy bases keep types in scope.
        var base_x = TileTensor(x.ptr, base_x_layout)
        var base_out = TileTensor(output.ptr, base_out_layout)
        var base_calc = TileTensor(calc_buf.ptr, base_out_layout)
        if active:
            var x_off: Int
            var out_off: Int
            comptime if use_rc:
                x_off = _nd_line_base_scalar_offset[
                    config.geo.dims, dim_idx, complex_width=x_complex_in
                ](batch_id)
                out_off = _nd_line_base_scalar_offset[
                    config.geo.dims, dim_idx, complex_width=2
                ](batch_id)
            else:
                x_off = x_stride * batch_id
                out_off = out_stride * batch_id
            base_x = TileTensor(x.ptr + x_off, base_x_layout)
            base_out = TileTensor(output.ptr + out_off, base_out_layout)
            base_calc = TileTensor(calc_buf.ptr + out_off, base_out_layout)

        comptime for stage_b in range(stage_plan.stage_count):
            comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
            comptime stage = _FftStageRouteParams[stage_exec]()
            if active:
                comptime if not config.use_shared_memory:
                    var payload = pipeline.stage_payload(base_x)
                    for e in range(Int(ept)):
                        var sample_i = lane + UInt(e) * tpt
                        if sample_i < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, sample_i, twiddle_factors
                            )
                elif stage_plan.input_from_x:
                    var payload = pipeline.stage_payload(base_x)
                    for e in range(Int(ept)):
                        var sample_i = lane + UInt(e) * tpt
                        if sample_i < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, sample_i, twiddle_factors
                            )
                elif stage_plan.write_global_lhs:
                    var payload = pipeline.stage_payload(base_calc)
                    for e in range(Int(ept)):
                        var sample_i = lane + UInt(e) * tpt
                        if sample_i < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, sample_i, twiddle_factors
                            )
                else:
                    var payload = pipeline.stage_payload(base_out)
                    for e in range(Int(ept)):
                        var sample_i = lane + UInt(e) * tpt
                        if sample_i < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, sample_i, twiddle_factors
                            )
            # Barrier must be uniform across the block (not under `if active`).
            pipeline.sync_stage()

        comptime if config.use_shared_memory:
            if active:
                var stage = pipeline.stage_payload(base_x.as_immut())
                var boundary = _FftDimBoundaryPayload[
                    type_of(stage), type_of(base_out), type_of(base_calc)
                ](stage, base_out, base_calc)
                for e in range(Int(ept)):
                    var sample_i = lane + UInt(e) * tpt
                    if sample_i < UInt(config.dim):
                        boundary.scatter_dim_result[stage_plan](sample_i)

    for i in range(batched_iters):
        _run_batch_group_at(
            Int((batch_group + i * config.packed_batch_size) * tpb)
        )
        pipeline.batch_sync()

    comptime full_iters = batched_iters * config.packed_batch_size
    comptime remainder = config.packed_batch_groups - full_iters

    comptime if remainder > 0:
        if batch_group < remainder:
            _run_batch_group_at(Int((full_iters + batch_group) * tpb))
        pipeline.batch_sync()


@always_inline
def _sbrc_column_gpu_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    """Column FFT with coalesced tile gather/scatter (no full-volume transpose).

    `thread_idx.x` = line within the tile (adjacent lines → contiguous addresses).
    `thread_idx.y` = sample along the FFT axis. Shared is line-major with a
    padded line stride so consecutive-line stores do not alias banks.
    """
    var line_in_tile = UInt(thread_idx.x)
    var sample_lane = UInt(thread_idx.y)
    var group = UInt(block_idx.y)

    comptime L = config.sbrc_lines_per_block
    comptime ept = config.sbrc_ept
    comptime tpt = config.sbrc_tpt
    comptime exact_ept = ept * tpt == UInt(config.dim)
    comptime dim_pad = config.dim + 1
    comptime line_elems = dim_pad * 2
    comptime fast_ortho = config.sbrc_fast_ortho_len
    comptime tiles_per_row = config.sbrc_tiles_per_row
    comptime axis_stride = _spatial_axis_scalar_stride[config.geo.dims, dim_idx]()
    comptime line_adj = 2  # adjacent lines in the fast ortho axis
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]
    comptime line_layout = row_major[config.dim, 2]()

    comptime shared_f_layout = row_major[Int(L), dim_pad, 2]()
    var shared_f_lhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)
    var shared_f_rhs = shared_f_lhs
    comptime if not config.sbrc_half_lds:
        shared_f_rhs = stack_allocation[
            out_dtype, address_space=shared_address_space
        ](shared_f_layout)

    var lhs_line = TileTensor(
        shared_f_lhs.ptr + Int(line_in_tile) * line_elems, line_layout
    )
    var rhs_line = TileTensor(
        shared_f_rhs.ptr + Int(line_in_tile) * line_elems, line_layout
    )

    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = True,
    ]()
    comptime LhsTile = type_of(lhs_line)
    comptime RhsTile = type_of(rhs_line)
    comptime pipeline_type = _FftStockhamPipeline[
        out_dtype, LhsTile, RhsTile, _fft_block_stage_sync
    ]
    var pipeline = pipeline_type(lhs_line, rhs_line)

    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=2,
        global_complex_stride=2,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    comptime gather_to_lhs = (
        True if config.sbrc_half_lds else not stage_plan.write_global_lhs
    )

    comptime batched_iters = max(
        config.sbrc_num_groups // config.sbrc_batch_size, 1
    )

    @always_inline
    def _run_group_at(
        group_id: Int,
    ) {
        imm x,
        imm output,
        imm calc_buf,
        mut pipeline,
        imm line_in_tile,
        imm sample_lane,
        imm twiddle_factors,
        mut shared_f_lhs,
        mut shared_f_rhs,
        imm lhs_line,
        imm rhs_line,
    }:
        var row = group_id // Int(tiles_per_row)
        var tile = group_id % Int(tiles_per_row)
        var line0 = row * Int(fast_ortho) + tile * Int(L)
        var n_active = min(Int(L), Int(fast_ortho) - tile * Int(L))
        var line_active = Int(line_in_tile) < n_active

        var out_line0 = _nd_line_base_scalar_offset[
            config.geo.dims, dim_idx, complex_width=2
        ](line0)

        if line_active:
            comptime if not config.sbrc_half_lds:
                comptime if exact_ept:
                    comptime for e in range(Int(ept)):
                        var sample_i = sample_lane + UInt(e) * tpt
                        var gaddr = (
                            out_line0
                            + Int(sample_i) * axis_stride
                            + Int(line_in_tile) * line_adj
                        )
                        var val: SIMD[out_dtype, 2]
                        comptime if stage_plan.write_global_lhs:
                            val = calc_buf.ptr.load[2](gaddr)
                        else:
                            val = output.ptr.load[2](gaddr)
                        var sm_off = (
                            Int(line_in_tile) * dim_pad + Int(sample_i)
                        ) * 2
                        comptime if gather_to_lhs:
                            shared_f_lhs.ptr.store(sm_off, val)
                        else:
                            shared_f_rhs.ptr.store(sm_off, val)
                else:
                    for e in range(Int(ept)):
                        var sample_i = sample_lane + UInt(e) * tpt
                        if sample_i >= UInt(config.dim):
                            continue
                        var gaddr = (
                            out_line0
                            + Int(sample_i) * axis_stride
                            + Int(line_in_tile) * line_adj
                        )
                        var val: SIMD[out_dtype, 2]
                        comptime if stage_plan.write_global_lhs:
                            val = calc_buf.ptr.load[2](gaddr)
                        else:
                            val = output.ptr.load[2](gaddr)
                        var sm_off = (
                            Int(line_in_tile) * dim_pad + Int(sample_i)
                        ) * 2
                        comptime if gather_to_lhs:
                            shared_f_lhs.ptr.store(sm_off, val)
                        else:
                            shared_f_rhs.ptr.store(sm_off, val)
        comptime if not config.sbrc_half_lds:
            pipeline.sync_stage()

        @always_inline
        def _run_sbrc_stages(
            ref payload: _FftStockhamPayload,
        ) {imm sample_lane, imm twiddle_factors, mut pipeline}:
            comptime for stage_b in range(stage_plan.stage_count):
                comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
                comptime stage = _FftStageRouteParams[stage_exec]()
                comptime n_bfly = (
                    stage_exec.config.length // stage_exec.config.base
                )
                comptime use_bfly = _stage_use_reg_bfly_partial[
                    stage_exec.config.base, stage_exec.config.length, tpt
                ]()
                comptime bfly_even = n_bfly % tpt == 0
                comptime n_work_exact = n_bfly // tpt
                comptime n_work_ceil = (n_bfly + tpt - 1) // tpt
                comptime if use_bfly and bfly_even:
                    comptime for e in range(Int(n_work_exact)):
                        var idx = sample_lane + UInt(e) * tpt
                        stage.run_stockham_butterfly_once(
                            payload, idx, twiddle_factors
                        )
                elif use_bfly:
                    comptime if n_work_ceil == 1:
                        if sample_lane < n_bfly:
                            stage.run_stockham_butterfly_once(
                                payload, sample_lane, twiddle_factors
                            )
                    else:
                        for e in range(Int(n_work_ceil)):
                            var idx = sample_lane + UInt(e) * tpt
                            if idx < n_bfly:
                                stage.run_stockham_butterfly_once(
                                    payload, idx, twiddle_factors
                                )
                else:
                    comptime for e in range(Int(ept)):
                        var idx = sample_lane + UInt(e) * tpt
                        if idx < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, idx, twiddle_factors
                            )
                pipeline.sync_stage()

        if line_active:
            comptime if config.sbrc_half_lds:
                # Build line view from mut SM here (avoids imm capture issues).
                var sm_line = TileTensor(
                    shared_f_lhs.ptr + Int(line_in_tile) * line_elems,
                    line_layout,
                )
                comptime last_b = stage_plan.stage_count - 1
                var g_base = out_line0 + Int(line_in_tile) * line_adj
                # Stage 0: Stockham load from global (skip coalesced gather).
                comptime stage0 = _Fft1dStageExec[stage_plan, 0]()
                comptime cfg0 = stage0.config
                comptime n_bfly0 = cfg0.length // cfg0.base
                comptime use0 = _stage_use_reg_bfly_partial[
                    cfg0.base, cfg0.length, tpt
                ]()
                comptime n_work0 = (n_bfly0 + tpt - 1) // tpt
                comptime assert use0
                comptime assert n_work0 == 1
                comptime cfg0_g = _FFTKernelExecConfig[
                    cfg0.length,
                    cfg0.do_rfft,
                    cfg0.base,
                    cfg0.processed,
                    cfg0.inverse,
                    cfg0.ordered_bases,
                    cfg0.inline_twfs,
                    cfg0.runtime_twfs,
                    cfg0.run_inplace,
                    in_complex_stride = axis_stride,
                    out_complex_stride = 2,
                ]()
                var idx0 = sample_lane if sample_lane < n_bfly0 else UInt(0)
                comptime if stage_plan.write_global_lhs:
                    var g_in = TileTensor(
                        calc_buf.ptr + g_base, line_layout
                    )
                    var xs0 = _stockham_bfly_load_dft[out_dtype, cfg0_g](
                        sm_line.as_immut(),
                        g_in.as_immut(),
                        idx0,
                        twiddle_factors,
                    )
                    # Stage0 reads global only — no barrier before SM store.
                    if sample_lane < n_bfly0:
                        _stockham_bfly_store[out_dtype, cfg0](
                            sm_line, sample_lane, xs0
                        )
                    barrier()
                else:
                    var g_in = TileTensor(
                        output.ptr + g_base, line_layout
                    )
                    var xs0 = _stockham_bfly_load_dft[out_dtype, cfg0_g](
                        sm_line.as_immut(),
                        g_in.as_immut(),
                        idx0,
                        twiddle_factors,
                    )
                    if sample_lane < n_bfly0:
                        _stockham_bfly_store[out_dtype, cfg0](
                            sm_line, sample_lane, xs0
                        )
                    barrier()
                # Mid stages: in-place half-LDS (n_work≤1).
                comptime for stage_b in range(1, last_b):
                    comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
                    comptime cfg = stage_exec.config
                    comptime n_bfly = cfg.length // cfg.base
                    comptime use_bfly = _stage_use_reg_bfly_partial[
                        cfg.base, cfg.length, tpt
                    ]()
                    comptime n_work_ceil = (n_bfly + tpt - 1) // tpt
                    comptime assert use_bfly
                    comptime assert n_work_ceil == 1
                    var idx = (
                        sample_lane if sample_lane < n_bfly else UInt(0)
                    )
                    var xs = _stockham_bfly_load_dft[out_dtype, cfg](
                        sm_line.as_immut(),
                        sm_line.as_immut(),
                        idx,
                        twiddle_factors,
                    )
                    barrier()
                    if sample_lane < n_bfly:
                        _stockham_bfly_store[out_dtype, cfg](
                            sm_line, sample_lane, xs
                        )
                    barrier()
                # Last stage: load SM → regs, store straight to global.
                comptime last_exec = _Fft1dStageExec[stage_plan, last_b]()
                comptime last_cfg = last_exec.config
                comptime last_n_bfly = last_cfg.length // last_cfg.base
                comptime last_use = _stage_use_reg_bfly_partial[
                    last_cfg.base, last_cfg.length, tpt
                ]()
                comptime last_n_work = (last_n_bfly + tpt - 1) // tpt
                comptime assert last_use
                comptime assert last_n_work == 1
                comptime last_cfg_g = _FFTKernelExecConfig[
                    last_cfg.length,
                    last_cfg.do_rfft,
                    last_cfg.base,
                    last_cfg.processed,
                    last_cfg.inverse,
                    last_cfg.ordered_bases,
                    last_cfg.inline_twfs,
                    last_cfg.runtime_twfs,
                    last_cfg.run_inplace,
                    in_complex_stride = last_cfg.in_complex_stride,
                    out_complex_stride = axis_stride,
                ]()
                var last_idx = (
                    sample_lane if sample_lane < last_n_bfly else UInt(0)
                )
                var last_xs = _stockham_bfly_load_dft[out_dtype, last_cfg](
                    sm_line.as_immut(),
                    sm_line.as_immut(),
                    last_idx,
                    twiddle_factors,
                )
                if sample_lane < last_n_bfly:
                    comptime if stage_plan.write_global_lhs:
                        var g_line = TileTensor(
                            output.ptr + g_base, line_layout
                        )
                        _stockham_bfly_store[out_dtype, last_cfg_g](
                            g_line, sample_lane, last_xs
                        )
                    else:
                        var g_line = TileTensor(
                            calc_buf.ptr + g_base, line_layout
                        )
                        _stockham_bfly_store[out_dtype, last_cfg_g](
                            g_line, sample_lane, last_xs
                        )
            elif gather_to_lhs:
                var payload = pipeline.stage_payload(lhs_line.as_immut())
                _run_sbrc_stages(payload)
            else:
                var payload = pipeline.stage_payload(rhs_line.as_immut())
                _run_sbrc_stages(payload)
        else:
            comptime if config.sbrc_half_lds:
                # Stage0: one barrier after SM store. Mids: two each.
                # Last write-through: none.
                barrier()
                comptime for stage_b in range(1, stage_plan.stage_count - 1):
                    barrier()
                    barrier()
            else:
                comptime for stage_b in range(stage_plan.stage_count):
                    pipeline.sync_stage()

        # Half-LDS last-stage write-through already hit global; skip scatter.
        comptime if not config.sbrc_half_lds:
            if line_active:
                comptime if exact_ept:
                    comptime for e in range(Int(ept)):
                        var sample_i = sample_lane + UInt(e) * tpt
                        var sm_off = (
                            Int(line_in_tile) * dim_pad + Int(sample_i)
                        ) * 2
                        var c_num: SIMD[out_dtype, 2]
                        comptime if stage_plan.last_write_lhs:
                            c_num = shared_f_lhs.ptr.load[2](sm_off)
                        else:
                            c_num = shared_f_rhs.ptr.load[2](sm_off)
                        var saddr = (
                            out_line0
                            + Int(sample_i) * axis_stride
                            + Int(line_in_tile) * line_adj
                        )
                        comptime if stage_plan.write_global_lhs:
                            output.ptr.store(saddr, c_num)
                        else:
                            calc_buf.ptr.store(saddr, c_num)
                else:
                    for e in range(Int(ept)):
                        var sample_i = sample_lane + UInt(e) * tpt
                        if sample_i >= UInt(config.dim):
                            continue
                        var sm_off = (
                            Int(line_in_tile) * dim_pad + Int(sample_i)
                        ) * 2
                        var c_num: SIMD[out_dtype, 2]
                        comptime if stage_plan.last_write_lhs:
                            c_num = shared_f_lhs.ptr.load[2](sm_off)
                        else:
                            c_num = shared_f_rhs.ptr.load[2](sm_off)
                        var saddr = (
                            out_line0
                            + Int(sample_i) * axis_stride
                            + Int(line_in_tile) * line_adj
                        )
                        comptime if stage_plan.write_global_lhs:
                            output.ptr.store(saddr, c_num)
                        else:
                            calc_buf.ptr.store(saddr, c_num)
            comptime if batched_iters > 1:
                pipeline.sync_stage()
        elif batched_iters > 1:
            pipeline.sync_stage()

    for i in range(batched_iters):
        _run_group_at(Int(group + i * config.sbrc_batch_size))

    comptime full_iters = batched_iters * config.sbrc_batch_size
    comptime remainder = config.sbrc_num_groups - full_iters
    comptime if remainder > 0:
        if group < remainder:
            _run_group_at(Int(full_iters + group))
        pipeline.sync_stage()


@always_inline
def _fuse_back_row_gpu_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    """Last-axis FFT of L packed rows; store transposed to skip standalone back-T.

    `thread_idx.x` = line in the tile, `thread_idx.y` = sample along W.
    Transposed stores `((b*W+w)*H+h)*2` are coalesced across consecutive lines.
    """
    var line_in_tile = UInt(thread_idx.x)
    var sample_lane = UInt(thread_idx.y)
    var group = UInt(block_idx.y)

    comptime L = config.fuse_back_L
    comptime ept = config.fuse_back_ept
    comptime tpt = config.fuse_back_tpt
    comptime H = config.fuse_back_H
    comptime W = config.fuse_back_W
    comptime tiles_per_row = config.fuse_back_tiles_per_row
    comptime sm_stride = config.sm_complex_stride
    comptime line_elems = Int(config.dim) * sm_stride
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]
    comptime line_layout = row_major[config.dim, sm_stride]()

    comptime shared_f_layout = row_major[Int(L), config.dim, sm_stride]()
    var shared_f_lhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)
    var shared_f_rhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)

    var lhs_line = TileTensor(
        shared_f_lhs.ptr + Int(line_in_tile) * line_elems, line_layout
    )
    var rhs_line = TileTensor(
        shared_f_rhs.ptr + Int(line_in_tile) * line_elems, line_layout
    )

    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = True,
    ]()
    comptime LhsTile = type_of(lhs_line)
    comptime RhsTile = type_of(rhs_line)
    comptime pipeline_type = _FftStockhamPipeline[
        out_dtype, LhsTile, RhsTile, _fft_block_stage_sync
    ]
    var pipeline = pipeline_type(lhs_line, rhs_line)

    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=sm_stride,
        global_complex_stride=2,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    comptime gather_to_lhs = not stage_plan.write_global_lhs
    comptime batched_iters = max(
        config.fuse_back_num_groups // config.fuse_back_batch_size, 1
    )

    @always_inline
    def _run_group_at(
        group_id: Int,
    ) {
        imm x,
        imm output,
        imm calc_buf,
        mut pipeline,
        imm line_in_tile,
        imm sample_lane,
        imm twiddle_factors,
        mut shared_f_lhs,
        mut shared_f_rhs,
        imm lhs_line,
        imm rhs_line,
    }:
        var b = group_id // Int(tiles_per_row)
        var tile = group_id % Int(tiles_per_row)
        var h0 = tile * Int(L)
        var n_active = min(Int(L), Int(H) - h0)
        var line_active = Int(line_in_tile) < n_active
        var h = h0 + Int(line_in_tile)

        # Gather native [B,H,W] rows into SM (strided across lines).
        if line_active:
            for e in range(Int(ept)):
                var sample_i = sample_lane + UInt(e) * tpt
                if sample_i < UInt(W):
                    var gaddr = (
                        (b * Int(H) + h) * Int(W) + Int(sample_i)
                    ) * x_complex_in
                    var val: SIMD[out_dtype, 2]
                    comptime if stage_plan.input_from_x:
                        comptime if x_complex_in == 1:
                            var re = x.ptr.load(gaddr).cast[out_dtype]()
                            val = SIMD[out_dtype, 2](re, 0)
                        else:
                            val = x.ptr.load[2](gaddr).cast[out_dtype]()
                    elif stage_plan.write_global_lhs:
                        val = calc_buf.ptr.load[2](gaddr)
                    else:
                        val = output.ptr.load[2](gaddr)
                    var sm_off = (
                        Int(line_in_tile) * Int(config.dim) + Int(sample_i)
                    ) * sm_stride
                    comptime if gather_to_lhs:
                        shared_f_lhs.ptr.store(sm_off, val)
                    else:
                        shared_f_rhs.ptr.store(sm_off, val)
        pipeline.sync_stage()

        @always_inline
        def _run_fuse_stages(
            ref payload: _FftStockhamPayload,
        ) {imm sample_lane, imm twiddle_factors, mut pipeline}:
            comptime for stage_b in range(stage_plan.stage_count):
                comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
                comptime stage = _FftStageRouteParams[stage_exec]()
                comptime n_bfly = (
                    stage_exec.config.length // stage_exec.config.base
                )
                comptime use_bfly = _stage_use_reg_bfly[
                    stage_exec.config.base, stage_exec.config.length, tpt
                ]()
                comptime n_work = n_bfly // tpt if use_bfly else ept
                comptime for e in range(Int(n_work)):
                    var idx = sample_lane + UInt(e) * tpt
                    comptime if use_bfly:
                        stage.run_stockham_butterfly_once(
                            payload, idx, twiddle_factors
                        )
                    else:
                        if idx < UInt(config.dim):
                            stage.run_elem_per_thread_once(
                                payload, idx, twiddle_factors
                            )
                pipeline.sync_stage()

        if line_active:
            comptime if gather_to_lhs:
                var payload = pipeline.stage_payload(lhs_line.as_immut())
                _run_fuse_stages(payload)
            else:
                var payload = pipeline.stage_payload(rhs_line.as_immut())
                _run_fuse_stages(payload)
        else:
            comptime for stage_b in range(stage_plan.stage_count):
                pipeline.sync_stage()

        # Scatter transposed [B,W,H] — consecutive lines coalesce.
        if line_active:
            for e in range(Int(ept)):
                var sample_i = sample_lane + UInt(e) * tpt
                if sample_i < UInt(W):
                    var sm_off = (
                        Int(line_in_tile) * Int(config.dim) + Int(sample_i)
                    ) * sm_stride
                    var c_num: SIMD[out_dtype, 2]
                    comptime if stage_plan.last_write_lhs:
                        c_num = shared_f_lhs.ptr.load[2](sm_off)
                    else:
                        c_num = shared_f_rhs.ptr.load[2](sm_off)
                    var saddr = (
                        (b * Int(W) + Int(sample_i)) * Int(H) + h
                    ) * 2
                    comptime if stage_plan.write_global_lhs:
                        output.ptr.store(saddr, c_num)
                    else:
                        calc_buf.ptr.store(saddr, c_num)
        pipeline.sync_stage()

    for i in range(batched_iters):
        _run_group_at(Int(group + i * config.fuse_back_batch_size))

    comptime full_iters = batched_iters * config.fuse_back_batch_size
    comptime remainder = config.fuse_back_num_groups - full_iters
    comptime if remainder > 0:
        if group < remainder:
            _run_group_at(Int(full_iters + group))
        pipeline.sync_stage()


def _contig_coop_fs_gpu_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    """Contiguous Bailey `n1×n2` with SM FFT-n1 (32 thr) + register FFT-n2.

    FFT-32 is in-place CT on the line in shared memory. Each CT stage loops
    over all `n2` columns then hits one block barrier (~5 barriers total).
    """
    _ = twiddle_factors
    _ = runtime_twfs
    _ = bases

    var lane = UInt(thread_idx.x)
    var line_in_tile = UInt(thread_idx.y)
    var group = UInt(block_idx.y)

    comptime n1 = config.warp_fs_n1
    comptime n2 = config.warp_fs_n2
    comptime N = config.dim
    comptime L = Int(config.contig_fs_lines_per_block)
    comptime tiles_per_row = config.warp_fs_tiles_per_row
    comptime Co = ComplexScalar[out_dtype]
    comptime inv_scale = (1.0 / Float64(N)).cast[out_dtype]()
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]

    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = config.skip_volume_transpose,
        skip_forward_transpose_stages = (
            _fft_nd_fuse_forward_restore[config.geo.start_dim_idx]()
            and not config.skip_volume_transpose
        ),
    ]()
    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=2,
        global_complex_stride=2,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    comptime line_layout = row_major[L, N, 2]()
    var sm_line = stack_allocation[
        out_dtype, address_space=AddressSpace.SHARED
    ](line_layout)

    comptime batched_iters = max(
        config.warp_fs_num_groups // config.warp_fs_batch_size, 1
    )

    @always_inline
    def _run_group_at(
        group_id: Int,
    ) {
        imm x,
        imm output,
        imm calc_buf,
        imm lane,
        imm line_in_tile,
        mut sm_line,
    }:
        var tile = group_id % Int(tiles_per_row)
        var line0 = tile * L
        var n_active = min(L, Int(config.batches) - line0)
        var line_active = Int(line_in_tile) < n_active

        var line_base = 0
        var line_sm = sm_line.ptr
        if line_active:
            line_base = (line0 + Int(line_in_tile)) * Int(N) * 2
            line_sm = sm_line.ptr + Int(line_in_tile) * Int(N) * 2
            for e in range(n2):
                var idx = Int(lane) * n2 + e
                if idx < Int(N):
                    var gaddr = line_base + idx * 2
                    var vv: SIMD[out_dtype, 2]
                    comptime if stage_plan.input_from_x:
                        comptime if x_complex_in == 1:
                            var re = x.ptr.load(
                                (line0 + Int(line_in_tile)) * Int(N) + idx
                            ).cast[out_dtype]()
                            vv = SIMD[out_dtype, 2](re, 0)
                        else:
                            vv = x.ptr.load[2](gaddr).cast[out_dtype]()
                    elif stage_plan.write_global_lhs:
                        vv = calc_buf.ptr.load[2](gaddr)
                    else:
                        vv = output.ptr.load[2](gaddr)
                    line_sm.store(idx * 2, vv)
        barrier()

        # Bit-reverse rows into registers then write back (all n2 columns).
        var row = Array[Co, n2](fill=Co(0, 0))
        if line_active:
            var br = _bit_reverse[5](Int(lane))
            comptime for k in range(n2):
                var pv = line_sm.load[2]((br * n2 + k) * 2)
                row[k] = Co(pv[0], pv[1])
        barrier()
        if line_active:
            comptime for k in range(n2):
                line_sm.store(
                    (Int(lane) * n2 + k) * 2,
                    SIMD[out_dtype, 2](row[k].re, row[k].im),
                )
        barrier()

        # In-place CT: even lanes own each pair; batch all columns per stage.
        comptime for s in range(5):
            comptime mh = 1 << s
            if line_active and (Int(lane) & mh) == 0:
                var partner = Int(lane) ^ mh
                var j_lane = Int(lane) & (mh - 1)
                var wt = _get_twiddle_factor[
                    out_dtype, inverse=inverse, N = UInt(mh * 2)
                ](UInt(j_lane))
                comptime for k in range(n2):
                    var u = line_sm.load[2]((Int(lane) * n2 + k) * 2)
                    var v = line_sm.load[2]((partner * n2 + k) * 2)
                    var uc = Co(u[0], u[1])
                    var vc = Co(v[0], v[1])
                    var t = wt * vc
                    var sum = uc + t
                    var diff = uc - t
                    line_sm.store(
                        (Int(lane) * n2 + k) * 2,
                        SIMD[out_dtype, 2](sum.re, sum.im),
                    )
                    line_sm.store(
                        (partner * n2 + k) * 2,
                        SIMD[out_dtype, 2](diff.re, diff.im),
                    )
            barrier()

        # Twiddle, length-n2 FFT per row, write transposed.
        if line_active:
            comptime for k in range(n2):
                var pv = line_sm.load[2]((Int(lane) * n2 + k) * 2)
                var v32 = Co(pv[0], pv[1])
                if Int(lane) > 0 and k > 0:
                    var w = _get_twiddle_factor[
                        out_dtype, inverse=inverse, N = UInt(N)
                    ](UInt(Int(lane) * k))
                    v32 = w * v32
                row[k] = v32
            _fft_len_r[out_dtype, n2, inverse](row)
            var dst = (
                output.ptr if stage_plan.write_global_lhs else calc_buf.ptr
            )
            comptime for k in range(n2):
                var yk = row[k]
                comptime if inverse:
                    yk = yk * inv_scale
                var gaddr = line_base + (k * n1 + Int(lane)) * 2
                dst.store(gaddr, SIMD[out_dtype, 2](yk.re, yk.im))

    for i in range(batched_iters):
        _run_group_at(Int(group + i * config.warp_fs_batch_size))

    comptime full_iters = batched_iters * config.warp_fs_batch_size
    comptime remainder = config.warp_fs_num_groups - full_iters
    comptime if remainder > 0:
        if group < remainder:
            _run_group_at(Int(full_iters + group))


@always_inline
def _warp_fs_column_gpu_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    twf_layout_type: TensorLayout,
    twf_origin: ImmOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin],
    x: TileTensor[in_dtype, in_layout_type, in_origin],
    twiddle_factors: TileTensor[out_dtype, twf_layout_type, twf_origin],
    calc_buf: TileTensor[out_dtype, out_layout_type, calc_buf_origin],
):
    """Column FFT via Bailey four-step; warp×warp SM tiles for coalescing.

    Block is `(n1, n1)`: `thread_idx.x` = line, `thread_idx.y` = sample lane.
    Pass 1 streams `k` through an `n1×n1` shared tile; pass 2 streams `j`
    through an `n1×n2` tile (n2 ≤ n1). Final writes go to `output`.
    """
    _ = x
    _ = twiddle_factors
    _ = runtime_twfs
    _ = bases

    var line_in_tile = UInt(thread_idx.x)
    var lane = UInt(thread_idx.y)
    var group = UInt(block_idx.y)

    comptime n1 = config.warp_fs_n1
    comptime n2 = config.warp_fs_n2
    comptime N = config.dim
    comptime fast_ortho = config.sbrc_fast_ortho_len
    comptime tiles_per_row = config.warp_fs_tiles_per_row
    comptime axis_stride = _spatial_axis_scalar_stride[config.geo.dims, dim_idx]()
    comptime line_adj = 2
    comptime Co = ComplexScalar[out_dtype]
    comptime inv_scale = (1.0 / Float64(N)).cast[out_dtype]()
    comptime n1_pad = n1 + 1

    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
        skip_transpose_stages = True,
    ]()
    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
        sm_complex_stride=2,
        global_complex_stride=2,
    ]()
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    # Ping-pong for length-n1 Stockham; reused as n1×n2 buffer in pass 2.
    comptime sm_layout = row_major[n1, n1_pad, 2]()
    var sm_lhs = stack_allocation[out_dtype, address_space=AddressSpace.SHARED](
        sm_layout
    )
    var sm_rhs = stack_allocation[out_dtype, address_space=AddressSpace.SHARED](
        sm_layout
    )

    comptime batched_iters = max(
        config.warp_fs_num_groups // config.warp_fs_batch_size, 1
    )

    comptime SmTile = type_of(sm_lhs)

    @always_inline
    def _sm_fft_pow2_lane(
        mut lhs: SmTile,
        mut rhs: SmTile,
        line: UInt,
        lane: UInt,
        line_active: Bool,
    ):
        """In-place length-n1 Stockham on one shared line (pow2 n1)."""
        comptime LOG = 5  # n1 == 32
        comptime for s in range(1, LOG + 1):
            comptime m = 1 << s
            comptime mh = m // 2
            comptime R = 2
            var write_rhs = (s % 2) == 1
            if line_active:
                var b = lane
                if b < UInt(n1 // R):
                    var k0 = (b % UInt(mh)) + (b // UInt(mh)) * UInt(m)
                    var i0 = Int(line) * n1_pad + Int(k0)
                    var i1 = Int(line) * n1_pad + Int(k0) + mh
                    var src = lhs.ptr if write_rhs else rhs.ptr
                    # After first stage read from lhs (gather); alternate.
                    comptime if s == 1:
                        src = lhs.ptr
                    var u = src.load[2](i0 * 2)
                    var v = src.load[2](i1 * 2)
                    var w = _get_twiddle_factor[
                        out_dtype, inverse=inverse, N = UInt(m)
                    ](UInt(b % UInt(mh)))
                    var vr = v[0] * w.re - v[1] * w.im
                    var vi = v[0] * w.im + v[1] * w.re
                    var dst = rhs.ptr if write_rhs else lhs.ptr
                    dst.store(
                        i0 * 2,
                        SIMD[out_dtype, 2](u[0] + vr, u[1] + vi),
                    )
                    dst.store(
                        i1 * 2,
                        SIMD[out_dtype, 2](u[0] - vr, u[1] - vi),
                    )
            barrier()

    @always_inline
    def _run_group_at(
        group_id: Int,
    ) {
        imm output,
        imm calc_buf,
        imm line_in_tile,
        imm lane,
        mut sm_lhs,
        mut sm_rhs,
    }:
        var row = group_id // Int(tiles_per_row)
        var tile = group_id % Int(tiles_per_row)
        var line0 = row * Int(fast_ortho) + tile * n1
        var n_active = min(n1, Int(fast_ortho) - tile * n1)
        var line_active = Int(line_in_tile) < n_active

        var line_base = _nd_line_base_scalar_offset[
            config.geo.dims, dim_idx, complex_width=2
        ](line0)

        var src = calc_buf.ptr if stage_plan.write_global_lhs else output.ptr
        var scratch = calc_buf.ptr
        var dst = output.ptr

        # Pass 1: for each k, gather n1 samples → SM, FFT32, twiddle, scatter.
        for k in range(n2):
            if line_active and lane < UInt(n1):
                var gaddr = (
                    line_base
                    + (Int(lane) * n2 + k) * axis_stride
                    + Int(line_in_tile) * line_adj
                )
                var val = src.load[2](gaddr)
                sm_lhs.ptr.store(
                    (Int(line_in_tile) * n1_pad + Int(lane)) * 2, val
                )
            barrier()

            # Decimation-in-frequency CT in shared (n1 threads cooperate).
            comptime LOG = 5
            comptime for s in range(LOG):
                comptime mh = 1 << s
                comptime m = mh << 1
                if line_active and lane < UInt(n1 // 2):
                    var half = UInt(n1 // 2)
                    var b = lane
                    var k0 = (b % UInt(mh)) + (b // UInt(mh)) * UInt(m)
                    var idx0 = Int(line_in_tile) * n1_pad + Int(k0)
                    var idx1 = idx0 + mh
                    var src_p = sm_lhs.ptr if (s % 2 == 0) else sm_rhs.ptr
                    var dst_p = sm_rhs.ptr if (s % 2 == 0) else sm_lhs.ptr
                    var u = src_p.load[2](idx0 * 2)
                    var v = src_p.load[2](idx1 * 2)
                    var w = _get_twiddle_factor[
                        out_dtype, inverse=inverse, N = UInt(m)
                    ](UInt(b % UInt(mh)))
                    var tr = v[0] * w.re - v[1] * w.im
                    var ti = v[0] * w.im + v[1] * w.re
                    dst_p.store(
                        idx0 * 2,
                        SIMD[out_dtype, 2](u[0] + tr, u[1] + ti),
                    )
                    dst_p.store(
                        idx1 * 2,
                        SIMD[out_dtype, 2](u[0] - tr, u[1] - ti),
                    )
                barrier()

            comptime last_even = (LOG % 2 == 0)
            var sm_out = sm_lhs.ptr if last_even else sm_rhs.ptr
            if line_active and lane < UInt(n1):
                var j = Int(lane)
                var c = sm_out.load[2](
                    (Int(line_in_tile) * n1_pad + j) * 2
                )
                var cj = Co(c[0], c[1])
                if j > 0 and k > 0:
                    var w = _get_twiddle_factor[
                        out_dtype, inverse=inverse, N = UInt(N)
                    ](UInt(j * k))
                    cj = w * cj
                var gaddr = (
                    line_base
                    + (j * n2 + k) * axis_stride
                    + Int(line_in_tile) * line_adj
                )
                scratch.store(
                    gaddr, SIMD[out_dtype, 2](cj.re, cj.im)
                )
            barrier()

        # Pass 2: for each j, load n2 samples, DFT/factored FFT, scatter.
        for j in range(n1):
            if line_active and lane < UInt(n2):
                var gaddr = (
                    line_base
                    + (j * n2 + Int(lane)) * axis_stride
                    + Int(line_in_tile) * line_adj
                )
                sm_lhs.ptr.store(
                    (Int(line_in_tile) * n1_pad + Int(lane)) * 2,
                    scratch.load[2](gaddr),
                )
            barrier()

            if line_active and lane == 0:
                var ys = Array[Co, n2](fill=Co(0, 0))
                for k in range(n2):
                    var v = sm_lhs.ptr.load[2](
                        (Int(line_in_tile) * n1_pad + k) * 2
                    )
                    ys[k] = Co(v[0], v[1])
                _fft_len_r[out_dtype, n2, inverse](ys)
                for k in range(n2):
                    var yk = ys[k]
                    comptime if inverse:
                        yk = yk * inv_scale
                    sm_rhs.ptr.store(
                        (Int(line_in_tile) * n1_pad + k) * 2,
                        SIMD[out_dtype, 2](yk.re, yk.im),
                    )
            barrier()

            if line_active and lane < UInt(n2):
                var k = Int(lane)
                var gaddr = (
                    line_base
                    + (k * n1 + j) * axis_stride
                    + Int(line_in_tile) * line_adj
                )
                dst.store(
                    gaddr,
                    sm_rhs.ptr.load[2](
                        (Int(line_in_tile) * n1_pad + k) * 2
                    ),
                )
            barrier()

    for i in range(batched_iters):
        _run_group_at(Int(group + i * config.warp_fs_batch_size))

    comptime full_iters = batched_iters * config.warp_fs_batch_size
    comptime remainder = config.warp_fs_num_groups - full_iters
    comptime if remainder > 0:
        if group < remainder:
            _run_group_at(Int(full_iters + group))


def _run_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    inverse: Bool,
    bases: List[List[UInt]],
    runtime_twfs: Bool,
    max_cluster_size: UInt,
    //,
    *,
    test: Optional[_GPUTest] = None,
](
    output: TileTensor[mut=True, out_dtype, out_layout_type, ...],
    x: TileTensor[mut=False, in_dtype, in_layout_type, ...],
    ctx: DeviceContext,
    plan: _GPUPlan[
        out_dtype,
        out_layout_type,
        inverse,
        bases,
        test,
        ctx.default_device_info,
        max_cluster_size=max_cluster_size,
        runtime_twfs=runtime_twfs,
    ],
) raises:
    comptime assert (
        out_dtype.is_floating_point()
    ), "out_dtype must be floating point"
    comptime assert (
        has_accelerator()
    ), "The non-cpu implementation is for GPU only"

    var calc_buf = TileTensor(
        ptr=plan.calc_buf.unsafe_ptr().unsafe_mut_cast[True](),
        layout=output.layout,
    )

    @always_inline
    @__parameter
    def _schedule_four_step[dim_idx: Int]() raises:
        """Column FFT (via T) → twiddle → row FFT → T to natural order.

        Specialized to 1D C2C with N=N1×N2. Row-first four-step does not
        match a length-N DFT; column-first plus a final transpose does.
        """
        comptime config = plan.config[dim_idx]
        comptime factors = _fft_axis_four_step_factors[
            config.dim, config.max_threads_per_block
        ]()
        comptime N1 = factors[0]
        comptime N2 = factors[1]
        comptime B = Int(config.batches)
        comptime xw = Int(in_layout_type.static_shape[in_layout_type.rank - 1])
        comptime assert xw == 2, "two-upload path is C2C only"

        comptime layout_mat_v = row_major[B, N1, N2, 2]()
        comptime layout_mat_t_v = row_major[B, N2, N1, 2]()
        comptime layout_row_v = row_major[B * N1, N2, 2]()
        comptime layout_col_v = row_major[B * N2, N1, 2]()
        comptime layout_mat = type_of(layout_mat_v)
        comptime layout_mat_t = type_of(layout_mat_t_v)
        comptime layout_row = type_of(layout_row_v)
        comptime layout_col = type_of(layout_col_v)
        comptime bases_row: List[List[UInt]] = [_estimate_length_bases[N2]()]
        comptime bases_col: List[List[UInt]] = [_estimate_length_bases[N1]()]
        comptime cfg_col = _GPUExecConfig[
            out_dtype,
            layout_col,
            inverse,
            bases_col,
            test,
            ctx.default_device_info,
            max_cluster_size,
            True,
            0,
        ]()
        comptime cfg_row = _GPUExecConfig[
            out_dtype,
            layout_row,
            inverse,
            bases_row,
            test,
            ctx.default_device_info,
            max_cluster_size,
            True,
            0,
        ]()
        comptime twf_col = row_major[cfg_col.dim, 2]()
        comptime twf_row = row_major[cfg_row.dim, 2]()
        comptime grid_col = (Int(cfg_col.num_blocks), Int(cfg_col.batch_size))
        comptime grid_row = (Int(cfg_row.num_blocks), Int(cfg_row.batch_size))
        comptime tmax = config.max_threads_available
        comptime tnum = UInt(ceildiv(N1, 32) * ceildiv(N2, 32) * 32 * 32)
        comptime fair = max(UInt(1), tmax // max(tnum, UInt(1)))
        comptime sched_b = min(
            UInt(B), min(UInt(65535), max(fair * 128, UInt(512)))
        )

        var mat_out = TileTensor(ptr=output.ptr, layout=layout_mat_v)
        var mat_calc = TileTensor(ptr=calc_buf.ptr, layout=layout_mat_v)
        var matt_out = TileTensor(ptr=output.ptr, layout=layout_mat_t_v)
        var matt_calc = TileTensor(ptr=calc_buf.ptr, layout=layout_mat_t_v)

        # Copy input into calc, then T so length-N1 columns are contiguous.
        _enqueue_buffer_copy_n[
            out_dtype, in_dtype, out_layout_type, in_layout_type
        ](ctx, calc_buf, x)
        _enqueue_matrix_transpose[
            out_dtype, layout_mat, scheduled_batches=sched_b
        ](ctx, mat_out, mat_calc.as_immut())

        var col_in = TileTensor(
            ptr=output.ptr.mut_cast[False]().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            layout=layout_col_v,
        )
        var tw_col = stack_allocation[out_dtype](twf_col).as_immut()
        comptime fn_col = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
            in_dtype=out_dtype,
            out_dtype=out_dtype,
            in_layout_type=layout_col,
            out_layout_type=layout_col,
            in_origin=ImmUntrackedOrigin,
            out_origin=calc_buf.origin,
            twf_layout_type=type_of(twf_col),
            twf_origin=tw_col.origin,
            calc_buf_origin=output.origin,
            inverse=inverse,
            bases=bases_col,
            config=cfg_col,
            shared_address_space = AddressSpace.SHARED,
            runtime_twfs=True,
            dim_idx=0,
        ]
        ctx.enqueue_function[fn_col](
            TileTensor(ptr=calc_buf.ptr, layout=layout_col_v),
            col_in.as_immut(),
            tw_col,
            TileTensor(ptr=output.ptr, layout=layout_col_v),
            grid_dim=grid_col,
            block_dim=cfg_col.block_threads,
        )

        _enqueue_matrix_transpose[
            out_dtype, layout_mat_t, scheduled_batches=sched_b
        ](ctx, matt_out, matt_calc.as_immut())

        _enqueue_four_step_twiddle[
            out_dtype, layout_mat, inverse=inverse, N1=N1, N2=N2
        ](ctx, mat_out)

        var row_in = TileTensor(
            ptr=output.ptr.mut_cast[False]().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            layout=layout_row_v,
        )
        var tw_row = stack_allocation[out_dtype](twf_row).as_immut()
        comptime fn_row = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
            in_dtype=out_dtype,
            out_dtype=out_dtype,
            in_layout_type=layout_row,
            out_layout_type=layout_row,
            in_origin=ImmUntrackedOrigin,
            out_origin=calc_buf.origin,
            twf_layout_type=type_of(twf_row),
            twf_origin=tw_row.origin,
            calc_buf_origin=output.origin,
            inverse=inverse,
            bases=bases_row,
            config=cfg_row,
            shared_address_space = AddressSpace.SHARED,
            runtime_twfs=True,
            dim_idx=0,
        ]
        ctx.enqueue_function[fn_row](
            TileTensor(ptr=calc_buf.ptr, layout=layout_row_v),
            row_in.as_immut(),
            tw_row,
            TileTensor(ptr=output.ptr, layout=layout_row_v),
            grid_dim=grid_row,
            block_dim=cfg_row.block_threads,
        )

        _enqueue_matrix_transpose[
            out_dtype, layout_mat, scheduled_batches=sched_b
        ](ctx, mat_out, mat_calc.as_immut())

    @always_inline
    @__parameter
    def _schedule_run[dim_idx: Int]() raises:
        comptime config = plan.config[dim_idx]
        comptime use_strided_lines = (
            False and dim_idx < config.geo.start_dim_idx
        )
        comptime use_four_step = _fft_axis_uses_four_step[
            config.dim,
            config.max_threads_per_block,
            config.geo.amnt_dims,
        ]()
        comptime if use_four_step:
            _schedule_four_step[dim_idx]()
        else:
            comptime address_space = AddressSpace.SHARED if (
                config.can_run_in_block or config.can_run_in_warp
            ) else AddressSpace.SHARED_CLUSTER

            comptime twf_layout = row_major[config.dim, 2]()
            var twiddle_factors: TileTensor[
                mut=False, out_dtype, type_of(twf_layout), ImmUntrackedOrigin
            ]
            comptime if not (config.inline_twfs or config.runtime_twfs):
                twiddle_factors = TileTensor(
                    plan.twfs_buffer.unsafe_get(dim_idx)
                    .value()
                    .unsafe_ptr()
                    .mut_cast[False]()
                    .unsafe_origin_cast[ImmUntrackedOrigin](),
                    twf_layout,
                )
            else:
                twiddle_factors = stack_allocation[out_dtype](
                    twf_layout
                ).as_immut()

            comptime run_cluster = config.can_run_in_block_cluster and (
                config.num_blocks > 1
            )

            comptime if config.use_contig_fs:
                comptime wfs_grid = (1, Int(config.warp_fs_batch_size))
                comptime wfs_block = (
                    Int(config.warp_fs_n1),
                    Int(config.contig_fs_lines_per_block),
                )
                comptime wfs_func = _contig_coop_fs_gpu_fft_kernel[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[wfs_func](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=wfs_grid,
                    block_dim=wfs_block,
                )
            elif config.use_warp_fs_column:
                comptime wfs_grid = (1, Int(config.warp_fs_batch_size))
                comptime wfs_block = Int(config.warp_fs_threads_per_block)
                comptime wfs_func = _warp_fs_column_gpu_fft_kernel[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[wfs_func](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=wfs_grid,
                    block_dim=wfs_block,
                )
            elif config.use_fuse_back_row:
                comptime fb_grid = (1, Int(config.fuse_back_batch_size))
                comptime fb_block = (
                    Int(config.fuse_back_L),
                    Int(config.fuse_back_tpt),
                )
                comptime fb_func = _fuse_back_row_gpu_fft_kernel[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    shared_address_space=address_space,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[fb_func](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=fb_grid,
                    block_dim=fb_block,
                )
            elif config.use_sbrc_column:
                comptime sbrc_grid = (1, Int(config.sbrc_batch_size))
                comptime sbrc_block = (
                    Int(config.sbrc_lines_per_block),
                    Int(config.sbrc_tpt),
                )
                comptime sbrc_func = _sbrc_column_gpu_fft_kernel[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    shared_address_space=address_space,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[sbrc_func](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=sbrc_grid,
                    block_dim=sbrc_block,
                )
            elif config.use_packed_transforms:
                comptime packed_grid = (1, Int(config.packed_batch_size))
                comptime packed_block = (
                    Int(config.threads_per_transform),
                    Int(config.transforms_per_block),
                )
                comptime packed_func = _packed_gpu_fft_kernel_radix_n_multi_dim[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    shared_address_space=address_space,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[packed_func](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=packed_grid,
                    block_dim=packed_block,
                )
            else:
                comptime grid_dim = (
                    Int(config.num_blocks),
                    Int(config.batch_size),
                )
                comptime block_func_batch = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    in_layout_type=in_layout_type,
                    out_layout_type=out_layout_type,
                    in_origin=x.origin,
                    out_origin=output.origin,
                    twf_layout_type=type_of(twf_layout),
                    twf_origin=twiddle_factors.origin,
                    calc_buf_origin=calc_buf.origin,
                    inverse=inverse,
                    bases=bases,
                    config=config,
                    shared_address_space=address_space,
                    runtime_twfs=runtime_twfs,
                    dim_idx=dim_idx,
                ]
                ctx.enqueue_function[block_func_batch](
                    output,
                    x,
                    twiddle_factors,
                    calc_buf,
                    grid_dim=grid_dim,
                    cluster_dim=OptionalReg[Dim](
                        config.num_blocks
                    ) if run_cluster else None,
                    block_dim=config.block_threads,
                )

    @always_inline
    @__parameter
    def _schedule_transpose[dim_idx: Int, *, forward: Bool]() raises:
        comptime config = plan.config[dim_idx]
        comptime schedule = _FftStockhamSchedule[
            bases,
            config.geo.dims,
            config.geo.start_dim_idx,
            _use_shared_memory_fn[config, _],
            skip_forward_transpose_stages = _fft_nd_fuse_forward_restore[
                config.geo.start_dim_idx
            ](),
        ]()
        comptime tp = _FftGpuTransposePlan[
            schedule,
            dim_idx,
            forward,
            out_layout_type,
            config.max_threads_available,
        ]()

        comptime transpose_gpu = _transpose_gpu[
            dst_dtype=out_dtype,
            dst_layout_type=out_layout_type,
            dst_origin=...,
            src_origin=...,
            into_=tp.into_,
            from_=tp.from_,
            scheduled_batches=tp.scheduled_batches,
            tile_x=Int(tp.TILE_X),
            tile_y=Int(tp.TILE_Y),
            y_rep=Int(tp.Y_REP),
        ]
        comptime if tp.write_lhs:
            comptime func = transpose_gpu[
                dst_origin=output.origin,
                src_origin=calc_buf.origin,
            ]
            ctx.enqueue_function[func](
                output,
                calc_buf,
                grid_dim=tp.grid_dim,
                block_dim=tp.block_dim,
            )
        else:
            comptime func = transpose_gpu[
                dst_origin=calc_buf.origin,
                src_origin=output.origin,
            ]
            ctx.enqueue_function[func](
                calc_buf,
                output,
                grid_dim=tp.grid_dim,
                block_dim=tp.block_dim,
            )

    @always_inline
    @__parameter
    def _schedule_forward_restore_fused() raises:
        """Enqueue reversed→native restore (replaces the forward adjacent-T chain)."""
        comptime config0 = plan.config[0]
        comptime rp = _FftGpuRestorePlan[
            config0.geo.dims,
            out_layout_type,
            config0.max_threads_available,
        ]()
        comptime func = _restore_reversed_nd_gpu[
            dst_dtype=out_dtype,
            dst_layout_type=out_layout_type,
            dst_origin=output.origin,
            src_origin=calc_buf.origin,
            scheduled_batches=rp.scheduled_batches,
            tile_x=Int(rp.TILE_X),
            tile_y=Int(rp.TILE_Y),
            y_rep=Int(rp.Y_REP),
        ]
        ctx.enqueue_function[func](
            output,
            calc_buf,
            grid_dim=rp.grid_dim,
            block_dim=rp.block_dim,
        )

    comptime forward_restore_count = plan.config[0].geo.start_dim_idx
    comptime fuse_forward_restore = _fft_nd_fuse_forward_restore[
        forward_restore_count
    ]()
    comptime skip_volume = plan.config[0].skip_volume_transpose
    comptime fuse_back = plan.config[0].nd_fuse_back_transpose
    comptime fuse_restore_store = plan.config[0].fuse_restore_store

    _schedule_run[forward_restore_count]()
    comptime if skip_volume:
        comptime for dim_idx in reversed(range(forward_restore_count)):
            _schedule_run[dim_idx]()
    elif fuse_back:
        # Last-axis kernel already wrote transposed; only dim FFTs + restore.
        comptime for dim_idx in reversed(range(forward_restore_count)):
            _schedule_run[dim_idx]()
        comptime if fuse_forward_restore and not fuse_restore_store:
            _schedule_forward_restore_fused()
        elif not fuse_forward_restore:
            comptime for dim_idx in range(forward_restore_count):
                _schedule_transpose[dim_idx, forward=True]()
    else:
        comptime for dim_idx in reversed(range(forward_restore_count)):
            _schedule_transpose[dim_idx, forward=False]()
            _schedule_run[dim_idx]()
        comptime if fuse_forward_restore and not fuse_restore_store:
            _schedule_forward_restore_fused()
        elif not fuse_forward_restore:
            comptime for dim_idx in range(forward_restore_count):
                _schedule_transpose[dim_idx, forward=True]()
