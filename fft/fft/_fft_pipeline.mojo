"""Stockham stage pipeline — sync, routing, and shared CPU/GPU stage plans."""

from std.math import ceildiv
from layout import TileTensor, TensorLayout
from max.gpu.sync import barrier
from max.gpu import cluster_arrive_relaxed, cluster_wait

from ._utils import (
    _calc_batches_M_N,
    _dims,
    _get_ordered_bases_processed_list,
    _num_stages_end_of,
    _product_of_dims,
    _product_of_dims_slice,
)
from ._fft import _FFTKernelExecConfig
from ._fft_payload import _FftStockhamPayload
from ._fft_tile_io import _scatter_dim_result


@always_inline
def _compute_write_lhs[total_stages: Int, stage_idx: Int]() -> Bool:
    """Return which ping-pong buffer is written at `stage_idx`."""
    return (total_stages - (stage_idx + 1)) % 2 == 0


@always_inline
def _fft_no_shared_memory[idx: Int]() -> Bool:
    _ = idx
    return False


@always_inline
def _fft_noop_sync():
    pass


@always_inline
def _fft_block_stage_sync():
    barrier()


@always_inline
def _fft_cluster_stage_sync():
    cluster_arrive_relaxed()
    cluster_wait()


@fieldwise_init
struct _FftNdimGeometry[out_layout_type: TensorLayout](TrivialRegisterPassable):
    """FFT geometry derived from an output layout.

    Strips the leading batch axis (index 0) and trailing complex pair; the
    remaining axes are the spatial FFT dims, processed from `start_dim_idx`
    down to 0. `outer_batches` / `batch_stride` refer to that leading axis.
    """

    comptime rank = Self.out_layout_type.rank
    comptime dims = _dims[Self.out_layout_type]
    comptime amnt_dims = Self.dims.rank
    comptime prod = _product_of_dims[Self.dims]()
    comptime start_dim_idx = Self.amnt_dims - 1
    comptime outer_batches = UInt(Self.out_layout_type.static_shape[0])
    comptime batch_stride = Self.out_layout_type.static_stride[0]


@fieldwise_init
struct _FftStockhamSchedule[
    bases: List[List[UInt]],
    dims: TensorLayout,
    start_dim_idx: Int,
    use_shared_memory: def[Int]() thin -> Bool,
    *,
    skip_transpose_stages: Bool = False,
    skip_forward_transpose_stages: Bool = False,
](TrivialRegisterPassable):
    """Global ping-pong stage indexing for FFT, transpose, and scatter routing.

    `skip_transpose_stages`: omit all transpose slots (column ND schedules).
    `skip_forward_transpose_stages`: keep back-Ts; forward chain replaced by restore.
    """

    comptime transpose_stage_count = (
        0 if Self.skip_transpose_stages else (
            Self.start_dim_idx if Self.skip_forward_transpose_stages else (
                2 * Self.start_dim_idx
            )
        )
    )

    comptime total_stages = _num_stages_end_of[
        Self.bases, Self.dims, 0, Self.use_shared_memory
    ]() + Self.transpose_stage_count

    comptime stages_end_of[dim_idx: Int] = _num_stages_end_of[
        Self.bases, Self.dims, dim_idx, Self.use_shared_memory
    ]()

    comptime prev_stages[dim_idx: Int] = Self.stages_end_of[dim_idx + 1] + (
        0 if Self.skip_transpose_stages else (Self.start_dim_idx - dim_idx)
    )

    comptime write_lhs_at[stage_idx: Int] = _compute_write_lhs[
        Self.total_stages, stage_idx
    ]()

    comptime transpose_stage_idx[dim_idx: Int, *, forward: Bool] = (
        Self.stages_end_of[0 if forward else dim_idx + 1]
        + Self.start_dim_idx
        + (dim_idx if forward else -(dim_idx + 1))
    )

    comptime write_lhs_for_transpose[dim_idx: Int, *, forward: Bool] = (
        Self.write_lhs_at[Self.transpose_stage_idx[dim_idx, forward=forward]]
    )


@fieldwise_init
struct _FftStagePathConfig[
    inline_twfs: Bool,
    runtime_twfs: Bool,
    gate_first_on_start_dim: Bool,
    sm_complex_stride: Int = 2,
    global_complex_stride: Int = 2,
](TrivialRegisterPassable):
    """Per-path flags: twiddle policy, first-stage gating, shared/global strides.
    """

    pass


comptime _FftCpuStagePath = _FftStagePathConfig[
    inline_twfs=False,
    runtime_twfs=False,
    gate_first_on_start_dim=True,
    sm_complex_stride=2,
    global_complex_stride=2,
]()


@fieldwise_init
struct _Fft1dStagePlan[
    schedule: _FftStockhamSchedule,
    dim_idx: Int,
    inverse: Bool,
    x_complex_in: Int,
    path: _FftStagePathConfig,
](TrivialRegisterPassable):
    """Comptime 1-D radix stage metadata shared by CPU and GPU paths."""

    comptime length = UInt(Self.schedule.dims.static_shape[Self.dim_idx])
    comptime inline_twfs = Self.path.inline_twfs
    comptime runtime_twfs = Self.path.runtime_twfs
    comptime gate_first_on_start_dim = Self.path.gate_first_on_start_dim

    comptime start_dim_idx = Self.schedule.start_dim_idx
    comptime total_stages = Self.schedule.total_stages
    comptime prev_stages = Self.schedule.prev_stages[Self.dim_idx]
    comptime bases = Self.schedule.bases[Self.dim_idx]

    comptime bases_processed = materialize[
        _get_ordered_bases_processed_list[Self.length, Self.bases]()
    ]()
    comptime ordered_bases = Self.bases_processed[0]
    comptime processed_list = Self.bases_processed[1]
    comptime stage_count = len(Self.ordered_bases)

    comptime write_global_lhs = _compute_write_lhs[
        Self.total_stages, Self.prev_stages
    ]()
    comptime last_write_lhs = _compute_write_lhs[
        Self.total_stages, Self.prev_stages + Self.stage_count - 1
    ]()
    comptime input_from_x = Self.dim_idx == Self.start_dim_idx
    comptime sm_complex_stride = Self.path.sm_complex_stride
    comptime global_complex_stride = Self.path.global_complex_stride

    # Stage 0 reads global/input with `global_complex_stride` (2 after a layout
    # transpose; axis stride for in-place column FFT). Later shared stages use
    # `sm_complex_stride`. Last GPU stage may override `out_complex_stride` to
    # write global directly.
    comptime kernel_exec_config[stage_b: Int] = _FFTKernelExecConfig[
        Self.length,
        Self.x_complex_in == 1
        and Self.dim_idx == Self.start_dim_idx
        and stage_b == 0,
        Self.ordered_bases[stage_b],
        Self.processed_list[stage_b],
        Self.inverse,
        Self.ordered_bases,
        Self.inline_twfs,
        Self.runtime_twfs,
        False,
        in_complex_stride = (
            Self.global_complex_stride if stage_b == 0 else Self.sm_complex_stride
        ),
        out_complex_stride = Self.sm_complex_stride,
    ]


@fieldwise_init
struct _Fft1dStageExec[
    plan: _Fft1dStagePlan,
    stage_b: Int,
](TrivialRegisterPassable):
    """Per-stage exec bundle: routing flags and kernel config for one radix stage.
    """

    comptime kernel_config = Self.plan.kernel_exec_config[Self.stage_b]
    comptime config = Self.kernel_config()
    comptime is_first = Self.stage_b == 0 and (
        not Self.plan.gate_first_on_start_dim
        or Self.plan.dim_idx == Self.plan.start_dim_idx
    )
    comptime write_lhs = _compute_write_lhs[
        Self.plan.total_stages, Self.plan.prev_stages + Self.stage_b
    ]()


@fieldwise_init
struct _FftDimBoundaryPayload[
    stage_payload: type_of(_FftStockhamPayload),
    out_tile: type_of(TileTensor[mut=True, stage_payload.out_dtype, ...]),
    calc_tile: type_of(TileTensor[mut=True, stage_payload.out_dtype, ...]),
](TrivialRegisterPassable):
    """Stage payload plus global out/calc tiles for shared-memory dim finish."""

    var stage: _FftStockhamPayload[
        Self.stage_payload.out_dtype,
        Self.stage_payload.in_tile,
        Self.stage_payload.lhs_tile,
        Self.stage_payload.rhs_tile,
    ]
    var base_out: Self.out_tile
    var base_calc: Self.calc_tile

    @always_inline
    def scatter_dim_result[
        stage_plan: _Fft1dStagePlan,
    ](self, global_i: UInt):
        _scatter_dim_result[
            Self.stage_payload.out_dtype,
            stage_plan.write_global_lhs,
            stage_plan.last_write_lhs,
            sm_complex_stride = stage_plan.sm_complex_stride,
            global_complex_stride = stage_plan.global_complex_stride,
        ](
            global_i,
            self.stage.lhs,
            self.stage.rhs,
            self.base_out,
            self.base_calc,
        )


@fieldwise_init
struct _FftGpuTransposePlan[
    schedule: _FftStockhamSchedule,
    dim_idx: Int,
    forward: Bool,
    out_layout_type: TensorLayout,
    max_threads_available: UInt,
](TrivialRegisterPassable):
    """GPU transpose launch geometry and ping-pong buffer routing."""

    comptime into_ = Self.dim_idx + Int(Self.forward)
    comptime from_ = Self.dim_idx + Int(not Self.forward)
    comptime write_lhs = Self.schedule.write_lhs_for_transpose[
        Self.dim_idx, forward=Self.forward
    ]
    comptime dims = Self.schedule.dims
    comptime sizes = _calc_batches_M_N[
        Self.dims, into_=Self.into_, from_=Self.from_
    ]()
    comptime intra_fft_batches = Self.sizes[0]
    comptime M = Self.sizes[1]
    comptime N = Self.sizes[2]
    comptime TILE_X = UInt(32)
    # Non-square transposes benefit from fewer threads per block while
    # preserving 32-wide coalesced x-lanes (32x8 => 256 threads, ELEMS=4).
    comptime TILE_Y = UInt(8) if Self.M != Self.N else UInt(32)
    # Non-square full-tile: each block owns Y_REP tiles along M (amortize launch).
    comptime Y_REP = UInt(2) if Self.M != Self.N else UInt(1)
    comptime M_ = ceildiv(Self.M, Self.TILE_X * Self.Y_REP)
    comptime N_ = ceildiv(Self.N, Self.TILE_X)
    comptime num_threads = Self.N_ * Self.M_ * Self.TILE_X * Self.TILE_Y
    comptime extra_fft_batches = UInt(Self.out_layout_type.static_shape[0])
    comptime total_batches = Self.intra_fft_batches * Self.extra_fft_batches
    # Launch one batch plane per grid.z entry (up to hardware grid.z limit).
    # The old fair-share `max_threads_available // num_threads` left only a
    # handful of planes and forced each block to serialize hundreds of batches
    # — catastrophic for ND (e.g. 20 blocks × 1280 serial runs on 100×64³).
    comptime max_grid_z = UInt(65535)
    # Old fair-share left ~5 planes on 64³ (20 blocks × 1280 serial). Aim for
    # enough concurrent planes to fill the GPU without maxing grid.z always.
    comptime fair_batch_wave = max(
        UInt(1),
        Self.max_threads_available // max(Self.num_threads, UInt(1)),
    )
    comptime scheduled_batches = min(
        Self.total_batches,
        min(Self.max_grid_z, max(Self.fair_batch_wave * 128, UInt(512))),
    )
    comptime grid_dim = (Self.N_, Self.M_, Self.scheduled_batches)
    comptime block_dim = (Self.TILE_X, Self.TILE_Y, 1)


@fieldwise_init
struct _FftGpuRestorePlan[
    dims: TensorLayout,
    out_layout_type: TensorLayout,
    max_threads_available: UInt,
](TrivialRegisterPassable):
    """Launch geometry for reversed→native restore (batches × `(Dlast, D0)` tiles)."""

    comptime D0 = UInt(Self.dims.static_shape[0])
    comptime Dlast = UInt(Self.dims.static_shape[Self.dims.rank - 1])
    comptime middle_prod = UInt(
        _product_of_dims_slice[Self.dims, 1, Self.dims.rank - 1]()
    )
    comptime TILE_X = UInt(32)
    comptime TILE_Y = UInt(8) if Self.D0 != Self.Dlast else UInt(32)
    comptime Y_REP = UInt(2) if Self.D0 != Self.Dlast else UInt(1)
    comptime M_ = ceildiv(Self.Dlast, Self.TILE_X * Self.Y_REP)
    comptime N_ = ceildiv(Self.D0, Self.TILE_X)
    comptime num_threads = Self.M_ * Self.N_ * Self.TILE_X * Self.TILE_Y
    comptime extra_fft_batches = UInt(Self.out_layout_type.static_shape[0])
    comptime total_batches = Self.extra_fft_batches * Self.middle_prod
    comptime max_grid_z = UInt(65535)
    comptime fair_batch_wave = max(
        UInt(1),
        Self.max_threads_available // max(Self.num_threads, UInt(1)),
    )
    comptime scheduled_batches = min(
        Self.total_batches,
        min(Self.max_grid_z, max(Self.fair_batch_wave * 128, UInt(512))),
    )
    comptime grid_dim = (Self.N_, Self.M_, Self.scheduled_batches)
    comptime block_dim = (Self.TILE_X, Self.TILE_Y, 1)


@fieldwise_init
struct _FftStockhamPipeline[
    out_dtype: DType,
    lhs_tile: type_of(TileTensor[mut=True, out_dtype, ...]),
    rhs_tile: type_of(TileTensor[mut=True, out_dtype, ...]),
    stage_sync: def() thin -> None,
](TrivialRegisterPassable):
    """Owns ping-pong tiles and stage/batch sync."""

    var lhs: Self.lhs_tile
    var rhs: Self.rhs_tile

    @always_inline
    def stage_payload(
        self, x_in: TileTensor[mut=False, ...]
    ) -> _FftStockhamPayload[
        Self.out_dtype, type_of(x_in), Self.lhs_tile, Self.rhs_tile
    ]:
        return {self.lhs, self.rhs, x_in}

    @always_inline
    def sync_stage(self):
        """Sync between radix stages within one 1-D FFT."""
        Self.stage_sync()

    @always_inline
    def batch_sync(self):
        """Sync between batch iterations (GPU inter-batch barrier)."""
        Self.stage_sync()
