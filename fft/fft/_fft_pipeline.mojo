"""Stockham stage pipeline — sync, routing, and shared CPU/GPU stage plans."""

from std.math import ceildiv
from layout import TileTensor, TensorLayout
from std.gpu import barrier, cluster_arrive_relaxed, cluster_wait

from ._utils import (
    _calc_batches_M_N,
    _dims,
    _get_ordered_bases_processed_list,
    _num_stages_end_of,
    _product_of_dims,
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
](TrivialRegisterPassable):
    """Global ping-pong stage indexing for FFT, transpose, and scatter routing.
    """

    comptime total_stages = _num_stages_end_of[
        Self.bases, Self.dims, 0, Self.use_shared_memory
    ]() + 2 * Self.start_dim_idx

    comptime stages_end_of[dim_idx: Int] = _num_stages_end_of[
        Self.bases, Self.dims, dim_idx, Self.use_shared_memory
    ]()

    comptime prev_stages[dim_idx: Int] = Self.stages_end_of[dim_idx + 1] + (
        Self.start_dim_idx - dim_idx
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
](TrivialRegisterPassable):
    """Per-path flags: twiddle policy and first-stage `x_in` gating."""

    pass


comptime _FftCpuStagePath = _FftStagePathConfig[
    inline_twfs=False,
    runtime_twfs=False,
    gate_first_on_start_dim=True,
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
    ]

    @parameter
    def run[stage_fn: def[stage_b: Int]() capturing[_] -> None](self: Self):
        comptime for b in range(Self.stage_count):
            stage_fn[b]()


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
    comptime TILE = 32
    comptime M_ = ceildiv(Self.M, Self.TILE)
    comptime N_ = ceildiv(Self.N, Self.TILE)
    comptime num_threads = Self.N_ * Self.M_ * Self.TILE * Self.TILE
    comptime extra_fft_batches = UInt(Self.out_layout_type.static_shape[0])
    comptime total_batches = Self.intra_fft_batches * Self.extra_fft_batches
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime scheduled_batches = min(
        Self.total_batches, max(Self.thread_batch_size, UInt(1))
    )
    comptime grid_dim = (Self.N_, Self.M_, Self.scheduled_batches)
    comptime block_dim = (Self.TILE, Self.TILE, 1)


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
