from std.algorithm import parallelize
from std.complex import ComplexScalar
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from std.runtime.asyncrt import parallelism_level
from std.memory import ArcPointer

from ._fft_pipeline import (
    _Fft1dStageExec,
    _Fft1dStagePlan,
    _FftCpuStagePath,
    _FftNdimGeometry,
    _FftStockhamPipeline,
    _FftStockhamSchedule,
    _fft_no_shared_memory,
    _fft_noop_sync,
)
from ._fft_stage_route import _FftStageRouteParams
from ._fft_tile_io import _transpose_cpu

from ._utils import (
    _get_twiddle_factors,
    _min,
    _tail_tile_layout,
)


struct _CPUPlan[
    out_dtype: DType,
    out_layout_type: TensorLayout,
    inverse: Bool,
    bases: List[List[UInt]],
](Copyable):
    comptime geo = _FftNdimGeometry[Self.out_layout_type]
    # TODO: this should somehow be dependent on the CPU register size
    comptime max_stack_seq_len = 128
    """Maximum sequence length to fully unroll on the stack."""
    # TODO: this should somehow be dependent on the CPU register size
    comptime max_butterfly_base = 16
    """Maximum radix base to use the butterfly algorithm."""

    comptime L = List[ComplexScalar[Self.out_dtype]]

    var twiddle_factors: ArcPointer[List[Optional[Self.L]]]
    var calc_buf: ArcPointer[Self.L]

    def __init__(out self):
        self.twiddle_factors = {Self._get_dims_twfs()}
        comptime size = Self.out_layout_type.static_cosize // 2  # Self.L is complex
        self.calc_buf = {Self.L(capacity=size)}

    @staticmethod
    def _get_dims_twfs(out twfs: List[Optional[Self.L]]):
        twfs = {capacity = Self.geo.amnt_dims}

        comptime for dim_idx in range(Self.geo.amnt_dims):
            comptime length = UInt(Self.geo.dims.static_shape[dim_idx])
            comptime if length <= Self.max_stack_seq_len:
                twfs.append(None)
                continue
            twfs.append(
                _get_twiddle_factors[length, Self.out_dtype, Self.inverse]()
            )


def _run_cpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
](
    output: TileTensor[out_dtype, out_layout_type, out_origin, ...],
    x: TileTensor[in_dtype, in_layout_type, in_origin, ...],
    *,
    plan: _CPUPlan[out_dtype, out_layout_type, inverse, bases],
    cpu_workers: Optional[UInt] = None,
):
    comptime geo = plan.geo
    comptime x_complex_in = in_layout_type.static_shape[geo.rank - 1]
    comptime in_batch_stride = in_layout_type.static_stride[0]
    # NOTE: extract the unsafe pointer to avoid the arcpointer refcount
    var twfs_runtime_ptr = plan.twiddle_factors[].unsafe_ptr()

    @parameter
    def _find_max_batch_prod(out max_batch_prod: UInt):
        max_batch_prod = 0

        comptime for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(geo.dims.static_shape[i])
            max_batch_prod = max(dim // val, max_batch_prod)

    comptime max_batch_prod = _find_max_batch_prod()

    # Thread schedule (2-level parallelize):
    # - max_batch_prod ≈ largest intra-dim fan-out (dim // min_radix). That is
    #   the useful parallelism inside one batch when amnt_dims > 1.
    # - per_batch_workers: dedicate up to that many workers to the inner
    #   dim-batch loop; for 1-D there is nothing to fan out, so keep 1.
    # - parallel_batches: remaining workers run independent outer batches.
    #   Reserve (per_batch_workers - 1) so nested parallelize does not
    #   oversubscribe when both levels are active; clamp to [1, batches].
    var threads = cpu_workers.or_else(UInt(parallelism_level()))
    var per_batch_workers = (
        min(threads, max_batch_prod) if geo.amnt_dims > 1 else 1
    )
    var parallel_batches = min(
        max(threads - (per_batch_workers - 1), 1), geo.outer_batches
    )
    comptime schedule = _FftStockhamSchedule[
        bases, geo.dims, geo.start_dim_idx, _fft_no_shared_memory
    ]()

    @always_inline
    @parameter
    def _run_1d_fft[
        dtype_in: DType, //, dim_idx: Int
    ](
        shared_f_lhs: TileTensor[mut=True, out_dtype, ...],
        shared_f_rhs: TileTensor[mut=True, out_dtype, ...],
        x_in: TileTensor[mut=False, dtype_in, ...],
    ):
        comptime length = UInt(geo.dims.static_shape[dim_idx])
        comptime twfs_layout = row_major[Int(length), 2]()

        var twfs: TileTensor[
            mut=False, out_dtype, type_of(twfs_layout), ImmUntrackedOrigin
        ]
        comptime if length <= plan.max_stack_seq_len:
            twfs = stack_allocation[out_dtype](twfs_layout).as_immut()
        else:
            twfs = TileTensor(
                twfs_runtime_ptr[unsafe_offset=dim_idx]
                .value()
                .unsafe_ptr()
                .unsafe_bitcast[Scalar[out_dtype]]()
                .mut_cast[False]()
                .unsafe_origin_cast[ImmUntrackedOrigin](),
                twfs_layout,
            )

        comptime stage_plan = _Fft1dStagePlan[
            schedule, dim_idx, inverse, x_complex_in, _FftCpuStagePath
        ]()
        comptime run_butterfly = (
            Int(stage_plan.ordered_bases[0]) < plan.max_butterfly_base
        )
        var pipeline = _FftStockhamPipeline[
            out_dtype,
            type_of(shared_f_lhs),
            type_of(shared_f_rhs),
            _fft_noop_sync,
        ](shared_f_lhs, shared_f_rhs)
        var payload = pipeline.stage_payload(x_in)

        @parameter
        def _run_stage[stage_b: Int]():
            comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
            comptime stage = _FftStageRouteParams[stage_exec]()
            comptime if run_butterfly and length <= plan.max_stack_seq_len:
                stage.run_butterfly_comptime(payload)
            elif run_butterfly:
                stage.run_butterfly_stage[plan.max_stack_seq_len](payload, twfs)
            elif length <= plan.max_stack_seq_len:
                stage.run_elem_comptime(payload)
            else:
                stage.run_elem_per_thread(payload, twfs)

        stage_plan.run[_run_stage]()

    # NOTE: extract the pointer to avoid the arcpointer refcount.
    var calc_buf_ptr = (
        plan.calc_buf[]
        .unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_bitcast[Scalar[out_dtype]]()
    )
    comptime o_layout = _tail_tile_layout[out_layout_type]()
    comptime x_tail_layout = _tail_tile_layout[in_layout_type]()

    @always_inline
    @parameter
    def _run_batch(block_num: Int):
        var block_offset = geo.batch_stride * block_num
        var base_out = TileTensor(output.ptr + block_offset, o_layout)
        var base_calc = TileTensor(
            UnsafePointer(
                calc_buf_ptr.unsafe_offset(block_offset)
            ).unsafe_mut_cast[True](),
            o_layout,
        )
        var base_x = TileTensor(
            x.ptr + in_batch_stride * block_num, x_tail_layout
        )

        @always_inline
        @parameter
        def _run_transpose[dim_idx: Int, *, forward: Bool]():
            comptime from_ = dim_idx + Int(not forward)
            comptime into_ = dim_idx + Int(forward)
            comptime write_lhs = schedule.write_lhs_for_transpose[
                dim_idx, forward=forward
            ]
            comptime tp = _transpose_cpu[from_=from_, into_=into_]
            comptime if write_lhs:
                tp(base_out, base_calc, Int(per_batch_workers))
            else:
                tp(base_calc, base_out, Int(per_batch_workers))

        comptime if geo.amnt_dims == 1:
            _run_1d_fft[geo.start_dim_idx](base_out, base_calc, base_x)
        else:
            comptime for idx in reversed(range(geo.amnt_dims)):
                comptime dim = Int(geo.dims.static_shape[idx])
                comptime batch_prod = UInt(geo.prod // dim)

                comptime if idx != geo.start_dim_idx:
                    _run_transpose[idx, forward=False]()

                comptime dim_x_layout = row_major[dim, x_complex_in]()
                comptime x_offset = dim * x_complex_in
                comptime dim_out_layout = row_major[dim, 2]()
                comptime out_offset = dim * 2

                @always_inline
                @parameter
                def _run_dim_batch(flat_idx: Int):
                    var dim_batch_x = TileTensor(
                        base_x.ptr + flat_idx * x_offset, dim_x_layout
                    )
                    var dim_batch_out = TileTensor(
                        base_out.ptr + flat_idx * out_offset, dim_out_layout
                    )
                    var dim_batch_calc = TileTensor(
                        base_calc.ptr + flat_idx * out_offset, dim_out_layout
                    )
                    _run_1d_fft[idx](dim_batch_out, dim_batch_calc, dim_batch_x)

                parallelize[func=_run_dim_batch](
                    Int(batch_prod), Int(per_batch_workers)
                )

            comptime for idx in range(geo.amnt_dims - 1):
                _run_transpose[idx, forward=True]()

    parallelize[func=_run_batch](Int(geo.outer_batches), Int(parallel_batches))
