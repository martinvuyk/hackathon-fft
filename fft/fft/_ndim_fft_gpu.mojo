from max.algorithm import parallelize, vectorize
from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from std.collections import Optional, OptionalReg
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
)
from max.gpu.host import DeviceContext, DeviceBuffer, Dim, DeviceFunction
from max.gpu.memory import AddressSpace
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
)
from ._fft_pipeline import (
    _Fft1dStageExec,
    _Fft1dStagePlan,
    _FftDimBoundaryPayload,
    _FftGpuTransposePlan,
    _FftNdimGeometry,
    _FftStagePathConfig,
    _FftStockhamPipeline,
    _FftStockhamSchedule,
    _fft_block_stage_sync,
    _fft_cluster_stage_sync,
)
from ._fft_stage_route import _FftStageRouteParams
from ._fft_tile_io import _transpose_gpu


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
    comptime shared_mem_per_m = UInt(
        Self.gpu_info.shared_memory_per_multiprocessor
    )
    comptime shared_mem_per_t = Self.shared_mem_per_m // UInt(
        Self.threads_per_m
    )
    comptime shared_mem_per_block = (
        Self.shared_mem_per_t * Self.max_threads_per_block
    )
    comptime shared_mem_per_warp = (Self.shared_mem_per_t * Self.warp_size)
    comptime shared_mem_per_cluster = (
        Self.shared_mem_per_block * Self.max_cluster_size
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

    comptime dim_layout = row_major[Self.dim, 2]()
    comptime dim_size = UInt(Self.dim_layout.size())
    comptime dim_byte_size = UInt(size_of[Self.out_dtype]()) * Self.dim_size

    comptime use_shared_mem = (
        2 * Self.dim_byte_size <= Self.shared_mem_per_warp
    ) or (2 * Self.dim_byte_size <= Self.shared_mem_per_block) or (
        2 * Self.dim_byte_size <= Self.shared_mem_per_cluster
    )

    comptime block_threads = ceildiv(Self.num_threads, Self.num_blocks)
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime batch_size = min(Self.batches, Self.thread_batch_size)

    comptime inline_twfs = Self.dim_size <= UInt(
        Self.gpu_info.max_registers_per_block // 2
    )

    comptime max_shared_mem_size = (
        Self.shared_mem_per_warp if (
            Self.can_run_in_warp
        ) else Self.shared_mem_per_block if (
            Self.can_run_in_block
        ) else Self.shared_mem_per_cluster
    )

    comptime use_shared_memory = 2 * Self.dim_size <= Self.max_shared_mem_size

    comptime use_cluster_sync = not (
        Self.can_run_in_block or Self.can_run_in_warp
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
        comptime out_size = Self.out_layout_type.static_cosize
        self.calc_buf = ctx.enqueue_create_buffer[Self.out_dtype](out_size)

        comptime amnt_dims = Self.config[0].geo.amnt_dims
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
    var block_num = UInt(block_dim.y * block_idx.y)

    comptime total_threads = config.block_threads * config.num_blocks
    comptime x_complex_in = in_layout_type.static_shape[config.geo.rank - 1]

    comptime base_out_layout = row_major[config.dim, 2]()
    comptime base_x_layout = row_major[config.dim, x_complex_in]()

    comptime shared_f_layout = row_major[config.dim, 2]()

    var shared_f_lhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)
    var shared_f_rhs = stack_allocation[
        out_dtype, address_space=shared_address_space
    ](shared_f_layout)

    comptime schedule = _FftStockhamSchedule[
        bases,
        config.geo.dims,
        config.geo.start_dim_idx,
        _use_shared_memory_fn[config, _],
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

    comptime path = _FftStagePathConfig[
        inline_twfs=config.inline_twfs,
        runtime_twfs=runtime_twfs,
        gate_first_on_start_dim=False,
    ]()
    comptime stage_plan = _Fft1dStagePlan[
        schedule, dim_idx, inverse, x_complex_in, path
    ]()

    comptime batched_iters = max(config.batches // config.batch_size, 1)
    comptime x_stride = Int(config.dim) * x_complex_in
    comptime out_stride = Int(config.dim) * 2

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
        var base_x = TileTensor(x.ptr + x_stride * offset, base_x_layout)
        var base_out = TileTensor(
            output.ptr + out_stride * offset, base_out_layout
        )
        var base_calc = TileTensor(
            calc_buf.ptr + out_stride * offset, base_out_layout
        )

        @always_inline
        def _run_stages(
            x_in: TileTensor[mut=False, ...],
        ) {mut pipeline, imm global_i, imm twiddle_factors,}:
            var payload = pipeline.stage_payload(x_in)
            comptime for stage_b in range(stage_plan.stage_count):
                comptime stage_exec = _Fft1dStageExec[stage_plan, stage_b]()
                comptime stage = _FftStageRouteParams[stage_exec]()
                stage.run_elem_per_thread_once(
                    payload, global_i, twiddle_factors
                )
                pipeline.sync_stage()

        comptime if not config.use_shared_memory:
            _run_stages(base_x)
        else:
            comptime if stage_plan.input_from_x:
                _run_stages(base_x)
            elif stage_plan.write_global_lhs:
                _run_stages(base_calc)
            else:
                _run_stages(base_out)

            var stage = pipeline.stage_payload(base_x.as_immut())
            var boundary = _FftDimBoundaryPayload[
                type_of(stage), type_of(base_out), type_of(base_calc)
            ](stage, base_out, base_calc)
            boundary.scatter_dim_result[stage_plan](global_i)

    for i in range(batched_iters):
        _run_batch_at(Int(block_num + i * config.batch_size))
        pipeline.batch_sync()

    comptime full_iters = batched_iters * config.batch_size
    comptime remainder = config.batches - full_iters

    comptime if remainder > 0:
        if block_num < remainder:
            _run_batch_at(Int(full_iters + block_num))
        pipeline.batch_sync()




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
    def _schedule_run[dim_idx: Int]() raises:
        comptime config = plan.config[dim_idx]

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
            twiddle_factors = stack_allocation[out_dtype](twf_layout).as_immut()

        comptime grid_dim = (Int(config.num_blocks), config.batch_size)
        comptime run_cluster = config.can_run_in_block_cluster and (
            config.num_blocks > 1
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

    comptime start_dim_idx = plan.config[0].geo.start_dim_idx
    _schedule_run[start_dim_idx]()

    comptime for dim_idx in reversed(range(start_dim_idx)):
        _schedule_transpose[dim_idx, forward=False]()
        _schedule_run[dim_idx]()

    comptime for dim_idx in range(start_dim_idx):
        _schedule_transpose[dim_idx, forward=True]()
