from std.algorithm import parallelize, vectorize
from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from std.collections import OptionalReg
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    barrier,
    cluster_arrive_relaxed,
    cluster_wait,
)
from std.gpu.host import DeviceContext, DeviceBuffer, Dim, DeviceFunction
from std.gpu.host.info import Vendor, is_cpu, GPUInfo
from layout import Layout, LayoutTensor, IntTuple
from std.utils.index import IndexList
from layout.int_tuple import IntArray
from std.runtime.asyncrt import parallelism_level
from std.sys.info import has_accelerator, size_of, simd_width_of
from std.math import ceildiv
from std.utils import Variant

from ._utils import (
    _get_dtype,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _product_of_dims,
    _get_cascade_idxes,
    _get_twiddle_factors,
    _num_stages_end_of,
)
from ._fft import _radix_n_fft_kernel_cooley_tukey, _radix_n_fft_kernel_stockham


@fieldwise_init
struct _GPUExecConfig[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
]:
    comptime rank = Self.out_layout.rank()
    comptime dims = Self.out_layout.shape[1 : Self.rank - 1]
    comptime prod = _product_of_dims(Self.dims)
    """The product of the dimensions in the tensor."""
    comptime start_dim_idx = len(Self.dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime max_dim_base = _max(Self.dims)
    """The smallest base for the biggest dimension in the tensor."""
    comptime num_threads = Self.max_dim_base
    """The total number of threads per worload."""
    comptime max_dim = _max(Self.dims)
    """The biggest dimension in the tensor."""
    comptime max_base = _max(Self.bases)
    """The biggest radix base."""

    comptime batches = UInt(Self.out_layout.shape[0].value())
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
    comptime buf_size_max_dim = UInt(
        size_of[ComplexScalar[Self.out_dtype]]() * Int(Self.max_dim)
    )
    comptime buf_size_full_output = UInt(
        size_of[ComplexScalar[Self.out_dtype]]() * Self.prod
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
        Self.gpu_info.vendor == Vendor.NVIDIA_GPU
        and Self.gpu_info.compute >= 9.0
    )
    comptime can_run_in_block_cluster = Self.num_blocks <= (
        Self.max_cluster_size
    ) and Self.is_sm_90_or_newer and (
        Self.test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    comptime use_shared_mem = (
        2 * Self.buf_size_full_output <= Self.shared_mem_per_warp
        or 2 * Self.buf_size_max_dim <= Self.shared_mem_per_warp
    ) or (
        2 * Self.buf_size_full_output <= Self.shared_mem_per_block
        or 2 * Self.buf_size_max_dim <= Self.shared_mem_per_block
    ) or (
        2 * Self.buf_size_full_output <= Self.shared_mem_per_cluster
        or 2 * Self.buf_size_max_dim <= Self.shared_mem_per_cluster
    )

    comptime block_threads = ceildiv(Self.num_threads, Self.num_blocks)
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime batch_size = min(Self.batches, Self.thread_batch_size)

    comptime inline_twfs = Self.buf_size_max_dim <= UInt(
        Self.gpu_info.max_registers_per_block // 2
    )

    comptime max_shared_mem_size = (
        Self.shared_mem_per_warp if (
            Self.can_run_in_warp
        ) else Self.shared_mem_per_block if (
            Self.can_run_in_block
        ) else Self.shared_mem_per_cluster
    )
    comptime shared_f_out_layout = Layout.row_major(
        IntTuple(Self.out_layout.shape[1:]).flatten()
    )
    comptime out_size = UInt(
        size_of[Self.out_dtype]() * Self.shared_f_out_layout.size()
    )
    comptime max_dim_layout = Layout.row_major(Int(Self.max_dim), 2)
    comptime max_dim_size = UInt(
        size_of[Self.out_dtype]() * Self.max_dim_layout.size()
    )

    comptime use_shared_f_total = 2 * Self.out_size <= Self.max_shared_mem_size
    comptime use_shared_f_max_dim = 2 * Self.max_dim_size <= Self.max_shared_mem_size
    comptime use_global_memory = not (
        Self.use_shared_f_total or Self.use_shared_f_max_dim
    )


@fieldwise_init
struct _GPUPlan[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
](Copyable):
    comptime config = _GPUExecConfig[
        Self.out_dtype,
        Self.out_layout,
        Self.inverse,
        Self.bases,
        Self.test,
        Self.gpu_info,
        Self.max_cluster_size,
        Self.runtime_twfs,
    ]()

    var twfs_buffer: InlineArray[
        DeviceBuffer[Self.out_dtype], Int(not Self.config.inline_twfs)
    ]
    var calc_buf: InlineArray[
        DeviceBuffer[Self.out_dtype], Int(Self.config.use_global_memory)
    ]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime assert Self.config.threads_per_m > 0, (
            "Unknown number of threads per sm for the given device. "
            "It is needed in order to run the gpu implementation."
        )

        comptime out_b_layout = Layout.row_major(Self.out_layout.shape[1:])
        comptime b_size = out_b_layout.size()

        comptime if Self.config.inline_twfs and Self.config.use_global_memory:
            return {
                {uninitialized = True},
                [ctx.enqueue_create_buffer[Self.out_dtype](b_size)],
            }
        elif Self.config.inline_twfs:
            return {{uninitialized = True}, {uninitialized = True}}

        var twfs = ctx.enqueue_create_buffer[Self.out_dtype](
            2 * Self.config.prod
        )
        var offset = UInt(0)

        comptime for idx in range(len(Self.bases)):
            comptime length = UInt(Self.config.dims[idx].value())
            comptime twfs_array = _get_twiddle_factors[
                length, Self.out_dtype, Self.inverse
            ]()
            comptime complex_len = 2 * length
            # FIXME(#5686): replace with this once it's solved
            # ref twfs_array_runtime = global_constant[twfs_array]()
            var twfs_array_runtime = materialize[twfs_array]()
            var view = DeviceBuffer(
                ctx, twfs.unsafe_ptr() + offset, Int(complex_len), owning=False
            )
            var ptr = twfs_array_runtime.unsafe_ptr()
            ctx.enqueue_copy(view, ptr.bitcast[Scalar[Self.out_dtype]]())
            offset += complex_len

        comptime if Self.config.use_global_memory:
            return {
                [twfs^],
                [ctx.enqueue_create_buffer[Self.out_dtype](b_size)],
            }
        else:
            return {[twfs^], {uninitialized = True}}


def _intra_something_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    stage_sync_fn: def(),
    runtime_twfs: Bool,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[out_dtype, twf_layout, twf_origin],
    calc_buf: LayoutTensor[out_dtype, out_layout, calc_buf_origin],
):
    var local_i = thread_idx.x
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y

    comptime total_threads = config.block_threads * config.num_blocks
    comptime x_complex_in = in_layout.shape[config.rank - 1].value()

    comptime base_out_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, base_out_layout, ...]
    comptime base_out_t = out_t[output.origin]
    comptime base_calc_t = out_t[calc_buf.origin]

    comptime base_x_layout = Layout.row_major(x.layout.shape[1:])
    comptime base_x_t = LayoutTensor[
        in_dtype, base_x_layout, x.origin, address_space=x.address_space
    ]

    comptime shared_f_t = type_of(
        LayoutTensor[
            out_dtype,
            Layout.row_major(0) if config.use_global_memory else (
                config.shared_f_out_layout if (
                    config.use_shared_f_total
                ) else config.max_dim_layout
            ),
            MutExternalOrigin,
            address_space=shared_address_space,
        ].stack_allocation()
    )

    var shared_f_lhs = shared_f_t.stack_allocation()
    var shared_f_rhs = shared_f_t.stack_allocation()

    @parameter
    def _run_1d_fft[
        dim_idx: Int, use_x: Bool = dim_idx == config.start_dim_idx
    ](
        shared_f_lhs: LayoutTensor[mut=True, out_dtype, ...],
        shared_f_rhs: LayoutTensor[mut=True, out_dtype, ...],
        x: LayoutTensor[mut=False, ...],
    ):
        comptime assert shared_f_lhs.layout == shared_f_rhs.layout
        comptime length = UInt(x.layout.shape[0].value())
        comptime do_rfft = dim_idx == config.start_dim_idx and x_complex_in == 1
        comptime bases_processed = materialize[
            _get_ordered_bases_processed_list[length, bases[dim_idx]]()
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        comptime num_stages = _num_stages_end_of[
            bases, config.dims, dim_idx + 1
        ]()

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime threads_for_base = length
            comptime func = _radix_n_fft_kernel_stockham[
                ...,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                ordered_bases=ordered_bases,
                runtime_twfs=runtime_twfs,
                inline_twfs=config.inline_twfs,
            ]

            @parameter
            def _run():
                var twfs = twiddle_factors
                comptime if _num_stages_end_of[
                    bases, config.dims, 0
                ]() % 2 == 0:
                    comptime if b == 0 and use_x:
                        func(shared_f_rhs, x, global_i, twfs)
                    elif (num_stages + b) % 2 == 0:
                        func(shared_f_rhs, shared_f_lhs, global_i, twfs)
                    else:
                        func(shared_f_lhs, shared_f_rhs, global_i, twfs)
                else:
                    comptime if b == 0 and use_x:
                        func(shared_f_lhs, x, global_i, twfs)
                    elif (num_stages + b) % 2 == 0:
                        func(shared_f_lhs, shared_f_rhs, global_i, twfs)
                    else:
                        func(shared_f_rhs, shared_f_lhs, global_i, twfs)

            comptime if threads_for_base == total_threads:
                _run()
            else:
                if global_i < threads_for_base:
                    _run()

            stage_sync_fn()

    @parameter
    def _copy_to_output[
        dim_idx: Int
    ](
        base_out: base_out_t,
        shared_f_lhs: LayoutTensor[mut=False, out_dtype, ...],
        shared_f_rhs: LayoutTensor[mut=False, out_dtype, ...],
    ):
        comptime flat_layout = Layout.row_major(config.prod, 2)
        comptime flat_t = LayoutTensor[out_dtype, flat_layout, ...]
        comptime batch_prod = UInt(config.prod) // config.max_dim
        comptime length = UInt(shared_f_lhs.layout.shape[dim_idx].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx]
        ]()

        for i in range(batch_prod):
            var batch_prod_offset = i * config.max_dim
            var b_idx = Int(batch_prod_offset + local_i)
            var num = flat_t(shared_f_lhs.ptr).load[width=2](b_idx, 0)
            flat_t(base_out.ptr).store(b_idx, 0, num)

    @parameter
    def _run_ndim_fft(
        base_out: base_out_t, base_calc: base_calc_t, base_x: base_x_t
    ):
        comptime if len(config.dims) == 1:
            comptime if config.use_shared_f_total:
                _run_1d_fft[0](shared_f_lhs, shared_f_rhs, base_x)
                _copy_to_output[0](base_out, shared_f_lhs, shared_f_rhs)
            else:
                _run_1d_fft[0](base_out, base_calc, base_x)
        else:
            comptime for idx in reversed(range(len(config.dims))):
                comptime dim = config.dims[idx].value()
                comptime batch_prod = config.prod // dim

                comptime exclude = (idx, config.rank - 2)
                comptime exclude_t = IntTuple(idx, config.rank - 2)
                comptime dim_sl = Slice(0, dim)
                comptime o_comp = Slice(0, 2)

                for batch_prod_idx in range(batch_prod):
                    comptime x_comp = Slice(0, x_complex_in)
                    var idxes = _get_cascade_idxes[
                        base_out.layout.shape, exclude_t
                    ](batch_prod_idx)
                    var dim_batch_x = base_x.slice[dim_sl, x_comp, exclude](
                        idxes
                    )

                    comptime if config.use_shared_f_total:
                        var lhs = shared_f_lhs.slice[dim_sl, o_comp, exclude](
                            idxes
                        )
                        var rhs = shared_f_rhs.slice[dim_sl, o_comp, exclude](
                            idxes
                        )
                        _run_1d_fft[idx](lhs, rhs, dim_batch_x)
                    elif config.use_global_memory:
                        var lhs = base_out.slice[dim_sl, o_comp, exclude](idxes)
                        var rhs = base_calc.slice[dim_sl, o_comp, exclude](
                            idxes
                        )
                        _run_1d_fft[idx](lhs, rhs, dim_batch_x)
                    else:
                        var local_idx = Int(local_i)
                        if local_idx < dim:
                            var dim_batch_out = base_out.slice[
                                dim_sl, o_comp, exclude
                            ](idxes)
                            comptime if idx == config.start_dim_idx:
                                _run_1d_fft[idx](
                                    shared_f_lhs, shared_f_rhs, dim_batch_x
                                )
                            else:
                                _run_1d_fft[idx, use_x=True](
                                    shared_f_lhs, shared_f_rhs, dim_batch_out
                                )

                            var c_num = shared_f_lhs.load[2](local_idx, 0)
                            dim_batch_out.store(local_idx, 0, c_num)
                        stage_sync_fn()

            comptime if config.use_shared_f_total:
                _copy_to_output[0](base_out, shared_f_lhs, shared_f_rhs)

    comptime batched_iters = max(config.batches // config.batch_size, 1)

    comptime for i in range(batched_iters):
        var offset = Int(block_num + i * config.batch_size)
        var base_x = base_x_t(x.ptr + x.stride[0]() * offset)
        var base_out = out_t(output.ptr + output.stride[0]() * offset)
        var base_calc = out_t(calc_buf.ptr + calc_buf.stride[0]() * offset)
        _run_ndim_fft(base_out, base_calc, base_x)
        stage_sync_fn()

    comptime full_iters = batched_iters * config.batch_size
    comptime remainder = config.batches - full_iters

    comptime if remainder > 0:
        if block_num < remainder:
            var offset = Int(full_iters + block_num)
            var base_x = base_x_t(x.ptr + x.stride[0]() * offset)
            var base_out = out_t(output.ptr + output.stride[0]() * offset)
            var base_calc = out_t(calc_buf.ptr + calc_buf.stride[0]() * offset)
            _run_ndim_fft(base_out, base_calc, base_x)
        stage_sync_fn()


@fieldwise_init
struct _GPUTest(Movable):
    comptime BLOCK = Self(0)
    comptime WARP = Self(1)
    comptime DEVICE_WIDE = Self(2)
    comptime CLUSTER = Self(3)
    var v: UInt


def _run_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    runtime_twfs: Bool,
    max_cluster_size: UInt = 8,
    test: Optional[_GPUTest] = None,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout, ...],
    x: LayoutTensor[mut=False, in_dtype, in_layout, ...],
    ctx: DeviceContext,
    var plan_in: Optional[
        _GPUPlan[
            out_dtype,
            out_layout,
            inverse,
            bases,
            test,
            ctx.default_device_info,
            max_cluster_size,
            runtime_twfs,
        ]
    ] = None,
) raises:
    comptime assert (
        out_dtype.is_floating_point()
    ), "out_dtype must be floating point"
    comptime assert (
        has_accelerator()
    ), "The non-cpu implementation is for GPU only"
    var plan = plan_in^.or_else({ctx})

    comptime run_warp = plan.config.can_run_in_warp

    @always_inline
    def cluster_stage_sync_fn():
        cluster_arrive_relaxed()
        cluster_wait()

    @always_inline
    def block_or_warp_stage_sync_fn():
        comptime if not run_warp:
            barrier()

    comptime stage_sync_fn = block_or_warp_stage_sync_fn if (
        plan.config.can_run_in_block or plan.config.can_run_in_warp
    ) else cluster_stage_sync_fn
    comptime address_space = AddressSpace.SHARED if (
        plan.config.can_run_in_block or plan.config.can_run_in_warp
    ) else AddressSpace.SHARED_CLUSTER

    comptime twf_layout = Layout.row_major(2 * plan.config.prod)
    var twiddle_factors: LayoutTensor[
        mut=False, out_dtype, twf_layout, MutAnyOrigin
    ]
    comptime if not plan.config.inline_twfs:
        twiddle_factors = {plan.twfs_buffer[0].unsafe_ptr()}
    else:
        twiddle_factors = {unsafe_ptr = {}}

    comptime og = origin_of(plan.calc_buf[0])
    var calc_buf: LayoutTensor[out_dtype, output.layout, og]
    comptime if plan.config.use_global_memory:
        calc_buf = {plan.calc_buf[0].unsafe_ptr().unsafe_origin_cast[og]()}
    else:
        calc_buf = {unsafe_ptr = {}}

    # TODO: this should get the ordered bases for all dims and iterate over them
    comptime block_func_batch = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        in_layout=x.layout,
        out_layout=output.layout,
        in_origin=x.origin,
        out_origin=output.origin,
        twf_layout=twiddle_factors.layout,
        twf_origin=twiddle_factors.origin,
        calc_buf_origin=calc_buf.origin,
        inverse=inverse,
        bases=bases,
        config=plan.config,
        stage_sync_fn=stage_sync_fn,
        shared_address_space=address_space,
        runtime_twfs=runtime_twfs,
    ]

    comptime grid_dim = (Int(plan.config.num_blocks), plan.config.batch_size)

    comptime run_cluster = plan.config.can_run_in_block_cluster and (
        plan.config.num_blocks > 1
    )
    comptime shared_mem = plan.config.buf_size_full_output if (
        plan.config.buf_size_full_output <= plan.config.max_shared_mem_size
    ) else plan.config.buf_size_max_dim
    ctx.enqueue_function[block_func_batch, block_func_batch](
        output,
        x,
        twiddle_factors,
        calc_buf,
        grid_dim=grid_dim,
        cluster_dim=OptionalReg[Dim](
            plan.config.num_blocks
        ) if run_cluster else None,
        shared_mem_bytes=OptionalReg[Int](
            Int(shared_mem)
        ) if run_cluster else None,
        block_dim=plan.config.block_threads,
    )
