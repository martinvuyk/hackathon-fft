from algorithm import parallelize, vectorize
from builtin.globals import global_constant
from complex import ComplexScalar
from collections import OptionalReg
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext, DeviceBuffer, Dim, DeviceFunction
from gpu.host.info import Vendor, is_cpu, GPUInfo
from layout import Layout, LayoutTensor, IntTuple
from utils.index import IndexList
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, size_of, simd_width_of
from math import ceil
from std.utils import Variant

from ._utils import (
    _get_dtype,
    _get_flat_twfs,
    _get_flat_twfs_inline,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _product_of_dims,
    _get_cascade_idxes,
    _get_flat_twfs_total_offsets,
)
from ._fft import (
    _radix_n_fft_kernel,
    _inter_multiprocessor_fft_kernel_radix_n,
    _radix_n_fft_kernel_exp_twfs_runtime,
    _radix_n_fft_kernel_exp_no_params,
)


@fieldwise_init
struct _GPUDeviceWidePlan[out_dtype: DType](ImplicitlyCopyable):
    var twfs_buffer: Optional[DeviceBuffer[Self.out_dtype]]
    var inter_layer_buf: Optional[DeviceBuffer[Self.out_dtype]]


@fieldwise_init
struct _GPUSharedMemPlan[out_dtype: DType](ImplicitlyCopyable):
    ...


comptime _GPUExecutionPlan[out_dtype: DType] = Variant[
    _GPUDeviceWidePlan[out_dtype], _GPUSharedMemPlan[out_dtype]
]


fn _calc_total_bases_offsets[
    dims: IntTuple, bases: List[List[UInt]]
]() -> Tuple[
    UInt,
    InlineArray[UInt, len(dims)],
    InlineArray[List[UInt], len(dims)],
]:
    var absolute_total = UInt(0)
    var dim_totals = InlineArray[UInt, len(dims)](uninitialized=True)
    var absolute_offsets = InlineArray[List[UInt], len(dims)](
        uninitialized=True
    )

    @parameter
    for idx in range(len(dims)):
        comptime length = UInt(dims[idx].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[idx]
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        ref total_offsets = _get_flat_twfs_total_offsets(
            materialize[ordered_bases](), length, absolute_total
        )
        dim_totals[idx] = total_offsets[0]
        absolute_total += total_offsets[0]
        absolute_offsets[idx] = total_offsets[1].copy()
    return absolute_total, dim_totals^, absolute_offsets^


fn _get_gpu_device_wide_plan[
    out_dtype: DType,
    out_layout: Layout,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
](ctx: DeviceContext) raises -> _GPUDeviceWidePlan[out_dtype]:
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]

    comptime total_bases_offsets = _calc_total_bases_offsets[dims, bases]()
    comptime total_twfs = total_bases_offsets[0]
    comptime dim_total_twfs = total_bases_offsets[1]
    comptime absolute_offsets = total_bases_offsets[2]
    comptime twf_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = ctx.enqueue_create_buffer[out_dtype](twf_layout.size())

    @parameter
    for idx in range(len(bases)):
        comptime length = UInt(dims[idx].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[idx]
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        comptime dim_total_twfs_idx = dim_total_twfs[idx]
        comptime twfs_array = _get_flat_twfs[
            out_dtype,
            length,
            dim_total_twfs_idx,
            ordered_bases,
            processed_list,
            inverse,
        ]()
        comptime absolute_offsets_idx_start = absolute_offsets[idx][0]
        var view = DeviceBuffer(
            ctx,
            twfs.unsafe_ptr() + absolute_offsets_idx_start * 2,
            Int(dim_total_twfs_idx) * 2,
            owning=False,
        )
        # FIXME(#5686): replace with this once it's solved
        # ref twfs_array_runtime = global_constant[twfs_array]()
        var twfs_array_runtime = materialize[twfs_array]()
        ctx.enqueue_copy(view, twfs_array_runtime.unsafe_ptr())

    @parameter
    if len(dims) == 1:
        return {twfs, None}

    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    comptime out_b_layout = Layout.row_major(out_layout.shape[1:])
    return {twfs, ctx.enqueue_create_buffer[out_dtype](out_b_layout.size())}


struct _GPUPlan[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
](ImplicitlyCopyable):
    comptime rank = Self.out_layout.rank()
    comptime dims = Self.out_layout.shape[1 : Self.rank - 1]
    comptime prod = _product_of_dims(Self.dims)

    @staticmethod
    fn _find_max_threads(out threads_base: Tuple[UInt, UInt]):
        threads_base = {0, 0}

        @parameter
        for i, base_set in enumerate(Self.bases):
            comptime val = _min(base_set)
            comptime dim = UInt(Self.dims[i].value())
            comptime threads = dim // val
            if threads_base[0] < threads:
                threads_base = threads, val

    comptime threads_base = Self._find_max_threads()
    comptime num_threads = Self.threads_base[0]
    comptime max_dim_base = Self.threads_base[1]
    comptime max_dim = Self.num_threads * Self.max_dim_base

    comptime batches = UInt(Self.out_layout.shape[0].value())

    comptime max_threads_per_block = UInt(Self.gpu_info.max_thread_block_size)
    comptime threads_per_m = Self.gpu_info.threads_per_multiprocessor
    comptime max_threads_available = UInt(
        Self.threads_per_m * Self.gpu_info.sm_count
    )

    comptime num_blocks = UInt(
        ceil(Self.num_threads / Self.max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_per_m = UInt(
        Self.gpu_info.shared_memory_per_multiprocessor
    )
    comptime shared_mem_per_t = Self.shared_mem_per_m // UInt(
        Self.threads_per_m
    )
    comptime shared_mem_per_block = Self.shared_mem_per_t * Self.max_threads_per_block
    comptime shared_mem_per_warp = Self.shared_mem_per_t * Self.warp_size
    comptime shared_mem_per_cluster = Self.shared_mem_per_block * Self.max_cluster_size
    comptime out_size_max_dim = UInt(
        size_of[Self.out_dtype]()
    ) * Self.max_dim * 2
    comptime full_output_size = UInt(size_of[Self.out_dtype]() * Self.prod * 2)
    comptime warp_size = UInt(Self.gpu_info.warp_size)

    comptime can_run_in_warp = Self.num_threads <= Self.warp_size and (
        Self.full_output_size <= Self.shared_mem_per_warp
        or Self.out_size_max_dim <= Self.shared_mem_per_warp
    ) and (Self.test.or_else(_GPUTest.WARP).v == _GPUTest.WARP.v)

    comptime can_run_in_block = Self.num_threads <= Self.max_threads_per_block and (
        Self.full_output_size <= Self.shared_mem_per_block
        or Self.out_size_max_dim <= Self.shared_mem_per_block
    ) and (
        Self.test.or_else(_GPUTest.BLOCK).v
        in (_GPUTest.BLOCK.v, _GPUTest.WARP.v)
    )

    comptime is_sm_90_or_newer = (
        Self.gpu_info.vendor == Vendor.NVIDIA_GPU
        and Self.gpu_info.compute >= 9.0
    )
    comptime can_run_in_block_cluster = Self.num_blocks <= Self.max_cluster_size and (
        Self.full_output_size <= Self.shared_mem_per_cluster
        or Self.out_size_max_dim <= Self.shared_mem_per_cluster
    ) and Self.is_sm_90_or_newer and (
        Self.test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    comptime block_threads = UInt(
        ceil(Self.num_threads / Self.num_blocks).cast[DType.uint]()
    )
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime batch_size = min(Self.batches, Self.thread_batch_size)
    comptime run_intra_something = Self.can_run_in_warp or Self.can_run_in_block or Self.can_run_in_block_cluster

    var execution_plan: _GPUExecutionPlan[Self.out_dtype]

    fn __init__(
        ctx: DeviceContext,
        out self: _GPUPlan[
            Self.out_dtype,
            Self.out_layout,
            Self.inverse,
            Self.bases,
            Self.test,
            ctx.default_device_info,
            Self.max_cluster_size,
        ],
    ) raises:
        __comptime_assert Self.threads_per_m > 0, (
            "Unknown number of threads per sm for the given device. "
            "It is needed in order to run the gpu implementation."
        )

        @parameter
        if Self.run_intra_something:
            self.execution_plan = _GPUSharedMemPlan[Self.out_dtype]()
        else:
            self.execution_plan = _get_gpu_device_wide_plan[
                Self.out_dtype,
                Self.out_layout,
                inverse = Self.inverse,
                bases = Self.bases,
            ](ctx)


fn _fft_gpu_device_wide[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_cluster_size: UInt,
    grid_dim: Tuple[Int, Int],
    block_threads: UInt,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin, ...],
    x: LayoutTensor[in_dtype, in_layout, in_origin, ...],
    ctx: DeviceContext,
    plan: _GPUDeviceWidePlan[out_dtype],
) raises:
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime amnt_dims = len(bases)
    comptime batches = out_layout.shape[0].value()
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    comptime total_bases_offsets = _calc_total_bases_offsets[dims, bases]()
    comptime total_twfs = total_bases_offsets[0]
    comptime absolute_offsets = total_bases_offsets[2]
    comptime twf_layout = Layout.row_major(Int(total_twfs), 2)
    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        plan.twfs_buffer.value().unsafe_ptr()
    )

    @parameter
    fn _run_1d_fft[
        batch_x_dtype: DType,
        batch_out_layout: Layout,
        batch_x_layout: Layout,
        batch_x_origin: ImmutOrigin,
        //,
        dim_idx: Int,
    ](
        batch_output: LayoutTensor[
            out_dtype, batch_out_layout, output.origin, ...
        ],
        batch_x: LayoutTensor[
            batch_x_dtype, batch_x_layout, batch_x_origin, ...
        ],
    ) raises:
        comptime length = UInt(batch_x.layout.shape[0].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx]
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        comptime func[b: Int] = _inter_multiprocessor_fft_kernel_radix_n[
            batch_x.dtype,
            out_dtype,
            batch_x.layout,
            batch_output.layout,
            twiddle_factors.layout,
            batch_x.origin,
            output.origin,
            twiddle_factors.origin,
            twiddle_factors.address_space,
            length=length,
            base = ordered_bases[b],
            ordered_bases=ordered_bases,
            processed = processed_list[b],
            do_rfft = dim_idx == start_dim_idx and x_complex_in == 1,
            inverse=inverse,
            twf_offset = absolute_offsets[dim_idx][b],
        ]

        @parameter
        for b in range(len(ordered_bases)):
            ctx.enqueue_function[func[b], func[b]](
                batch_output,
                batch_x,
                twiddle_factors,
                grid_dim=grid_dim,
                block_dim=block_threads,
            )

    comptime out_b_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_b_t = LayoutTensor[out_dtype, out_b_layout, address_space=_]
    comptime size = out_b_layout.size()
    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    var inter_layer_buf = plan.inter_layer_buf.value()

    for block_num in range(batches):
        var block_offset = output.stride[0]() * block_num
        var base_out = out_b_t(output.ptr + block_offset)
        var base_inter_out = out_b_t(inter_layer_buf)
        comptime base_x_layout = Layout.row_major(x.layout.shape[1:])
        var base_x = LayoutTensor[in_dtype, base_x_layout, address_space=_](
            x.ptr + x.stride[0]() * block_num
        )

        @parameter
        if len(dims) == 1:
            _run_1d_fft[start_dim_idx](base_out, base_x)
        else:

            @parameter
            for idx in reversed(range(len(dims))):
                comptime dim_tuple = dims[idx]
                comptime dim = dim_tuple.value()
                comptime batch_prod = prod // dim

                @parameter
                if idx != start_dim_idx:
                    var out_buf = DeviceBuffer(
                        ctx, ptr=base_out.ptr, size=size, owning=False
                    )
                    inter_layer_buf.enqueue_copy_from(out_buf)

                for flat_idx in range(batch_prod):
                    comptime exclude = (idx, rank - 2)
                    comptime exclude_t = IntTuple(idx, rank - 2)
                    comptime dim_sl = Slice(0, dim)
                    comptime o_comp = Slice(0, 2)
                    comptime dims_comp = base_out.layout.shape
                    var idxes = _get_cascade_idxes[dims_comp, exclude_t](
                        flat_idx
                    )
                    var dim_batch_out = base_out.slice[dim_sl, o_comp, exclude](
                        idxes
                    )

                    @parameter
                    if idx == start_dim_idx:
                        comptime x_comp = Slice(0, x_complex_in)
                        var dim_batch_x = base_x.slice[dim_sl, x_comp, exclude](
                            idxes
                        )
                        _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                    else:
                        var dim_batch_inter_out = base_inter_out.slice[
                            dim_sl, o_comp, exclude
                        ](idxes).get_immutable()
                        _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)


fn _intra_something_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_base: UInt,
    max_dim: UInt,
    max_dim_base: UInt,
    total_threads: UInt,
    max_shared_mem_size: UInt,
    shared_address_space: AddressSpace,
    stage_sync_fn: fn (),
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `biggest_dimension // smallest_base <=
    max_threads_per_multiprocessor` and that `x_dim * y_dim [* z_dim]` out_dtype
    items fit in the thread group's shared memory."""

    var local_i = block_idx.x + thread_idx.x
    var block_num = block_idx.y

    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    comptime o_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, o_layout, address_space=_]
    comptime size = UInt(
        _product_of_dims(o_layout.shape) * size_of[out_dtype]()
    )
    var shared_f_total: type_of(
        out_t[
            MutExternalOrigin, address_space=shared_address_space
        ].stack_allocation()
    )
    var shared_f_max_dim: type_of(
        LayoutTensor[
            out_dtype,
            Layout.row_major(Int(max_dim), 2),
            MutExternalOrigin,
            address_space=shared_address_space,
        ].stack_allocation()
    )

    @parameter
    if size <= max_shared_mem_size:
        shared_f_total = type_of(shared_f_total).stack_allocation()
        shared_f_max_dim = {unsafe_ptr = {}}
    else:
        shared_f_total = {unsafe_ptr = {}}
        shared_f_max_dim = type_of(shared_f_max_dim).stack_allocation()

    comptime x_out_layout = Layout.row_major(Int(max_base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutExternalOrigin
    ].stack_allocation()

    @parameter
    fn _run_1d_fft[
        dim_idx: Int
    ](
        shared_f: LayoutTensor[mut=True, out_dtype, ...],
        x: LayoutTensor[mut=False, address_space=_, ...],
    ):
        comptime length = UInt(x.layout.shape[0].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx]
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime threads_for_base = length // base
            comptime func = _radix_n_fft_kernel_exp_twfs_runtime[
                do_rfft = dim_idx == start_dim_idx and x_complex_in == 1,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                ordered_bases=ordered_bases,
            ]

            @parameter
            if threads_for_base == total_threads:
                func(shared_f, x, local_i, x_out)
            else:
                if local_i < threads_for_base:
                    func(shared_f, x, local_i, x_out)

            stage_sync_fn()

    comptime x_layout = Layout.row_major(x.layout.shape[1:])
    var base_x = LayoutTensor[in_dtype, x_layout, address_space=_](
        x.ptr + x.stride[0]() * Int(block_num)
    )
    var base_out = out_t(output.ptr + output.stride[0]() * Int(block_num))

    @parameter
    fn _copy_shared_f_total_to_output():
        comptime flat_layout = Layout.row_major(prod, 2)
        comptime flat_t = LayoutTensor[out_dtype, flat_layout, address_space=_]
        comptime batch_prod = UInt(prod) // max_dim

        for i in range(batch_prod):
            var batch_prod_offset = i * max_dim

            @parameter
            for b in range(max_dim_base):
                var local_idx = Int(local_i * max_dim_base + b)
                var b_idx = Int(batch_prod_offset) + local_idx
                var complex_num = flat_t(shared_f_total.ptr).load[width=2](
                    b_idx, 0
                )
                flat_t(base_out.ptr).store(b_idx, 0, complex_num)

        stage_sync_fn()

    @parameter
    if len(dims) == 1:
        __comptime_assert size <= max_shared_mem_size, (
            "internal implementation error for the given shape, please file"
            " an issue"
        )
        _run_1d_fft[start_dim_idx](shared_f_total, base_x)
    else:

        @parameter
        for idx in reversed(range(len(dims))):
            comptime dim = dims[idx].value()
            comptime batch_prod = prod // dim

            @parameter
            if idx != start_dim_idx and size <= max_shared_mem_size:
                _copy_shared_f_total_to_output()

            comptime exclude = (idx, rank - 2)
            comptime exclude_t = IntTuple(idx, rank - 2)
            comptime dim_sl = Slice(0, dim)
            comptime o_comp = Slice(0, 2)
            comptime dims_comp = base_out.layout.shape

            for batch_prod_idx in range(batch_prod):
                var idxes = _get_cascade_idxes[dims_comp, exclude_t](
                    batch_prod_idx
                )

                @parameter
                fn _run(
                    dim_batch_out: LayoutTensor[
                        mut=True,
                        out_dtype,
                        address_space=shared_address_space,
                        ...,
                    ]
                ):
                    @parameter
                    if idx == start_dim_idx:
                        comptime x_comp = Slice(0, x_complex_in)
                        var dim_batch_x = base_x.slice[dim_sl, x_comp, exclude](
                            idxes
                        )
                        _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                    else:
                        var dim_batch_inter_out = base_out.slice[
                            dim_sl, o_comp, exclude
                        ](idxes).get_immutable()
                        _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)

                @parameter
                if size <= max_shared_mem_size:
                    _run(shared_f_total.slice[dim_sl, o_comp, exclude](idxes))
                else:
                    _run(shared_f_max_dim)
                    var dim_batch_inter_out = base_out.slice[
                        dim_sl, o_comp, exclude
                    ](idxes)

                    @parameter
                    for b in range(max_dim_base):
                        var local_idx = Int(local_i * max_dim_base + b)

                        @parameter
                        if (total_threads - 1) * max_dim_base + b >= (
                            UInt(dim)
                        ):
                            if local_idx >= dim:
                                break
                        var complex_num = shared_f_max_dim.load[width=2](
                            local_idx, 0
                        )
                        dim_batch_inter_out.store(local_idx, 0, complex_num)

                    stage_sync_fn()

    @parameter
    if size <= max_shared_mem_size:
        _copy_shared_f_total_to_output()


@fieldwise_init
struct _GPUTest(Movable):
    comptime BLOCK = Self(0)
    comptime WARP = Self(1)
    comptime DEVICE_WIDE = Self(2)
    comptime CLUSTER = Self(3)
    var v: UInt


fn _run_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_cluster_size: UInt = 8,
    test: Optional[_GPUTest] = None,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout, ...],
    x: LayoutTensor[mut=False, in_dtype, in_layout, ...],
    ctx: DeviceContext,
    plan_in: Optional[
        _GPUPlan[
            out_dtype,
            out_layout,
            inverse,
            bases,
            test,
            ctx.default_device_info,
            max_cluster_size,
        ]
    ] = None,
) raises:
    __comptime_assert (
        out_dtype.is_floating_point()
    ), "out_dtype must be floating point"
    __comptime_assert (
        has_accelerator()
    ), "The non-cpu implementation is for GPU only"
    var plan = plan_in.or_else({ctx})

    @always_inline
    @parameter
    fn _launch_fn[batch_size: Int](offset: Int) raises:
        comptime out_tuple = IntTuple(batch_size, plan.dims, 2)
        comptime out_batch_layout = Layout.row_major(out_tuple.flatten())
        var out_batch = LayoutTensor[
            out_dtype,
            out_batch_layout,
            output.origin,
            address_space = output.address_space,
        ](output.ptr + output.stride[0]() * offset)
        comptime in_complex = in_layout.shape[plan.rank - 1].value()
        comptime x_tuple = IntTuple(batch_size, plan.dims, in_complex)
        comptime x_batch_layout = Layout.row_major(x_tuple.flatten())
        var x_batch = LayoutTensor[
            in_dtype, x_batch_layout, x.origin, address_space = x.address_space
        ](x.ptr + x.stride[0]() * offset)
        comptime run_warp = plan.can_run_in_warp

        fn cluster_stage_sync_fn():
            cluster_arrive_relaxed()
            cluster_wait()

        fn block_or_warp_stage_sync_fn():
            @parameter
            if run_warp:
                barrier()

        comptime stage_sync_fn = block_or_warp_stage_sync_fn if (
            plan.can_run_in_block or plan.can_run_in_warp
        ) else cluster_stage_sync_fn
        comptime address_space = AddressSpace.SHARED if (
            plan.can_run_in_block or plan.can_run_in_warp
        ) else AddressSpace.SHARED_CLUSTER
        comptime max_shared_mem_size = plan.shared_mem_per_warp if (
            plan.can_run_in_warp
        ) else (
            plan.shared_mem_per_block if plan.can_run_in_block else plan.shared_mem_per_cluster
        )

        comptime block_func_batch = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            in_layout = x_batch.layout,
            out_layout = out_batch.layout,
            in_origin = x_batch.origin,
            out_origin = out_batch.origin,
            inverse=inverse,
            bases=bases,
            max_base = _max(bases),
            max_dim = plan.max_dim,
            max_dim_base = plan.max_dim_base,
            max_shared_mem_size=max_shared_mem_size,
            total_threads = plan.block_threads * plan.num_blocks,
            stage_sync_fn=stage_sync_fn,
            shared_address_space=address_space,
        ]

        comptime grid_dim = (Int(plan.num_blocks), batch_size)

        @parameter
        if plan.run_intra_something:
            __comptime_assert (
                plan.full_output_size <= max_shared_mem_size
                or plan.out_size_max_dim <= max_shared_mem_size
            ), (
                "internal implementation error for the given shape, please file"
                " an issue"
            )
            comptime run_cluster = plan.can_run_in_block_cluster and plan.num_blocks > 1
            comptime shared_mem = plan.full_output_size if (
                plan.full_output_size <= max_shared_mem_size
            ) else plan.out_size_max_dim
            ctx.enqueue_function[block_func_batch, block_func_batch](
                out_batch,
                x_batch,
                grid_dim=grid_dim,
                cluster_dim=OptionalReg[Dim](
                    plan.num_blocks
                ) if run_cluster else None,
                shared_mem_bytes=OptionalReg[Int](
                    Int(shared_mem)
                ) if run_cluster else None,
                block_dim=plan.block_threads,
            )
        else:
            _fft_gpu_device_wide[
                max_cluster_size=max_cluster_size,
                inverse=inverse,
                bases=bases,
                grid_dim=grid_dim,
                block_threads = plan.block_threads,
            ](
                output,
                x,
                ctx=ctx,
                plan=plan.execution_plan[_GPUDeviceWidePlan[out_dtype]],
            )

    comptime full_batches = plan.batches // plan.batch_size

    for i in range(full_batches):
        _launch_fn[Int(plan.batch_size)](Int(i * plan.batch_size))

    comptime remainder = plan.batches % plan.batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder)](Int(full_batches * plan.batch_size))
