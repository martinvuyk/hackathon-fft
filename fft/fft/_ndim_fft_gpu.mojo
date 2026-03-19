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
from std.math import ceil
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
)
from ._fft import _radix_n_fft_kernel, _radix_n_fft_kernel_spread


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
    comptime rank = Self.out_layout.rank()
    comptime dims = Self.out_layout.shape[1 : Self.rank - 1]
    comptime prod = _product_of_dims(Self.dims)
    """The product of the dimensions in the tensor."""

    comptime max_dim_base = _max(Self.dims)
    """The smallest base for the biggest dimension in the tensor."""
    comptime num_consumer_threads = Self.max_dim_base
    """The amount of threads that will consume values (run the fft)."""
    # TODO: deactivating this for now because I can't get it to work
    comptime num_producer_threads = 0
    """The amount of threads that will produce values (deal with memory
    interop)."""
    comptime circular_buffer_multiple = 1
    """The multiple of the individually sized tensor to reserve for a circular
    buffer."""

    # TODO: this probably requires some smarter use of grid_dim instead of
    # just scheduling all as a unit, since they should run separately
    comptime num_threads = Self.num_consumer_threads + Self.num_producer_threads
    """The total number of threads per worload."""
    comptime max_dim = _max(Self.dims)
    """The biggest dimension in the tensor."""

    comptime batches = UInt(Self.out_layout.shape[0].value())
    """The total amount of batches in the workload."""

    comptime max_threads_per_block = UInt(Self.gpu_info.max_thread_block_size)
    comptime threads_per_m = Self.gpu_info.threads_per_multiprocessor
    comptime max_threads_available = UInt(
        Self.threads_per_m * Self.gpu_info.sm_count
    )

    comptime num_blocks = UInt(
        ceil(
            Float64(Self.num_threads) / Float64(Self.max_threads_per_block)
        ).cast[DType.uint]()
    )
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
        size_of[Self.out_dtype]()
        * Layout.row_major(
            Int(Self.max_dim), 2, Self.circular_buffer_multiple
        ).size()
    )
    comptime buf_size_full_output = UInt(
        size_of[Self.out_dtype]()
        * Layout.row_major(Self.prod, 2, Self.circular_buffer_multiple).size()
    )
    comptime warp_size = UInt(Self.gpu_info.warp_size)

    comptime can_run_in_warp = Self.num_threads <= Self.warp_size and (
        Self.buf_size_full_output <= Self.shared_mem_per_warp
        or Self.buf_size_max_dim <= Self.shared_mem_per_warp
    ) and (Self.test.or_else(_GPUTest.WARP).v == _GPUTest.WARP.v)

    comptime can_run_in_block = Self.num_threads <= (
        Self.max_threads_per_block
    ) and (
        Self.buf_size_full_output <= Self.shared_mem_per_block
        or Self.buf_size_max_dim <= Self.shared_mem_per_block
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
    ) and (
        Self.buf_size_full_output <= Self.shared_mem_per_cluster
        or Self.buf_size_max_dim <= Self.shared_mem_per_cluster
    ) and Self.is_sm_90_or_newer and (
        Self.test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    comptime block_threads = UInt(
        ceil(Float64(Self.num_threads) / Float64(Self.num_blocks)).cast[
            DType.uint
        ]()
    )
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime batch_size = min(Self.batches, Self.thread_batch_size)

    comptime inline_twfs = (
        Int(Self.max_dim) * size_of[Self.out_dtype]() * 2
    ) <= Self.gpu_info.max_registers_per_block // 2

    var twfs_buffer: InlineArray[
        DeviceBuffer[Self.out_dtype], Int(not Self.inline_twfs)
    ]
    var inter_layer_buf: InlineArray[
        DeviceBuffer[Self.out_dtype], Int(len(Self.bases) > 1)
    ]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime assert Self.threads_per_m > 0, (
            "Unknown number of threads per sm for the given device. "
            "It is needed in order to run the gpu implementation."
        )

        comptime rank = Self.out_layout.rank()
        comptime dims = Self.out_layout.shape[1 : rank - 1]
        comptime prod = _product_of_dims(dims)

        # When running ffts on multiple dimensions, we need to copy the output of
        # each dimension into an intermediate buffer for reordering
        comptime out_b_layout = Layout.row_major(Self.out_layout.shape[1:])
        comptime b_size = out_b_layout.size()

        comptime if Self.inline_twfs and len(dims) == 1:
            return {{uninitialized = True}, {uninitialized = True}}
        elif Self.inline_twfs:
            return {
                {uninitialized = True},
                [ctx.enqueue_create_buffer[Self.out_dtype](b_size)],
            }

        var twfs = ctx.enqueue_create_buffer[Self.out_dtype](2 * prod)
        var offset = UInt(0)

        comptime for idx in range(len(Self.bases)):
            comptime length = UInt(dims[idx].value())
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
            ctx.enqueue_copy(
                view,
                twfs_array_runtime.unsafe_ptr().bitcast[
                    Scalar[Self.out_dtype]
                ](),
            )
            offset += complex_len

        comptime if len(dims) == 1:
            return {[twfs^], {uninitialized = True}}
        else:
            return {
                [twfs^],
                [ctx.enqueue_create_buffer[Self.out_dtype](b_size)],
            }


def _intra_something_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_base: UInt,
    max_dim: UInt,
    max_dim_base: UInt,
    total_threads: UInt,
    max_shared_mem_size: UInt,
    shared_address_space: AddressSpace,
    stage_sync_fn: fn(),
    runtime_twfs: Bool,
    inline_twfs: Bool,
    batches: UInt,
    batch_size: UInt,
    circular_buffer_multiple: Int,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    var local_i = thread_idx.x
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y

    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    comptime worker_threads = max_dim

    comptime base_out_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, base_out_layout, ...]
    comptime base_out_t = out_t[
        output.origin, address_space=output.address_space
    ]

    comptime base_x_layout = Layout.row_major(x.layout.shape[1:])
    comptime base_x_t = LayoutTensor[
        in_dtype, base_x_layout, x.origin, address_space=x.address_space
    ]

    comptime shared_f_total_layout = Layout.row_major(
        IntTuple(circular_buffer_multiple, output.layout.shape[1:]).flatten()
    )
    comptime size = UInt(size_of[out_dtype]() * shared_f_total_layout.size())
    comptime shared_f_total_t[layout: Layout] = type_of(
        LayoutTensor[
            out_dtype,
            layout,
            MutExternalOrigin,
            address_space=shared_address_space,
        ].stack_allocation()
    )
    var shared_f_total_lhs: shared_f_total_t[shared_f_total_layout]
    var shared_f_total_rhs: shared_f_total_t[shared_f_total_layout]
    comptime max_dim_layout = Layout.row_major(
        circular_buffer_multiple, Int(max_dim), 2
    )
    comptime max_dim_size = UInt(size_of[out_dtype]() * max_dim_layout.size())
    comptime shared_f_max_dim_t[layout: Layout] = type_of(
        LayoutTensor[
            out_dtype,
            layout,
            MutExternalOrigin,
            address_space=shared_address_space,
        ].stack_allocation()
    )
    var shared_f_max_dim_lhs: shared_f_max_dim_t[max_dim_layout]
    var shared_f_max_dim_rhs: shared_f_max_dim_t[max_dim_layout]

    comptime run_inplace = 2 * size > max_shared_mem_size and (
        2 * max_dim_size > max_shared_mem_size
    )

    comptime if 2 * size <= max_shared_mem_size:  # stockham fft
        shared_f_total_lhs = type_of(shared_f_total_lhs).stack_allocation()
        shared_f_total_rhs = type_of(shared_f_total_rhs).stack_allocation()
        shared_f_max_dim_lhs = {unsafe_ptr = {}}
        shared_f_max_dim_rhs = {unsafe_ptr = {}}
    elif size <= max_shared_mem_size:  # cooley-tukey fft
        shared_f_total_lhs = type_of(shared_f_total_lhs).stack_allocation()
        shared_f_total_rhs = {unsafe_ptr = {}}
        shared_f_max_dim_lhs = {unsafe_ptr = {}}
        shared_f_max_dim_rhs = {unsafe_ptr = {}}
    elif 2 * max_dim_size <= max_shared_mem_size:  # stockham fft
        shared_f_total_lhs = {unsafe_ptr = {}}
        shared_f_total_rhs = {unsafe_ptr = {}}
        shared_f_max_dim_lhs = type_of(shared_f_max_dim_lhs).stack_allocation()
        shared_f_max_dim_rhs = type_of(shared_f_max_dim_rhs).stack_allocation()
    elif max_dim_size <= max_shared_mem_size:  # cooley-tukey fft
        shared_f_total_lhs = {unsafe_ptr = {}}
        shared_f_total_rhs = {unsafe_ptr = {}}
        shared_f_max_dim_lhs = type_of(shared_f_max_dim_lhs).stack_allocation()
        shared_f_max_dim_rhs = {unsafe_ptr = {}}
    else:  # cooley-tukey on global memory
        # TODO: stockham on global memory when possible with plan allocated buffer
        shared_f_total_lhs = {unsafe_ptr = {}}
        shared_f_total_rhs = {unsafe_ptr = {}}
        shared_f_max_dim_lhs = {unsafe_ptr = {}}
        shared_f_max_dim_rhs = {unsafe_ptr = {}}

    @parameter
    def _reorder_kernel[
        length: UInt, do_rfft: Bool, ordered_bases: List[UInt]
    ](
        shared_f: LayoutTensor[mut=True, out_dtype, ...],
        x: LayoutTensor[mut=False, ...],
    ):
        # Reorder input x(local_i) items to match F(current_item) layout.
        var i = Int(_mixed_radix_digit_reverse[length, ordered_bases](local_i))
        comptime w = 1 if do_rfft else 2
        shared_f.store(i, 0, x.load[w](Int(local_i), 0).cast[out_dtype]())

    @parameter
    def _run_1d_fft[
        dim_idx: Int
    ](
        shared_f_lhs: LayoutTensor[mut=True, out_dtype, ...],
        shared_f_rhs: LayoutTensor[mut=True, out_dtype, ...],
        x: LayoutTensor[mut=False, ...],
    ):
        comptime length = UInt(x.layout.shape[0].value())
        comptime do_rfft = dim_idx == start_dim_idx and x_complex_in == 1
        comptime bases_processed = materialize[
            _get_ordered_bases_processed_list[length, bases[dim_idx]]()
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]

        comptime if run_inplace:
            _reorder_kernel[length, do_rfft, ordered_bases](shared_f_lhs, x)
            stage_sync_fn()

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime threads_for_base = length
            comptime func = _radix_n_fft_kernel_spread[
                ...,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                ordered_bases=ordered_bases,
                runtime_twfs=runtime_twfs,
                inline_twfs=inline_twfs,
                run_inplace=run_inplace,
            ]

            @parameter
            def _run():
                comptime if not run_inplace and b == 0:
                    func(shared_f_lhs, x, global_i, twiddle_factors)
                elif run_inplace or b % 2 == 0:
                    func(shared_f_lhs, shared_f_rhs, global_i, twiddle_factors)
                else:
                    func(shared_f_rhs, shared_f_lhs, global_i, twiddle_factors)

            comptime if threads_for_base == total_threads:
                _run()
            else:
                if global_i < threads_for_base:
                    _run()

            stage_sync_fn()

    @parameter
    def _copy_shared_f_total_to_output[
        dim_idx: Int
    ](
        base_out: base_out_t,
        shared_f_lhs: shared_f_total_t[base_out_layout],
        shared_f_rhs: shared_f_total_t[base_out_layout],
    ):
        comptime flat_layout = Layout.row_major(prod, 2)
        comptime flat_t = LayoutTensor[out_dtype, flat_layout, ...]
        comptime batch_prod = UInt(prod) // max_dim
        comptime length = UInt(shared_f_lhs.layout.shape[0].value())
        comptime bases_processed = materialize[
            _get_ordered_bases_processed_list[length, bases[dim_idx]]()
        ]()
        comptime is_in_lhs = run_inplace or len(bases_processed[0]) % 2 == 1

        comptime for i in range(batch_prod):
            comptime batch_prod_offset = i * max_dim

            if local_i < max_dim:
                var b_idx = Int(batch_prod_offset + local_i)
                var num: SIMD[out_dtype, 2]

                comptime if is_in_lhs:
                    num = flat_t(shared_f_lhs.ptr).load[width=2](b_idx, 0)
                else:
                    num = flat_t(shared_f_rhs.ptr).load[width=2](b_idx, 0)
                flat_t(base_out.ptr).store(b_idx, 0, num)

    comptime amnt_circular_idxs = 1

    @parameter
    def _run_ndim_fft(
        base_out: base_out_t, base_x: base_x_t, mut circular_idx: Int
    ):
        comptime if len(dims) == 1:
            comptime if size <= max_shared_mem_size:
                comptime circular_offset = base_out_layout.size()
                var sliced_lhs = shared_f_total_t[base_out_layout](
                    shared_f_total_lhs.ptr + circular_idx * circular_offset
                )
                var sliced_rhs = shared_f_total_t[base_out_layout](
                    shared_f_total_rhs.ptr + circular_idx * circular_offset
                )
                # TODO: this could be parallelized using a circular buffer
                # and a producer consumer setup
                # if global_i < worker_threads:
                _run_1d_fft[start_dim_idx](sliced_lhs, sliced_rhs, base_x)
                # else:
                _copy_shared_f_total_to_output[start_dim_idx](
                    base_out, sliced_lhs, sliced_rhs
                )
                circular_idx = (circular_idx + 1) % amnt_circular_idxs
                # stage_sync_fn()
            else:
                _run_1d_fft[start_dim_idx](
                    base_out, base_out.as_any_origin(), base_x
                )
        else:
            comptime for idx in reversed(range(len(dims))):
                comptime dim = dims[idx].value()
                comptime batch_prod = prod // dim

                comptime exclude = (idx, rank - 2)
                comptime exclude_t = IntTuple(idx, rank - 2)
                comptime dim_sl = Slice(0, dim)
                comptime o_comp = Slice(0, 2)
                comptime _idx_fn = _get_cascade_idxes[
                    base_out.layout.shape, exclude_t
                ]

                for batch_prod_idx in range(batch_prod):
                    var idxes = _idx_fn(batch_prod_idx)

                    @parameter
                    def _run(
                        dim_batch_out_lhs: LayoutTensor[
                            mut=True, out_dtype, ...
                        ],
                        dim_batch_out_rhs: LayoutTensor[
                            mut=True, out_dtype, ...
                        ],
                    ):
                        comptime if idx == start_dim_idx:
                            comptime x_comp = Slice(0, x_complex_in)
                            var dim_batch_x = base_x.slice[
                                dim_sl, x_comp, exclude
                            ](idxes)
                            _run_1d_fft[idx](
                                dim_batch_out_lhs,
                                dim_batch_out_rhs,
                                dim_batch_x,
                            )
                        else:
                            var dim_batch_inter_out = base_out.slice[
                                dim_sl, o_comp, exclude
                            ](idxes).get_immutable()
                            _run_1d_fft[idx](
                                dim_batch_out_lhs,
                                dim_batch_out_rhs,
                                dim_batch_inter_out,
                            )

                    comptime if size <= max_shared_mem_size:
                        comptime circular_offset = base_out_layout.size()
                        # if global_i < worker_threads:
                        var sliced_lhs = shared_f_total_t[base_out_layout](
                            shared_f_total_lhs.ptr
                            + circular_idx * circular_offset
                        )
                        var sliced_rhs = shared_f_total_t[base_out_layout](
                            shared_f_total_rhs.ptr
                            + circular_idx * circular_offset
                        )
                        _run(
                            sliced_lhs.slice[dim_sl, o_comp, exclude](idxes),
                            sliced_lhs.slice[dim_sl, o_comp, exclude](idxes),
                        )
                    elif max_dim_size <= max_shared_mem_size:
                        comptime slice_layout = Layout.row_major(
                            Int(max_dim), 2
                        )
                        comptime circular_offset = slice_layout.size()
                        var sliced_lhs = shared_f_max_dim_t[slice_layout](
                            shared_f_max_dim_lhs.ptr
                            + circular_idx * circular_offset
                        )
                        var sliced_rhs = shared_f_max_dim_t[slice_layout](
                            shared_f_max_dim_rhs.ptr
                            + circular_idx * circular_offset
                        )
                        # if global_i < worker_threads:
                        _run(sliced_lhs, sliced_rhs)
                        # else:
                        var dim_batch_inter_out = base_out.slice[
                            dim_sl, o_comp, exclude
                        ](idxes)

                        var local_idx = Int(local_i)
                        comptime copy_idx = (worker_threads - 1)

                        comptime if copy_idx >= UInt(dim):
                            if local_idx >= dim:
                                break

                        comptime length = UInt(
                            sliced_lhs.layout.shape[0].value()
                        )
                        comptime bases_processed = materialize[
                            _get_ordered_bases_processed_list[
                                length, bases[idx]
                            ]()
                        ]()
                        comptime is_in_lhs = run_inplace or (
                            len(bases_processed[0]) % 2 == 1
                        )
                        var c_num: SIMD[out_dtype, 2]

                        comptime if is_in_lhs:
                            c_num = sliced_lhs.load[2](local_idx, 0)
                        else:
                            c_num = sliced_rhs.load[2](local_idx, 0)
                        dim_batch_inter_out.store(local_idx, 0, c_num)
                        circular_idx = (circular_idx + 1) % amnt_circular_idxs
                        stage_sync_fn()
                    else:
                        var e = base_out.slice[dim_sl, o_comp, exclude](idxes)
                        _run(e, e.as_any_origin())

                comptime if size <= max_shared_mem_size:
                    # if global_i >= worker_threads:
                    comptime circular_offset = base_out_layout.size()
                    var sliced_lhs = shared_f_total_t[base_out_layout](
                        shared_f_total_lhs.ptr + circular_idx * circular_offset
                    )
                    var sliced_rhs = shared_f_total_t[base_out_layout](
                        shared_f_total_rhs.ptr + circular_idx * circular_offset
                    )
                    _copy_shared_f_total_to_output[idx](
                        base_out, sliced_lhs, sliced_rhs
                    )
                    circular_idx = (circular_idx + 1) % amnt_circular_idxs
                    stage_sync_fn()

    comptime batched_iters = max(batches // batch_size, 1)

    var circular_idx = 0

    comptime for i in range(batched_iters):
        var offset = Int(block_num + i * batch_size)
        var base_x = base_x_t(x.ptr + x.stride[0]() * offset)
        var base_out = out_t(output.ptr + output.stride[0]() * offset)
        _run_ndim_fft(base_out, base_x, circular_idx)
        stage_sync_fn()

    comptime full_iters = batched_iters * batch_size
    comptime remainder = batches - full_iters

    comptime if remainder > 0:
        if block_num < remainder:
            var offset = Int(full_iters + block_num)
            var base_x = base_x_t(x.ptr + x.stride[0]() * offset)
            var base_out = out_t(output.ptr + output.stride[0]() * offset)
            _run_ndim_fft(base_out, base_x, circular_idx)
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

    comptime run_warp = plan.can_run_in_warp

    @always_inline
    def cluster_stage_sync_fn():
        cluster_arrive_relaxed()
        cluster_wait()

    @always_inline
    def block_or_warp_stage_sync_fn():
        comptime if not run_warp:
            barrier()

    comptime stage_sync_fn = block_or_warp_stage_sync_fn if (
        plan.can_run_in_block or plan.can_run_in_warp
    ) else cluster_stage_sync_fn
    comptime address_space = AddressSpace.SHARED if (
        plan.can_run_in_block or plan.can_run_in_warp
    ) else AddressSpace.SHARED_CLUSTER
    comptime max_shared_mem_size = (
        plan.shared_mem_per_warp if (
            plan.can_run_in_warp
        ) else plan.shared_mem_per_block if (
            plan.can_run_in_block
        ) else plan.shared_mem_per_cluster
    )

    comptime twf_layout = Layout.row_major(2 * plan.prod)
    var twiddle_factors: LayoutTensor[
        mut=False, out_dtype, twf_layout, MutAnyOrigin
    ]
    comptime if not plan.inline_twfs:
        twiddle_factors = {plan.twfs_buffer[0].unsafe_ptr()}
    else:
        twiddle_factors = {unsafe_ptr = {}}
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
        twf_address_space=twiddle_factors.address_space,
        inverse=inverse,
        bases=bases,
        max_base=_max(bases),
        max_dim=plan.max_dim,
        max_dim_base=plan.max_dim_base,
        max_shared_mem_size=max_shared_mem_size,
        total_threads=plan.block_threads * plan.num_blocks,
        stage_sync_fn=stage_sync_fn,
        shared_address_space=address_space,
        runtime_twfs=runtime_twfs,
        inline_twfs=plan.inline_twfs,
        batches=plan.batches,
        batch_size=plan.batch_size,
        circular_buffer_multiple=plan.circular_buffer_multiple,
    ]

    comptime grid_dim = (Int(plan.num_blocks), plan.batch_size)

    comptime run_cluster = plan.can_run_in_block_cluster and plan.num_blocks > 1
    comptime shared_mem = plan.buf_size_full_output if (
        plan.buf_size_full_output <= max_shared_mem_size
    ) else plan.buf_size_max_dim
    ctx.enqueue_function[block_func_batch, block_func_batch](
        output,
        x,
        twiddle_factors,
        grid_dim=grid_dim,
        cluster_dim=OptionalReg[Dim](plan.num_blocks) if run_cluster else None,
        shared_mem_bytes=OptionalReg[Int](
            Int(shared_mem)
        ) if run_cluster else None,
        block_dim=plan.block_threads,
    )
