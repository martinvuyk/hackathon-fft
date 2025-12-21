from algorithm import parallelize, vectorize
from builtin.globals import global_constant
from complex import ComplexScalar
from collections import OptionalReg
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext, DeviceBuffer, Dim
from gpu.host.info import Vendor, is_cpu
from layout import Layout, LayoutTensor, IntTuple
from utils.index import IndexList
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, size_of, simd_width_of
from math import ceil

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
    _radix_n_fft_kernel_exp,
)


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
    output: LayoutTensor[out_dtype, out_layout, out_origin, **_],
    x: LayoutTensor[in_dtype, in_layout, in_origin, **_],
    ctx: DeviceContext,
) raises:
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime amnt_dims = len(bases)
    comptime batches = out_layout.shape[0].value()
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    @parameter
    fn calc_total_bases_offsets() -> (
        Tuple[
            UInt,
            InlineArray[UInt, amnt_dims],
            InlineArray[List[UInt], amnt_dims],
        ]
    ):
        var absolute_total = UInt(0)
        var dim_totals = InlineArray[UInt, amnt_dims](uninitialized=True)
        var absolute_offsets = InlineArray[List[UInt], amnt_dims](
            uninitialized=True
        )

        @parameter
        for idx in range(amnt_dims):
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

    comptime total_bases_offsets = calc_total_bases_offsets()
    comptime total_twfs = total_bases_offsets[0]
    comptime dim_total_twfs = total_bases_offsets[1]
    comptime absolute_offsets = total_bases_offsets[2]
    comptime twf_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = ctx.enqueue_create_buffer[out_dtype](twf_layout.size())

    @parameter
    for idx in range(amnt_dims):
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
            twfs.unsafe_ptr().offset(absolute_offsets_idx_start * 2),
            Int(dim_total_twfs_idx) * 2,
            owning=False,
        )
        # FIXME(#5686): replace with this once it's solved
        # ref twfs_array_runtime = global_constant[twfs_array]()
        var twfs_array_runtime = materialize[twfs_array]()
        ctx.enqueue_copy(view, twfs_array_runtime.unsafe_ptr())

    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
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
            out_dtype, batch_out_layout, output.origin, **_
        ],
        batch_x: LayoutTensor[
            batch_x_dtype, batch_x_layout, batch_x_origin, **_
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
            ctx.enqueue_function_checked[func[b], func[b]](
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
    var inter_layer_buf = ctx.enqueue_create_buffer[out_dtype](
        size * Int(len(dims) > 1)
    )

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
            MutOrigin.external, address_space=shared_address_space
        ].stack_allocation()
    )
    var shared_f_max_dim: type_of(
        LayoutTensor[
            out_dtype,
            Layout.row_major(Int(max_dim), 2),
            MutOrigin.external,
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
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    @parameter
    fn _run_1d_fft[
        dim_idx: Int
    ](
        shared_f: LayoutTensor[mut=True, out_dtype, *_, **_],
        x: LayoutTensor[mut=False, address_space=_, *_, **_],
    ):
        comptime length = UInt(x.layout.shape[0].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx]
        ]()
        comptime ordered_bases = [length]  # bases_processed[0]
        comptime processed_list = [UInt(1)]  # bases_processed[1]
        comptime total_offsets = _get_flat_twfs_total_offsets(
            ordered_bases, length
        )
        comptime total_twfs = total_offsets[0]
        comptime twf_offsets = total_offsets[1]
        # FIXME(#5686): replace with _get_flat_twfs once it's solved
        comptime twfs_array = _get_flat_twfs_inline[
            out_dtype,
            length,
            total_twfs,
            ordered_bases,
            processed_list,
            inverse,
        ]()
        comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
        ref twfs_array_runtime = global_constant[twfs_array]()
        var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
            twfs_array_runtime.unsafe_ptr()
        )

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime threads_for_base = length // base
            comptime func = _radix_n_fft_kernel[
                do_rfft = dim_idx == start_dim_idx and x_complex_in == 1,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
                ordered_bases=ordered_bases,
            ]

            @parameter
            if threads_for_base == total_threads:
                func(shared_f, x, local_i, twfs, x_out)
            else:
                if local_i < threads_for_base:
                    func(shared_f, x, local_i, twfs, x_out)

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
                        *_, **_,
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
    output: LayoutTensor[mut=True, out_dtype, out_layout, **_],
    x: LayoutTensor[mut=False, in_dtype, in_layout, **_],
    ctx: DeviceContext,
) raises:
    __comptime_assert (
        out_dtype.is_floating_point()
    ), "out_dtype must be floating point"
    __comptime_assert (
        has_accelerator()
    ), "The non-cpu implementation is for GPU only"
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)

    @parameter
    fn _find_max_threads(out threads_base: Tuple[UInt, UInt]):
        threads_base = {0, 0}

        @parameter
        for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(dims[i].value())
            comptime threads = dim // val
            if threads_base[0] < threads:
                threads_base = threads, val

    comptime threads_base = _find_max_threads()
    comptime num_threads = threads_base[0]
    comptime max_dim_base = threads_base[1]
    comptime max_dim = num_threads * max_dim_base

    comptime batches = UInt(in_layout.shape[0].value())
    comptime in_complex = in_layout.shape[rank - 1].value()

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_m = gpu_info.threads_per_multiprocessor
    __comptime_assert threads_per_m > 0, (
        "Unknown number of threads per sm for the given device. "
        "It is needed in order to run the gpu implementation."
    )
    comptime max_threads_available = UInt(threads_per_m * gpu_info.sm_count)

    comptime num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_per_m = UInt(gpu_info.shared_memory_per_multiprocessor)
    comptime shared_mem_per_t = shared_mem_per_m // UInt(threads_per_m)
    comptime shared_mem_per_block = shared_mem_per_t * max_threads_per_block
    comptime shared_mem_per_warp = shared_mem_per_t * warp_size
    comptime shared_mem_per_cluster = shared_mem_per_block * max_cluster_size
    comptime out_size_max_dim = UInt(size_of[out_dtype]()) * max_dim * 2
    comptime full_output_size = UInt(size_of[out_dtype]() * prod * 2)
    comptime warp_size = UInt(gpu_info.warp_size)

    comptime can_run_in_warp = num_threads <= warp_size and (
        full_output_size <= shared_mem_per_warp
        or out_size_max_dim <= shared_mem_per_warp
    ) and (test.or_else(_GPUTest.WARP).v == _GPUTest.WARP.v)

    comptime can_run_in_block = num_threads <= max_threads_per_block and (
        full_output_size <= shared_mem_per_block
        or out_size_max_dim <= shared_mem_per_block
    ) and (
        test.or_else(_GPUTest.BLOCK).v in (_GPUTest.BLOCK.v, _GPUTest.WARP.v)
    )

    comptime is_sm_90_or_newer = (
        gpu_info.vendor == Vendor.NVIDIA_GPU and gpu_info.compute >= 9.0
    )
    comptime can_run_in_block_cluster = num_blocks <= max_cluster_size and (
        full_output_size <= shared_mem_per_cluster
        or out_size_max_dim <= shared_mem_per_cluster
    ) and is_sm_90_or_newer and (
        test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    comptime block_threads = UInt(
        ceil(num_threads / num_blocks).cast[DType.uint]()
    )
    comptime thread_batch_size = max_threads_available // num_threads
    comptime batch_size = min(batches, thread_batch_size)

    @parameter
    fn _launch_fn[batch_size: Int](offset: Int) raises:
        comptime out_tuple = IntTuple(batch_size, dims, 2)
        comptime out_batch_layout = Layout.row_major(out_tuple.flatten())
        var out_batch = LayoutTensor[
            out_dtype,
            out_batch_layout,
            output.origin,
            address_space = output.address_space,
        ](output.ptr + output.stride[0]() * offset)
        comptime x_tuple = IntTuple(batch_size, dims, in_complex)
        comptime x_batch_layout = Layout.row_major(x_tuple.flatten())
        var x_batch = LayoutTensor[
            in_dtype, x_batch_layout, x.origin, address_space = x.address_space
        ](x.ptr + x.stride[0]() * offset)

        fn cluster_stage_sync_fn():
            cluster_arrive_relaxed()
            cluster_wait()

        fn block_or_warp_stage_sync_fn():
            @parameter
            if not can_run_in_warp:
                barrier()

        comptime stage_sync_fn = block_or_warp_stage_sync_fn if (
            can_run_in_block or can_run_in_warp
        ) else cluster_stage_sync_fn
        comptime address_space = AddressSpace.SHARED if (
            can_run_in_block or can_run_in_warp
        ) else AddressSpace.SHARED_CLUSTER
        comptime max_shared_mem_size = shared_mem_per_warp if (
            can_run_in_warp
        ) else (
            shared_mem_per_block if can_run_in_block else shared_mem_per_cluster
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
            max_dim=max_dim,
            max_dim_base=max_dim_base,
            max_shared_mem_size=max_shared_mem_size,
            total_threads = block_threads * num_blocks,
            stage_sync_fn=stage_sync_fn,
            shared_address_space=address_space,
        ]
        comptime grid_dim = (Int(num_blocks), batch_size)

        @parameter
        if can_run_in_warp or can_run_in_block or can_run_in_block_cluster:
            __comptime_assert (
                full_output_size <= max_shared_mem_size
                or out_size_max_dim <= max_shared_mem_size
            ), (
                "internal implementation error for the given shape, please file"
                " an issue"
            )
            comptime run_cluster = can_run_in_block_cluster and num_blocks > 1
            comptime shared_mem = full_output_size if (
                full_output_size <= max_shared_mem_size
            ) else out_size_max_dim
            # constrained[
            #     False,
            #     String(
            #         "grid_dim: ",
            #         grid_dim[0],
            #         ", ",
            #         grid_dim[1],
            #         " block_threads: ",
            #         block_threads,
            #     ),
            # ]()
            ctx.enqueue_function_checked[block_func_batch, block_func_batch](
                out_batch,
                x_batch,
                grid_dim=grid_dim,
                cluster_dim=OptionalReg[Dim](
                    num_blocks
                ) if run_cluster else None,
                shared_mem_bytes=OptionalReg[Int](
                    Int(shared_mem)
                ) if run_cluster else None,
                block_dim=block_threads,
            )
        else:
            _fft_gpu_device_wide[
                max_cluster_size=max_cluster_size,
                inverse=inverse,
                bases=bases,
                grid_dim=grid_dim,
                block_threads=block_threads,
            ](output, x, ctx=ctx)

    comptime full_batches = batches // batch_size

    for i in range(full_batches):
        _launch_fn[Int(batch_size)](Int(i * batch_size))

    comptime remainder = batches % batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder)](Int(full_batches * batch_size))
