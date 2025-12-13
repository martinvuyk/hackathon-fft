from algorithm import parallelize, vectorize
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.info import Vendor, is_cpu
from layout import Layout, LayoutTensor, IntTuple
from utils.index import IndexList
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, size_of, simd_width_of
from math import ceil

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _product_of_dims,
    _get_cascade_idxes,
)
from ._fft import (
    _radix_n_fft_kernel,
    _launch_inter_or_intra_multiprocessor_fft,
    _intra_block_fft_kernel_radix_n,
    _inter_multiprocessor_fft_kernel_radix_n,
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
    comptime last_dim_idx = len(dims) - 1

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

            @parameter
            fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
                comptime last_base = ordered_bases[len(ordered_bases) - 1]
                var bases = materialize[ordered_bases]()
                var c = Int((length // last_base) * (last_base - 1))
                var offsets = List[UInt](capacity=c * len(bases))
                var val = UInt(0)
                for base in bases:
                    offsets.append(absolute_total)
                    val += (length // base) * (base - 1)
                return val, offsets^

            comptime total_offsets = _calc_total_offsets()
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
            twfs.unsafe_ptr().offset(absolute_offsets_idx_start),
            Int(dim_total_twfs_idx),
            owning=False,
        )
        ctx.enqueue_copy(view, twfs_array.unsafe_ptr())
    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
    )

    @parameter
    fn _run_1d_fft[
        batch_x_dtype: DType,
        batch_out_layout: Layout,
        batch_x_layout: Layout,
        batch_x_origin: ImmutOrigin, //,
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
            do_rfft = dim_idx == last_dim_idx and x_complex_in == 1,
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

    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    var inter_layer_buf: DeviceBuffer[out_dtype]
    comptime batch_t = LayoutTensor[
        out_dtype, Layout.row_major(output.layout.shape[1:]), address_space=_
    ]

    @parameter
    if len(dims) == 1:
        inter_layer_buf = {handle = {}, device_ptr = {}}
    else:
        inter_layer_buf = ctx.enqueue_create_buffer[out_dtype](
            batch_t.layout.size()
        )

    @parameter
    fn _run_batch(block_num: Int) raises:
        var block_offset = output.stride[0]() * block_num
        var base_out = batch_t(output.ptr + block_offset)
        var base_inter_out = batch_t(
            inter_layer_buf.unsafe_ptr() + block_offset
        )
        var base_x = LayoutTensor[
            in_dtype,
            Layout.row_major(x.layout.shape[1:]),
            address_space=_,
        ](x.ptr + x.stride[0]() * block_num)

        @parameter
        for idx in reversed(range(len(dims))):
            comptime dim_tuple = dims[idx]
            comptime dim = dim_tuple.value()
            comptime batch_prod = prod // dim
            __comptime_assert dim != 1, "no inner dimension should be of size 1"

            @parameter
            if idx != last_dim_idx:
                base_inter_out.copy_from(base_out)

            @parameter
            for inner_batch_n in range(batch_prod):
                comptime idxes = _get_cascade_idxes[
                    output.layout.shape[1:], (idx, rank - 2)
                ](inner_batch_n)
                var dim_batch_out = base_out.slice[
                    Slice(0, dim), Slice(0, 2), (idx, rank - 2)
                ](idxes)

                # We are running the ffts from right to left in the layout
                @parameter
                if idx == last_dim_idx:
                    var dim_batch_x = base_x.slice[
                        Slice(0, dim),
                        Slice(0, x_complex_in),
                        (idx, rank - 2),
                    ](idxes)
                    _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                else:
                    var dim_batch_inter_out = base_inter_out.slice[
                        Slice(0, dim), Slice(0, 2), (idx, rank - 2)
                    ](idxes).get_immutable()
                    _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)

    for i in range(batches):
        _run_batch(i)


fn _run_intra_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    x_out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    x_out_origin: MutOrigin,
    shared_f_origin: MutOrigin,
    out_address_space: AddressSpace,
    sharef_f_address_space: AddressSpace,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    total_threads: UInt,
    stage_sync_fn: fn (),
    last_base: UInt,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space
    ],
    shared_f: LayoutTensor[
        out_dtype,
        out_layout,
        shared_f_origin,
        address_space=sharef_f_address_space,
    ],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    x_out: LayoutTensor[out_dtype, x_out_layout, x_out_origin],
    local_i: UInt,
):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime last_dim_idx = len(dims) - 1

    comptime x_complex_in = in_layout.shape[rank - 1].value()

    @parameter
    fn _run_1d_fft[
        dtype_in: DType,
        layout_out: Layout,
        out_origin: MutOrigin,
        out_as: AddressSpace,
        layout_in: Layout,
        x_in_origin: ImmutOrigin,
        x_in_as: AddressSpace, //,
        dim_idx: Int,
    ](
        shared_f: LayoutTensor[
            out_dtype, layout_out, out_origin, address_space=out_as, **_
        ],
        x: LayoutTensor[
            dtype_in, layout_in, x_in_origin, address_space=x_in_as, **_
        ],
    ):
        comptime length = UInt(layout_in.shape[0].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx]
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]

        @parameter
        fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
            comptime last_base = ordered_bases[len(ordered_bases) - 1]
            var bases = materialize[ordered_bases]()
            var c = Int((length // last_base) * (last_base - 1))
            var offsets = List[UInt](capacity=c * len(bases))
            var val = UInt(0)
            for base in bases:
                offsets.append(val)
                val += (length // base) * (base - 1)
            return val, offsets^

        comptime total_offsets = _calc_total_offsets()
        comptime total_twfs = total_offsets[0]
        comptime twf_offsets = total_offsets[1]
        comptime twfs_array = _get_flat_twfs[
            out_dtype,
            length,
            total_twfs,
            ordered_bases,
            processed_list,
            inverse,
        ]()
        comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
        var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
            twfs_array.unsafe_ptr()
        )

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime amnt_threads = length // base
            comptime func = _radix_n_fft_kernel[
                do_rfft = dim_idx == last_dim_idx and x_complex_in == 1,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
                ordered_bases=ordered_bases,
            ]

            @parameter
            if amnt_threads == total_threads:
                func(shared_f, x, local_i, twfs, x_out)
            else:
                if local_i < amnt_threads:  # is execution thread
                    func(shared_f, x, local_i, twfs, x_out)

            stage_sync_fn()

    @parameter
    for idx in reversed(range(len(dims))):
        comptime dim_tuple = dims[idx]
        comptime dim = dim_tuple.value()
        comptime batch_prod = prod // dim
        __comptime_assert dim != 1, "no inner dimension should be of size 1"

        @parameter
        if idx != last_dim_idx:

            @parameter
            for i in range(last_base):
                comptime offset = i * total_threads
                var idx = Int(local_i + offset)
                output.store(idx, 0, shared_f.load[width=2](idx, 0))

            stage_sync_fn()

        @parameter
        for inner_batch_n in range(batch_prod):
            comptime idxes = _get_cascade_idxes[
                output.layout.shape, (idx, rank - 2)
            ](inner_batch_n)
            var dim_batch_out = shared_f.slice[
                Slice(0, dim), Slice(0, 2), (idx, rank - 2)
            ](idxes)

            # We are running the ffts from right to left in the layout
            @parameter
            if idx == last_dim_idx:
                var dim_batch_x = x.slice[
                    Slice(0, dim), Slice(0, x_complex_in), (idx, rank - 2)
                ](idxes)
                _run_1d_fft[idx](dim_batch_out, dim_batch_x)
            else:
                var dim_batch_inter_out = output.slice[
                    Slice(0, dim), Slice(0, 2), (idx, rank - 2)
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
    last_base: UInt,
    total_threads: UInt,
    shared_address_space: AddressSpace,
    stage_sync_fn: fn (),
    index_fn: fn () -> UInt,
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `biggest_dimension // smallest_base <=
    max_threads_per_multiprocessor` and that `x_dim * y_dim [* z_dim]` out_dtype
    items fit in a block's shared memory."""

    var local_i = index_fn()
    var block_num = block_dim.y * block_idx.y

    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = dims.product_flatten().value()

    comptime x_layout = Layout.row_major(in_layout.shape[1:rank])
    var x = LayoutTensor[in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(out_layout.shape[1:rank])
    var output = LayoutTensor[out_dtype, block_out_layout, batch_output.origin](
        batch_output.ptr + batch_output.stride[0]() * Int(block_num)
    )
    var shared_f = LayoutTensor[
        out_dtype,
        block_out_layout,
        MutOrigin.external,
        address_space=shared_address_space,
    ].stack_allocation()
    comptime x_out_layout = Layout.row_major(Int(max_base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    _run_intra_gpu_nd_fft[
        inverse=inverse,
        bases=bases,
        total_threads=total_threads,
        stage_sync_fn=stage_sync_fn,
        last_base=last_base,
    ](output, shared_f, x, x_out, local_i)

    @parameter
    for i in range(last_base):
        comptime offset = i * total_threads
        var idx = Int(local_i + offset)
        output.store(idx, 0, shared_f.load[width=2](idx, 0))

    stage_sync_fn()


fn _run_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_cluster_size: UInt,
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

    fn _find_max_threads(out threads_base: Tuple[UInt, UInt]):
        threads_base = {0, 0}

        @parameter
        for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(dims[i].value())
            if threads_base[0] < dim // val:
                threads_base = dim // val, val

    comptime threads_base = _find_max_threads()
    comptime num_threads = threads_base[0]
    comptime last_base = threads_base[1]

    comptime batches = UInt(in_layout.shape[0].value())
    comptime in_complex = in_layout.shape[rank - 1].value()

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_m = gpu_info.threads_per_multiprocessor
    constrained[
        threads_per_m > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    comptime max_threads_available = UInt(threads_per_m * gpu_info.sm_count)

    # TODO?: Implement for sequences > max_threads_available in the same GPU
    constrained[
        num_threads <= max_threads_available,
        "fft for sequences longer than max_threads_available",
        "is not implemented yet. max_threads_available: ",
        String(max_threads_available),
    ]()

    comptime num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_per_m = UInt(gpu_info.shared_memory_per_multiprocessor)
    comptime shared_mem_per_block = max_threads_per_block * (
        shared_mem_per_m // UInt(threads_per_m)
    )
    comptime flat_dims = dims.product_flatten().value()
    comptime output_size = UInt(size_of[out_dtype]() * flat_dims * 2)
    comptime run_in_block = num_threads <= max_threads_per_block and (
        output_size
    ) <= shared_mem_per_block

    comptime is_sm_90_or_newer = (
        gpu_info.vendor == Vendor.NVIDIA_GPU and gpu_info.compute >= 9.0
    )
    comptime run_in_block_cluster = num_blocks <= max_cluster_size and (
        output_size <= shared_mem_per_block
    ) and is_sm_90_or_newer

    comptime batch_size = max_threads_available // num_threads
    comptime block_threads = UInt(
        ceil(num_threads / num_blocks).cast[DType.uint]()
    )

    @parameter
    fn _launch_fn[batch_size: Int, offset: Int]() raises:
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

        fn index_fn() -> UInt:
            return block_dim.x * block_idx.x + thread_idx.x

        fn cluster_stage_sync_fn():
            cluster_arrive_relaxed()
            cluster_wait()

        fn block_stage_sync_fn():
            @parameter
            if UInt(gpu_info.warp_size) >= UInt(batch_size) * num_threads:
                barrier()

        comptime stage_sync_fn = block_stage_sync_fn if (
            run_in_block
        ) else cluster_stage_sync_fn
        comptime address_space = AddressSpace.SHARED if (
            run_in_block
        ) else AddressSpace.SHARED_CLUSTER

        comptime block_func_batch = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
            in_dtype,
            out_dtype,
            out_batch_layout,
            x_batch_layout,
            x.origin,
            output.origin,
            inverse=inverse,
            bases=bases,
            max_base = _max(bases),
            last_base=last_base,
            total_threads=num_threads,
            stage_sync_fn=stage_sync_fn,
            index_fn=index_fn,
            shared_address_space=address_space,
        ]
        comptime grid_dim = (Int(num_blocks), batch_size)

        @parameter
        if run_in_block or run_in_block_cluster:
            ctx.enqueue_function_checked[block_func_batch, block_func_batch](
                out_batch, x_batch, grid_dim=grid_dim, block_dim=block_threads
            )
        else:
            _fft_gpu_device_wide[
                max_cluster_size=max_cluster_size,
                inverse=inverse,
                bases=bases,
                grid_dim=grid_dim,
                block_threads=block_threads,
            ](output, x, ctx=ctx)

    @parameter
    for i in range(batches // batch_size):
        _launch_fn[Int(batch_size), Int(i * batch_size)]()

    comptime remainder = batches % batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder), Int((batches - remainder) * batch_size)]()
