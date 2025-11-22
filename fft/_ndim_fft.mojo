from algorithm import parallelize, vectorize
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext
from gpu.host.info import Vendor, is_cpu
from layout import Layout, LayoutTensor, IntTuple
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
)
from ._fft import _radix_n_fft_kernel
from .fft import fft, _1d_fft_gpu


fn _run_intra_something_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    x_out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    x_out_origin: MutOrigin,
    out_address_space: AddressSpace,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    total_threads: UInt,  # not required for CPU
    stage_sync_fn: fn (),  # not required for CPU
    target: StaticString,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space
    ],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    x_out: LayoutTensor[
        out_dtype, x_out_layout, x_out_origin
    ],  # not required for CPU
    *,
    local_i: UInt,  # not required for CPU
    cpu_workers: Optional[UInt] = None,
):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = dims.product_flatten().value()
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    @parameter
    fn _run_1d_fft[
        out_dtype: DType,
        dtype_in: DType,
        layout_out: Layout,
        layout_in: Layout,
        layout_x_out: Layout,
        x_in_origin: ImmutOrigin,
        x_in_address_space: AddressSpace, //,
        dim_idx: Int,
    ](
        shared_f: LayoutTensor[
            out_dtype, layout_out, out_origin, address_space=out_address_space
        ],
        x: LayoutTensor[
            dtype_in, layout_in, x_in_origin, address_space=x_in_address_space
        ],
        x_out: LayoutTensor[out_dtype, layout_x_out, x_out.origin],
    ):
        comptime length = UInt(x.layout.shape[0].value())
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[dim_idx], target
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
        if not is_cpu[target]():

            @parameter
            for b in range(len(ordered_bases)):
                comptime base = ordered_bases[b]
                comptime processed = processed_list[b]
                comptime amnt_threads = length // base
                comptime func = _radix_n_fft_kernel[
                    do_rfft = dim_idx == 0 and x_complex_in == 1,
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
            return

        @parameter
        fn _inner_cpu_kernel(global_i: Int):
            var block_num = UInt(global_i)
            comptime block_out_layout = Layout.row_major(Int(length), 2)
            var local_out = LayoutTensor[
                mut=True,
                out_dtype,
                block_out_layout,
                output.origin,
                address_space = shared_f.address_space,
            ](shared_f.ptr + shared_f.stride[0]() * Int(block_num))
            comptime x_layout = Layout.row_major(Int(length), x_complex_in)
            var local_x = LayoutTensor[
                mut=False,
                dtype_in,
                x_layout,
                x.origin,
                address_space = x.address_space,
            ](x.ptr + x.stride[0]() * Int(block_num))
            comptime max_base = Int(ordered_bases[0])
            var x_out_array = InlineArray[Scalar[out_dtype], max_base * 2](
                uninitialized=True
            )
            var x_out = LayoutTensor[
                mut=True, out_dtype, Layout.row_major(max_base, 2)
            ](x_out_array.unsafe_ptr())

            @parameter
            for b in range(len(ordered_bases)):
                comptime base = ordered_bases[b]
                comptime processed = processed_list[b]
                comptime func = _radix_n_fft_kernel[
                    do_rfft = dim_idx == 0 and x_complex_in == 1,
                    base=base,
                    length=length,
                    processed=processed,
                    inverse=inverse,
                    twf_offset = twf_offsets[b],
                    ordered_bases=ordered_bases,
                ]

                @parameter
                fn _run[width: Int](local_i: Int):
                    func(local_out, local_x, UInt(local_i), twfs, x_out)

                comptime width = UInt(simd_width_of[out_dtype]())
                comptime unroll_factor = Int(base) if base <= width else 1
                vectorize[
                    _run, 1, size = Int(base), unroll_factor=unroll_factor
                ]()

        parallelize[func=_inner_cpu_kernel](
            prod // dims[dim_idx].value(),
            Int(cpu_workers.or_else(UInt(parallelism_level()))),
        )

    @parameter
    for idx, dim_tuple in enumerate(dims):
        comptime dim = dim_tuple.value()

        @parameter
        if dim == 1:
            continue

        comptime batch_prod = prod // dim

        @parameter
        for j in range(batch_prod):
            var reshaped_out = output.reshape[
                Layout.row_major(batch_prod, dim, 2)
            ]()
            comptime dim_batch_out_layout = Layout.row_major(dim, 2)
            var dim_batch_out = LayoutTensor[
                out_dtype,
                dim_batch_out_layout,
                output.origin,
                address_space = output.address_space,
            ](reshaped_out.ptr + reshaped_out.stride[0]() * j)
            var reshaped_x_out = x_out.reshape[
                Layout.row_major(batch_prod, dim, 2)
            ]()
            var dim_batch_x_out = LayoutTensor[
                out_dtype,
                dim_batch_out_layout,
                x_out.origin,
                address_space = x_out.address_space,
            ](reshaped_x_out.ptr + reshaped_x_out.stride[0]() * j)

            @parameter
            if idx != 0:
                _run_1d_fft[idx](
                    dim_batch_out,
                    dim_batch_out.get_immutable(),
                    dim_batch_x_out,
                )
            else:
                var reshaped_x = x.reshape[
                    Layout.row_major(batch_prod, dim, x_complex_in)
                ]()
                comptime x_layout = Layout.row_major(dim, x_complex_in)
                var dim_batch_x = LayoutTensor[in_dtype, x_layout, x.origin](
                    reshaped_x.ptr + reshaped_x.stride[0]() * j
                )
                _run_1d_fft[0](dim_batch_out, dim_batch_x, dim_batch_x_out)


# NOTE: We aren't really using this because of difficulty properly scheduling
# block clusters in a generic and portable manner. We need some better GPU info.
fn _intra_multiprocessor_fft_kernel_radix_n_multi_dim[
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
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `biggest_dimension // smallest_base <=
    max_threads_per_multiprocessor` and that `x_dim * y_dim [* z_dim]` out_dtype
    items fit in a block's shared memory."""

    var local_i = thread_idx.x
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
    comptime x_out_layout = Layout.row_major(Int(max_base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    fn stage_sync_fn():
        cluster_arrive_relaxed()
        cluster_wait()

    # TODO: this should use distributed shared memory for the intermediate output
    _run_intra_something_nd_fft[
        inverse=inverse,
        bases=bases,
        total_threads=total_threads,
        stage_sync_fn=stage_sync_fn,
        target="gpu",
    ](output, x, x_out, local_i=local_i)

    stage_sync_fn()


fn _intra_block_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    warp_exec: Bool,
    bases: List[List[UInt]],
    max_base: UInt,
    last_base: UInt,
    total_threads: UInt,
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `biggest_dimension // smallest_base <=
    max_threads_per_block` and that `x_dim * y_dim [* z_dim]` out_dtype items
    fit in a block's shared memory."""

    var local_i = thread_idx.x
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
        out_dtype, block_out_layout, MutOrigin.external
    ].stack_allocation()
    comptime x_out_layout = Layout.row_major(Int(max_base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    fn stage_sync_fn():
        @parameter
        if not warp_exec:
            barrier()

    _run_intra_something_nd_fft[
        inverse=inverse,
        bases=bases,
        total_threads=total_threads,
        stage_sync_fn=stage_sync_fn,
        target="gpu",
    ](shared_f, x, x_out, local_i=local_i)

    @parameter
    for i in range(last_base):
        comptime offset = i * total_threads
        var res = shared_f.load[width=2](Int(local_i + offset), 0)
        output.store(Int(local_i + offset), 0, res)

    stage_sync_fn()


fn _run_cpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    *,
    cpu_workers: Optional[UInt] = None,
):
    fn _no_fn():
        ...

    var x_out = LayoutTensor[out_dtype, Layout(0), MutOrigin.external](
        UnsafePointer[Scalar[out_dtype], MutOrigin.external]()
    )
    _run_intra_something_nd_fft[
        inverse=inverse,
        bases=bases,
        total_threads=0,
        stage_sync_fn=_no_fn,
        target="cpu",
    ](output, x, x_out, local_i=0, cpu_workers=cpu_workers)


fn _generic_gpu_multi_dim_fallback[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    max_cluster_size: UInt,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    *,
    ctx: DeviceContext,
) raises:
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1:rank]
    comptime dims_with_batch = out_layout.shape[:rank]
    comptime prod = dims_with_batch.product_flatten().value()

    @parameter
    for i in range(len(bases)):
        comptime dim = dims[i].value()
        comptime out_view_layout = Layout.row_major(prod // dim, dim, 2)
        var reshaped_output = output.reshape[out_view_layout]()
        var out_view = LayoutTensor[
            output.dtype, out_view_layout, output.origin
        ](reshaped_output.ptr)

        @parameter
        if i != 0:
            _1d_fft_gpu[
                inverse=inverse,
                max_cluster_size=max_cluster_size,
                bases = bases[i],
            ](out_view, out_view.get_immutable(), ctx)
        else:
            comptime x_compl = x.layout.shape[rank - 1].value()
            comptime x_view_layout = Layout.row_major(prod // dim, dim, x_compl)
            var reshaped_input = x.reshape[x_view_layout]()
            var x_view = LayoutTensor[x.dtype, x_view_layout, x.origin](
                reshaped_input.ptr
            )

            _1d_fft_gpu[
                inverse=inverse,
                max_cluster_size=max_cluster_size,
                bases = bases[i],
            ](out_view, x_view.get_immutable(), ctx)


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
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()
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

    @parameter
    if not run_in_block:
        # NOTE: Here is where we would try to schedule the code to run in a
        # block cluster / intra-multiprocessor if we had some better generics
        return _generic_gpu_multi_dim_fallback[
            max_cluster_size=max_cluster_size, inverse=inverse, bases=bases
        ](output, x, ctx=ctx)

    comptime batch_size = max_threads_available // num_threads

    @parameter
    fn _launch_fn[batch_size: Int, offset: Int]() raises:
        comptime out_tuple = IntTuple(batch_size, dims, 2)
        comptime out_batch_layout = Layout.row_major(out_tuple.flatten())
        var out_batch = LayoutTensor[
            out_dtype, out_batch_layout, output.origin
        ](output.ptr + output.stride[0]() * offset)
        comptime x_tuple = IntTuple(batch_size, dims, in_complex)
        comptime x_batch_layout = Layout.row_major(x_tuple.flatten())
        var x_batch = LayoutTensor[in_dtype, x_batch_layout, x.origin](
            x.ptr + x.stride[0]() * offset
        )
        comptime block_func_batch = _intra_block_fft_kernel_radix_n_multi_dim[
            in_dtype,
            out_dtype,
            out_batch_layout,
            x_batch_layout,
            x.origin,
            output.origin,
            inverse=inverse,
            bases=bases,
            warp_exec = UInt(gpu_info.warp_size)
            >= UInt(batch_size) * num_threads,
            max_base = _max(bases),
            last_base=last_base,
            total_threads=num_threads,
        ]

        ctx.enqueue_function_checked[block_func_batch, block_func_batch](
            out_batch,
            x_batch,
            grid_dim=(1, batch_size),
            block_dim=Int(num_threads),
        )

    @parameter
    for i in range(batches // batch_size):
        _launch_fn[Int(batch_size), Int(i * batch_size)]()

    comptime remainder = batches % batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder), Int((batches - remainder) * batch_size)]()
