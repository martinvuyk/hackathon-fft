from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of


from ._utils import (
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _build_ordered_bases,
    _reduce_mul,
)
from ._fft import (
    _cpu_fft_kernel_radix_n,
    _intra_block_fft_kernel_radix_n,
    _launch_inter_or_intra_multiprocessor_fft,
)
from ._ndim_fft import (
    _intra_block_fft_kernel_radix_n_multi_dim,
    _run_cpu_nd_fft,
    _run_gpu_nd_fft,
)

comptime _DEFAULT_DEVICE = "cpu" if not has_accelerator() else "gpu"


fn _check_layout_conditions_nd[in_layout: Layout, out_layout: Layout]():
    comptime rank = out_layout.rank()
    constrained[
        rank > 2,
        "The rank should be bigger than 2. The first",
        " dimension represents the amount of batches, and the last the complex",
        " dimension.",
    ]()
    constrained[
        in_layout.rank() == rank,
        "in_layout and out_layout must have equal rank",
    ]()
    constrained[
        1 <= in_layout.shape[rank - 1].value() <= 2,
        "The last dimension of in_layout should be 1 or 2",
    ]()
    constrained[
        out_layout.shape[rank - 1].value() == 2,
        "out_layout must have the last dimension equal to 2",
    ]()
    constrained[
        out_layout.shape[: rank - 2] == in_layout.shape[: rank - 2],
        "out_layout and in_layout should have the same shape before",
        " the last dimension",
    ]()


fn _estimate_best_bases[
    out_layout: Layout, target: StaticString
](out bases: List[UInt]):
    constrained[out_layout.rank() > 1, "output rank must be > 1"]()
    comptime length = out_layout.shape[1].value()
    comptime max_radix_number = 32

    # NOTE: The smaller the base the better, but estimate the best ranges such
    # that the thread number preferably fits in a block for GPU.
    comptime common_thread_block_size = 1024
    comptime min_radix_for_block = Int(ceil(length / common_thread_block_size))

    @parameter
    if (
        not is_cpu[target]()
        and length // max_radix_number <= common_thread_block_size
    ):
        var radixes = range(min_radix_for_block, max_radix_number)
        # TODO: replace this with inline for generators once they properly
        # preallocate the sequence length (`[i for i in range(...)]`)
        var potential_bases = List[UInt](capacity=len(radixes))
        for r in radixes:
            if length % r == 0:
                potential_bases.append(UInt(r))

        bases = _build_ordered_bases[UInt(length)](potential_bases^)
        if _reduce_mul(bases) == UInt(length):
            return

    comptime lower_primes = InlineArray[UInt, 12](
        31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2
    )
    comptime amnt_primes = len(lower_primes)
    bases = {capacity = amnt_primes}

    for i in range(amnt_primes):
        var prime = lower_primes[i]
        if UInt(length) % prime == 0:
            bases.append(UInt(prime))


fn _estimate_best_bases_nd[
    in_layout: Layout, out_layout: Layout, target: StaticString
](out bases: List[List[UInt]]):
    _check_layout_conditions_nd[in_layout, out_layout]()
    comptime rank = out_layout.rank()
    comptime prod = out_layout.shape[: rank - 2].product_flatten().value()
    comptime dims = out_layout.shape[1 : rank - 2]
    comptime amnt_dims = len(dims)
    bases = {capacity = amnt_dims}

    @parameter
    for i in range(amnt_dims):
        comptime dim = dims[i].value()
        bases.append(_estimate_best_bases[{{prod // dim, dim, 2}}, target]())


fn _1d_fft_cpu[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[UInt],
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    *,
    cpu_workers: Optional[UInt],
) raises:
    comptime batches = UInt(in_layout.shape[0].value())
    comptime sequence_length = UInt(in_layout.shape[1].value())
    comptime do_rfft = in_layout.shape[2].value() == 1
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    comptime bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((sequence_length // last_base) * (last_base - 1))
        var offsets = List[UInt](capacity=c * len(bases))
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]

    _cpu_fft_kernel_radix_n[
        length=sequence_length,
        ordered_bases=ordered_bases,
        processed_list=processed_list,
        do_rfft=do_rfft,
        inverse=inverse,
        total_twfs=total_twfs,
        twf_offsets=twf_offsets,
    ](output, x, cpu_workers=cpu_workers)


fn _1d_fft_gpu[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[UInt],
    max_cluster_size: UInt,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    ctx: DeviceContext,
) raises:
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    comptime batches = UInt(in_layout.shape[0].value())
    comptime sequence_length = UInt(in_layout.shape[1].value())
    comptime in_complex = in_layout.shape[2].value()
    comptime do_rfft = in_complex == 1
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    comptime bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((sequence_length // last_base) * (last_base - 1))
        var offsets = List[UInt](capacity=c * len(bases))
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_m = gpu_info.threads_per_multiprocessor
    constrained[
        threads_per_m > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    comptime max_threads_available = UInt(threads_per_m * gpu_info.sm_count)
    comptime num_threads = sequence_length // ordered_bases[
        len(ordered_bases) - 1
    ]

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
    comptime output_size = UInt(size_of[out_dtype]()) * sequence_length * 2
    comptime run_in_block = num_threads <= max_threads_per_block and (
        output_size
    ) <= shared_mem_per_block
    comptime batch_size = max_threads_available // num_threads
    comptime block_dim_inter_multiprocessor = UInt(
        ceil(num_threads / num_blocks).cast[DType.uint]()
    )

    @parameter
    fn _launch_fn[batch_size: Int, offset: Int]() raises:
        comptime out_batch_layout = Layout.row_major(
            batch_size, Int(sequence_length), 2
        )
        var out_batch = LayoutTensor[
            out_dtype, out_batch_layout, output.origin
        ](output.ptr + output.stride[0]() * offset)
        comptime x_batch_layout = Layout.row_major(
            batch_size, Int(sequence_length), in_complex
        )
        var x_batch = LayoutTensor[in_dtype, x_batch_layout, x.origin](
            x.ptr + x.stride[0]() * offset
        )
        comptime block_func_batch = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            x_batch_layout,
            out_batch_layout,
            x.origin,
            output.origin,
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            warp_exec = UInt(gpu_info.warp_size) >= num_threads,
        ]

        @parameter
        if run_in_block:
            ctx.enqueue_function_checked[block_func_batch, block_func_batch](
                out_batch,
                x_batch,
                grid_dim=(1, batch_size),
                block_dim=Int(num_threads),
            )
        else:
            _launch_inter_or_intra_multiprocessor_fft[
                length=sequence_length,
                processed_list=processed_list,
                ordered_bases=ordered_bases,
                do_rfft=do_rfft,
                inverse=inverse,
                block_dim=block_dim_inter_multiprocessor,
                num_blocks=num_blocks,
                batches = UInt(batch_size),
                total_twfs=total_twfs,
                twf_offsets=twf_offsets,
                max_cluster_size=max_cluster_size,
            ](out_batch, x_batch, ctx)

    @parameter
    for i in range(batches // batch_size):
        _launch_fn[Int(batch_size), Int(i * batch_size)]()

    comptime remainder = batches % batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder), Int((batches - remainder) * batch_size)]()


fn fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool = False,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, "cpu"
    ](),
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        in_origin: The `Origin` of the input tensor.
        out_origin: The `Origin` of the output tensor.
        inverse: Whether to run the inverse fourier transform.
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`

    Notes:
        - This function automatically runs the rfft if the input is real-valued.
        - If the given bases list does not multiply together to equal the
        length, the builtin algorithm duplicates the biggest (CPU) values that
        can still divide the length until reaching it.
        - The amount of threads that will be launched is equal to the
        `sequence_length // smallest_base`.
    """
    _check_layout_conditions_nd[in_layout, out_layout]()
    constrained[
        len(bases) == out_layout.rank() - 2,
        "The bases list should have the same outer size as the amount of",
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->",
        " len(bases) == 3",
    ]()

    # Use a faster compilation path for 1D FFT
    @parameter
    if in_layout.rank() == 3:
        _1d_fft_cpu[inverse=inverse, bases = bases[0]](
            output, x, cpu_workers=cpu_workers
        )
    else:
        _run_cpu_nd_fft[inverse=inverse, bases=bases](
            output, x, cpu_workers=cpu_workers
        )


fn fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool = False,
    target: StaticString = _DEFAULT_DEVICE,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, target
    ](),
    # TODO: we'd need to know the cudaOccupancyMaxPotentialClusterSize for
    # every device to not use the portable 8
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters
    max_cluster_size: UInt = 8,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    ctx: DeviceContext,
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        in_origin: The `Origin` of the input tensor.
        out_origin: The `Origin` of the output tensor.
        inverse: Whether to run the inverse fourier transform.
        target: Target device ("cpu" or "gpu").
        bases: The list of bases for which to build the mixed-radix algorithm.
        max_cluster_size: In the case of NVIDIA GPUs, what the maximum cluster
            size for the device is.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`

    Notes:
        - This function automatically runs the rfft if the input is real-valued.
        - If the given bases list does not multiply together to equal the
        length, the builtin algorithm duplicates the biggest (CPU) / smallest
        (GPU) values that can still divide the length until reaching it.
        - For very long sequences on GPUs, it is worth considering bigger radix
        factors, due to the potential of being able to run the fft within a
        single block. Keep in mind that the amount of threads that will be
        launched is equal to the `sequence_length // smallest_base`.
    """
    _check_layout_conditions_nd[in_layout, out_layout]()
    constrained[
        len(bases) == out_layout.rank() - 2,
        "The bases list should have the same outer size as the amount of",
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->",
        " len(bases) == 3",
    ]()

    @parameter
    if is_cpu[target]():
        fft[inverse=inverse, bases=bases](output, x, cpu_workers=cpu_workers)
    elif in_layout.rank() == 3:  # Use a faster compilation path for 1D FFT
        _1d_fft_gpu[
            inverse=inverse, bases = bases[0], max_cluster_size=max_cluster_size
        ](output, x, ctx)
    else:
        _run_gpu_nd_fft[
            inverse=inverse, bases=bases, max_cluster_size=max_cluster_size
        ](output, x, ctx)


@always_inline
fn ifft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, "cpu"
    ](),
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the inverse Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`

    Notes:
        This function is provided as a wrapper for the `fft` function. The
        documentation is more complete there.
    """
    fft[bases=bases, inverse=True](output, x, cpu_workers=cpu_workers)


@always_inline
fn ifft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    target: StaticString = _DEFAULT_DEVICE,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, target
    ](),
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the inverse Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        target: Target device ("cpu" or "gpu").
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`

    Notes:
        This function is provided as a wrapper for the `fft` function. The
        documentation is more complete there.
    """
    fft[bases=bases, inverse=True, target=target](
        output, x, ctx, cpu_workers=cpu_workers
    )
