from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of


from ._utils import _build_ordered_bases, _reduce_mul
from ._ndim_fft import _run_cpu_nd_fft, _run_gpu_nd_fft

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

    comptime lower_primes = InlineArray[UInt, 11](
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
    comptime prod = out_layout.shape[: rank - 1].product_flatten().value()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime amnt_dims = len(dims)
    bases = {capacity = amnt_dims}

    @parameter
    for i in range(amnt_dims):
        comptime dim = dims[i].value()
        bases.append(_estimate_best_bases[{{prod // dim, dim, 2}}, target]())


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
        "   asdads: ",
    ]()

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
