from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from std.math import ceil, log2
from std.sys.info import has_accelerator, size_of, is_64bit
from std.bit import count_trailing_zeros

from ._utils import (
    _build_ordered_bases,
    _reduce_mul,
    _product_of_dims,
    _times_divisible_by,
)
from ._ndim_fft_cpu import _run_cpu_nd_fft, _CPUPlan
from ._ndim_fft_gpu import _run_gpu_nd_fft, _GPUPlan, _GPUTest

comptime _DEFAULT_DEVICE = "cpu" if not has_accelerator() else "gpu"


def _check_layout_conditions_nd[in_layout: Layout, out_layout: Layout]():
    comptime rank = out_layout.rank()
    comptime assert rank > 2, (
        "The rank should be bigger than 2. The first"
        " dimension represents the amount of batches, and the last the complex"
        " dimension."
    )
    comptime assert (
        in_layout.rank() == rank
    ), "in_layout and out_layout must have equal rank"
    comptime assert (
        1 <= in_layout.shape[rank - 1].value() <= 2
    ), "The last dimension of in_layout should be 1 or 2"
    comptime assert (
        out_layout.shape[rank - 1].value() == 2
    ), "out_layout must have the last dimension equal to 2"
    comptime assert (
        out_layout.shape[: rank - 2] == in_layout.shape[: rank - 2]
    ), (
        "out_layout and in_layout should have the same shape before"
        " the last dimension"
    )

    comptime for i in range(rank - 2):
        comptime assert (
            out_layout.shape[i + 1] != 1
        ), "no inner dimension should be of size 1"


def _estimate_best_bases[
    out_layout: Layout, target: StaticString
](out bases: List[UInt]):
    comptime assert out_layout.rank() > 1, "output rank must be > 1"
    comptime length = out_layout.shape[1].value()
    comptime max_radix_number = 32

    # NOTE: The smaller the base the better, but estimate the best ranges such
    # that the thread number preferably fits in a block for GPU.
    comptime common_thread_block_size = 1024
    comptime min_radix_for_block = Int(ceil(length / common_thread_block_size))

    comptime if (
        not is_cpu[target]()
        and length // max_radix_number <= common_thread_block_size
    ):
        var radixes = range(max(min_radix_for_block, 2), max_radix_number + 1)
        var potential_bases = List[UInt](capacity=len(radixes))
        var processed = 1
        for r in radixes:
            var amnt_divisible = _times_divisible_by(
                UInt(length // processed), UInt(r)
            )

            potential_bases.reserve(Int(amnt_divisible))
            for _ in range(amnt_divisible):
                potential_bases.append(UInt(r))
                processed *= r

            if processed == length:
                potential_bases.reverse()
                return potential_bases^

    # fmt: off
    var lower_primes: InlineArray[Byte, 25] = [
        97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19,
        17, 13, 11, 7, 5, 3, 2
    ]
    # fmt: on
    bases = {capacity = len(lower_primes)}
    var processed = 1

    for i in range(len(lower_primes)):
        var prime = UInt(lower_primes[i])
        var amnt_divisible = _times_divisible_by(
            UInt(length // processed), prime
        )

        bases.reserve(Int(amnt_divisible))
        for _ in range(amnt_divisible):
            bases.append(prime)
            processed *= Int(prime)

        if processed == length:
            bases.reverse()
            return


def _estimate_best_bases_nd[
    in_layout: Layout, out_layout: Layout, target: StaticString
](out bases: List[List[UInt]]):
    _check_layout_conditions_nd[in_layout, out_layout]()
    comptime dims = out_layout.shape[1 : out_layout.rank() - 1]
    comptime amnt_dims = len(dims)
    bases = {capacity = amnt_dims}

    comptime for i in range(amnt_dims):
        comptime dim = dims[i].value()
        bases.append(
            _estimate_best_bases[Layout.row_major(1, dim, 2), target]()
        )


@always_inline
def plan_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool = False,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, "cpu"
    ](),
](*, cpu_workers: Optional[UInt] = None) -> _CPUPlan[
    out_dtype, out_layout, inverse, bases
]:
    """Plan the Fast Fourier Transform on CPU.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        inverse: Whether to run the inverse fourier transform.
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        plan: The execution plan, it is best to build it outside this function
            if it is to be called repeatedly.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`.
    """
    return {}


@always_inline
def plan_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout, out_layout, "gpu"
    ](),
    inverse: Bool = False,
    runtime_twfs: Bool = True,
    # TODO: we'd need to know the cudaOccupancyMaxPotentialClusterSize for
    # every device to not use the portable 8
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters
    max_cluster_size: UInt = 8,
    _test: Optional[_GPUTest] = None,
](*, ctx: DeviceContext) raises -> _GPUPlan[
    out_dtype,
    out_layout,
    inverse,
    bases,
    _test,
    ctx.default_device_info,
    max_cluster_size=max_cluster_size,
    runtime_twfs=runtime_twfs,
]:
    """Plan the Fast Fourier Transform on GPU.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.
        inverse: Whether to run the inverse fourier transform.
        runtime_twfs: Whether to calculate the twiddle factors at runtime for
            big dimensions (faster, no global memory allocation) at the cost of
            lower precision.
        max_cluster_size: In the case of NVIDIA GPUs, what the maximum cluster
            size for the device is.
        _test: Internal use only.

    Args:
        ctx: The `DeviceContext`.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`.
    """
    return {ctx}


def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    inverse: Bool,
    bases: List[List[UInt]],
    //,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin, ...],
    x: LayoutTensor[in_dtype, in_layout, in_origin, ...],
    *,
    plan: _CPUPlan[out_dtype, out_layout, inverse, bases],
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the Fast Fourier Transform on CPU.

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
        plan: The execution plan, it is best to build it outside this function
            if it is to be called repeatedly.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`.
    """
    _check_layout_conditions_nd[in_layout, out_layout]()
    comptime assert len(bases) == out_layout.rank() - 2, (
        "The bases list should have the same outer size as the amount of"
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->"
        " len(bases) == 3"
    )
    _run_cpu_nd_fft(output, x, plan=plan, cpu_workers=cpu_workers)


def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    inverse: Bool,
    bases: List[List[UInt]],
    runtime_twfs: Bool,
    max_cluster_size: UInt,
    //,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    ctx: DeviceContext,
    *,
    plan: _GPUPlan[
        out_dtype,
        out_layout,
        inverse,
        bases,
        None,
        ctx.default_device_info,
        max_cluster_size=max_cluster_size,
        runtime_twfs=runtime_twfs,
    ],
) raises:
    """Calculate the Fast Fourier Transform on GPU.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        in_origin: The `Origin` of the input tensor.
        out_origin: The `Origin` of the output tensor.
        inverse: Whether to run the inverse fourier transform.
        bases: The list of bases for which to build the mixed-radix algorithm.
        runtime_twfs: Whether to calculate the twiddle factors at runtime (
            faster for big tensors) at the cost of lower precision.
        max_cluster_size: In the case of NVIDIA GPUs, what the maximum cluster
            size for the device is.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.
        plan: The execution plan, it is best to build it outside this function
            if it is to be called repeatedly.

    Constraints:
        The layout should match one of: `{(batches, dim_0 [, dim_1 [, ...]], 1),
        (batches, dim_0 [, dim_1 [, ...]], 2)}`.
    """
    _check_layout_conditions_nd[in_layout, out_layout]()
    comptime assert len(bases) == out_layout.rank() - 2, (
        "The bases list should have the same outer size as the amount of"
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->"
        " len(bases) == 3"
    )
    _run_gpu_nd_fft(output, x, ctx, plan)
