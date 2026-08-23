from max.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu
from layout import TileTensor, TensorLayout, row_major
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


@always_inline
def _check_layout_conditions[
    in_layout_type: TensorLayout, out_layout_type: TensorLayout
]():
    comptime rank = out_layout_type.rank
    comptime assert rank > 2, (
        "The rank should be bigger than 2. The first"
        " dimension represents the amount of batches, and the last the complex"
        " dimension."
    )
    comptime assert (
        in_layout_type.rank == rank
    ), "in_layout and out_layout must have equal rank"
    comptime assert (
        1 <= in_layout_type.static_shape[rank - 1] <= 2
    ), "The last dimension of in_layout should be 1 or 2"
    comptime assert (
        out_layout_type.static_shape[rank - 1] == 2
    ), "out_layout must have the last dimension equal to 2"
    comptime for i in range(rank - 2):
        comptime assert (
            in_layout_type.static_shape[i] == out_layout_type.static_shape[i]
        ), (
            "out_layout and in_layout should have the same shape before"
            " the last dimension"
        )
    comptime for i in range(rank - 2):
        comptime assert (
            out_layout_type.static_shape[i + 1] != 1
        ), "no inner dimension should be of size 1"


def _gcd_u(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x if x > 0 else 1


def _stockham_stage_gcd(length: Int, bases: List[UInt]) -> Int:
    """GCD of N/R over stages — sets 1-tpt vs multi-ept butterfly width."""
    if len(bases) == 0:
        return length
    var g = length // Int(bases[0])
    for i in range(1, len(bases)):
        g = _gcd_u(g, length // Int(bases[i]))
    return g


def _merge_leading_twos_to_fours(mut bases: List[UInt]):
    """Merge leading radix-2 pairs into 4s; odd count keeps one leading 2.

    Yields e.g. 640→`[2,4,4,4,5]`, 480→`[2,4,4,3,5]` (fewer stages, same
    product). Caller must gate on stage-GCD so multi-ept widths stay ≥ warp.
    """
    var n2 = 0
    while n2 < len(bases) and bases[n2] == 2:
        n2 += 1
    if n2 < 2:
        return
    var keep = n2 % 2
    var n4 = (n2 - keep) // 2
    var out = List[UInt](capacity=keep + n4 + (len(bases) - n2))
    if keep == 1:
        out.append(2)
    for _ in range(n4):
        out.append(4)
    for i in range(n2, len(bases)):
        out.append(bases[i])
    bases = out^


def _maybe_merge_leading_twos[warp_size: Int](
    length: Int, mut bases: List[UInt]
):
    """Apply `_merge_leading_twos_to_fours` when it does not kill multi-ept."""
    var g0 = _stockham_stage_gcd(length, bases)
    var merged = List[UInt](capacity=len(bases))
    for i in range(len(bases)):
        merged.append(bases[i])
    _merge_leading_twos_to_fours(merged)
    var g1 = _stockham_stage_gcd(length, merged)
    # Keep merge if still multi-ept (≥ warp) or already 1-tpt before merge.
    if g1 >= Int(warp_size) or g0 < Int(warp_size):
        bases = merged^


def _estimate_best_bases[
    out_layout_type: TensorLayout,
    target: StaticString,
    *,
    warp_size: Int = 32,
    max_registers_per_block: Int = 65536,
    max_thread_block_size: Int = 1024,
    threads_per_multiprocessor: Int = 2048,
](out bases: List[UInt]):
    comptime assert out_layout_type.rank > 1, "output rank must be > 1"
    comptime length = out_layout_type.static_shape[1]
    comptime max_radix_number = 32

    # GPU: fewer stages beat tiny radices when a 1-thread-per-sample FFT has
    # at least four warps. Radix cap from registers at the occupancy-sized
    # block (`threads_per_sm / 8`), not full max_tpb — that budget is what
    # intra-block Stockham actually launches. Also cap so `N/R >= warp` and
    # the gcd block stays a full warp. Mixed non-pow2 stays 2-first
    # ([15,32] / large-first lost ~23–29ms on 480).
    comptime occupancy_block = min(
        Int(max_thread_block_size),
        max(Int(warp_size), Int(threads_per_multiprocessor) // 8),
    )
    comptime regs_per_thread = Int(max_registers_per_block) // max(
        occupancy_block, 1
    )
    comptime radix_from_regs = max(2, regs_per_thread // 16)
    comptime radix_from_warp = max(2, Int(length) // max(Int(warp_size), 1))
    comptime gpu_max_radix = min(
        max_radix_number, radix_from_regs, radix_from_warp
    )
    comptime prefer_fewer_stages = (length & (length - 1)) == 0 and (
        length >= 4 * warp_size
    )

    comptime if (
        not is_cpu[target]()
        and length // max_radix_number <= Int(max_thread_block_size)
    ):
        # 640 for rect SBRC half-LDS: `[8,8,10]`@80 → n_work≤1.
        comptime if length == 640:
            var b640 = List[UInt](capacity=3)
            b640.append(UInt(8))
            b640.append(UInt(8))
            b640.append(UInt(10))
            return b640^
        # 480: `[8,6,10]` + partial reg-bfly + tpt=80 + n_work==1 path.
        comptime if length == 480:
            var b480 = List[UInt](capacity=3)
            b480.append(UInt(8))
            b480.append(UInt(6))
            b480.append(UInt(10))
            return b480^

        var potential_bases = List[UInt](capacity=16)
        var processed = 1
        comptime first_radix = (
            gpu_max_radix if prefer_fewer_stages else 2
        )
        var r = first_radix
        while r >= 2:
            var amnt_divisible = _times_divisible_by(
                UInt(length // processed), UInt(r)
            )
            potential_bases.reserve(Int(amnt_divisible))
            for _ in range(amnt_divisible):
                potential_bases.append(UInt(r))
                processed *= Int(r)
            if processed == length:
                # prefer_fewer peels large→small then reverses → large last.
                # Mixed peels 2s first; keep that order (stage-0 stays unit-stride).
                comptime if prefer_fewer_stages:
                    potential_bases.reverse()
                else:
                    _maybe_merge_leading_twos[warp_size](
                        length, potential_bases
                    )
                return potential_bases^
            r -= 1

        comptime rest_lo = 5 if prefer_fewer_stages else 3
        for r_rest in range(rest_lo, max_radix_number + 1):
            var amnt_divisible = _times_divisible_by(
                UInt(length // processed), UInt(r_rest)
            )
            potential_bases.reserve(Int(amnt_divisible))
            for _ in range(amnt_divisible):
                potential_bases.append(UInt(r_rest))
                processed *= Int(r_rest)
            if processed == length:
                comptime if prefer_fewer_stages:
                    potential_bases.reverse()
                else:
                    _maybe_merge_leading_twos[warp_size](
                        length, potential_bases
                    )
                return potential_bases^

    # fmt: off
    var lower_primes: Array[Byte, 25] = [
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
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    target: StaticString,
    *,
    warp_size: Int = 32,
    max_registers_per_block: Int = 65536,
    max_thread_block_size: Int = 1024,
    threads_per_multiprocessor: Int = 2048,
](out bases: List[List[UInt]]):
    _check_layout_conditions[in_layout_type, out_layout_type]()
    comptime amnt_dims = out_layout_type.rank - 2
    bases = {capacity = amnt_dims}

    comptime for i in range(amnt_dims):
        comptime dim = out_layout_type.static_shape[i + 1]
        comptime radix_layout_type = type_of(row_major[1, dim, 2]())
        bases.append(
            _estimate_best_bases[
                radix_layout_type,
                target,
                warp_size=warp_size,
                max_registers_per_block=max_registers_per_block,
                max_thread_block_size=max_thread_block_size,
                threads_per_multiprocessor=threads_per_multiprocessor,
            ]()
        )


@always_inline
def plan_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    *,
    inverse: Bool = False,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout_type, out_layout_type, "cpu"
    ](),
](*, cpu_workers: Optional[UInt] = None) -> _CPUPlan[
    out_dtype, out_layout_type, inverse, bases
]:
    """Plan the Fast Fourier Transform on CPU.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout_type: The `TensorLayout` of the input.
        out_layout_type: The `TensorLayout` of the output.
        inverse: Whether to run the inverse fourier transform.
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
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
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    *,
    warp_size: Int = 32,
    max_registers_per_block: Int = 65536,
    max_thread_block_size: Int = 1024,
    threads_per_multiprocessor: Int = 2048,
    bases: List[List[UInt]] = _estimate_best_bases_nd[
        in_layout_type,
        out_layout_type,
        "gpu",
        warp_size=warp_size,
        max_registers_per_block=max_registers_per_block,
        max_thread_block_size=max_thread_block_size,
        threads_per_multiprocessor=threads_per_multiprocessor,
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
    out_layout_type,
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
        in_layout_type: The `TensorLayout` of the input.
        out_layout_type: The `TensorLayout` of the output.
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


@always_inline
def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    inverse: Bool,
    bases: List[List[UInt]],
    //,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin, ...],
    x: TileTensor[in_dtype, in_layout_type, in_origin, ...],
    *,
    plan: _CPUPlan[out_dtype, out_layout_type, inverse, bases],
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the Fast Fourier Transform on CPU.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout_type: The `TensorLayout` of the input.
        out_layout_type: The `TensorLayout` of the output.
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
    _check_layout_conditions[in_layout_type, out_layout_type]()
    comptime assert len(bases) == out_layout_type.rank - 2, (
        "The bases list should have the same outer size as the amount of"
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->"
        " len(bases) == 3"
    )
    _run_cpu_nd_fft(output, x, plan=plan, cpu_workers=cpu_workers)


@always_inline
def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout_type: TensorLayout,
    out_layout_type: TensorLayout,
    in_origin: ImmOrigin,
    out_origin: MutOrigin,
    inverse: Bool,
    bases: List[List[UInt]],
    runtime_twfs: Bool,
    max_cluster_size: UInt,
    //,
](
    output: TileTensor[out_dtype, out_layout_type, out_origin, ...],
    x: TileTensor[in_dtype, in_layout_type, in_origin, ...],
    ctx: DeviceContext,
    *,
    plan: _GPUPlan[
        out_dtype,
        out_layout_type,
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
        in_layout_type: The `TensorLayout` of the input.
        out_layout_type: The `TensorLayout` of the output.
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
    _check_layout_conditions[in_layout_type, out_layout_type]()
    comptime assert len(bases) == out_layout_type.rank - 2, (
        "The bases list should have the same outer size as the amount of"
        " internal dimensions. e.g. (batches, dim_0, dim_1, dim_2, 2) ->"
        " len(bases) == 3"
    )
    _run_gpu_nd_fft(output, x, ctx, plan=plan)
