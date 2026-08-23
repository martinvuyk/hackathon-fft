from std.sys.info import is_64bit, is_nvidia_gpu, is_gpu
from std.complex import ComplexScalar, ComplexSIMD
from std.math import exp, pi, sin, cos, log2
from std.bit import count_trailing_zeros
from layout import CoordLike, IntTuple, RowMajorLayout, TensorLayout, row_major
from std.utils.index import IndexList

comptime EPSILON = 1e-9


def _get_dtype[length: UInt]() -> DType:
    comptime if length <= UInt(UInt8.MAX):
        return DType.uint8
    elif length <= UInt(UInt16.MAX):
        return DType.uint16
    elif length <= UInt(UInt32.MAX):
        return DType.uint32
    elif UInt64(length) <= UInt64.MAX:
        return DType.uint64
    elif UInt128(length) <= UInt128.MAX:
        return DType.uint128
    else:
        return DType.uint256


def _mixed_radix_digit_reverse[
    length: UInt, ordered_bases: List[UInt], reverse: Bool = False
](idx: Scalar) -> type_of(idx):
    """Performs mixed-radix digit reversal for an index `idx` based on a
    sequence of `ordered_bases`.

    Notes:
        Given `N = R_0 * R_1 * ... * R_{M-1}`, an input index `k` is represented
        as: `k = d_0 + d_1*R_0 + d_2*R_0*R_1 + ... + d_{M-1}*R_0*...*R_{M-2}`
        where d_i is the digit for radix R_i.

        The reversed index k' is:
        `k' = d_{M-1} + d_{M-2}*R_{M-1} + ... + d_1*R_{M-1}*...*R_2 + d_0*R_{M-1
        }*...*R_1`
    """
    var reversed_idx = type_of(idx)(0)
    var current_val = idx
    var base_offset: type_of(idx)

    comptime if reverse:
        base_offset = 1
    else:
        base_offset = {length}

    comptime for i in range(len(ordered_bases)):
        comptime base = type_of(idx)(
            ordered_bases[i if not reverse else (len(ordered_bases) - 1 - i)]
        )

        comptime if not reverse:
            base_offset //= base
        reversed_idx += (current_val % base) * base_offset
        current_val //= base

        comptime if reverse:
            base_offset *= base
    return reversed_idx


def _get_twiddle_factor[
    dtype: DType, *, inverse: Bool, N: Scalar
](n: Scalar) -> ComplexScalar[dtype]:
    """Returns `exp((-j * 2 * pi * n) / N)`."""
    comptime assert dtype.is_floating_point()
    comptime `-2π/N` = Scalar[dtype](-2.0 * pi) / Scalar[dtype](N)
    var theta = `-2π/N` * Scalar[dtype](n)

    var num: ComplexScalar[dtype]

    if __is_run_in_comptime_interpreter:
        var factor = 2 * n.cast[dtype]() / N.cast[dtype]()
        if factor < EPSILON:  # approx. 0
            num = {1, 0}
        elif abs(factor - 0.5) < EPSILON:
            num = {0, -1}
        elif abs(factor - 1) < EPSILON:
            num = {-1, 0}
        elif abs(factor - 1.5) < EPSILON:
            num = {0, 1}
        else:
            # FIXME: remove once comptime branch and comptime assert don't short circuit
            comptime if is_nvidia_gpu():
                num = {
                    cos(theta.cast[DType.float32]()).cast[dtype](),
                    sin(theta.cast[DType.float32]()).cast[dtype](),
                }
            else:
                num = {cos(theta), sin(theta)}
    else:
        comptime if is_nvidia_gpu():
            num = {
                cos(theta.cast[DType.float32]()).cast[dtype](),
                sin(theta.cast[DType.float32]()).cast[dtype](),
            }
        else:
            num = {cos(theta), sin(theta)}

    comptime if not inverse:
        return num
    else:
        return num.conj()


def _get_twiddle_factors[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: List[ComplexScalar[dtype]]):
    """Get all the twiddle factors for the length."""
    res = {unsafe_uninit_length = Int(length)}
    for n in range(length):
        res[n] = _get_twiddle_factor[dtype, inverse=inverse, N=length](n)


def _get_twiddle_factors_inline[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: Array[ComplexScalar[dtype], Int(length)]):
    """Get all the twiddle factors for the length."""
    res = {uninitialized = True}
    for n in range(length):
        res[n] = _get_twiddle_factor[dtype, inverse=inverse, N=length](n)


def _div_by(x: UInt, base: UInt) -> UInt:
    # TODO: benchmark whether this performs better than doing branches
    return 1 if base == x else (
        0 if (base > x or x % base != 0) else (_div_by(x // base, base) + 1)
    )


def _times_divisible_by(length: UInt, base: UInt, out amnt_divisible: UInt):
    debug_assert(base != 1, "The number 1 can infinitely divide any number")
    if base.is_power_of_two():
        amnt_divisible = UInt(
            count_trailing_zeros(Scalar[DType.uint](length))
            // log2(Float64(base)).cast[DType.uint]()
        )
    else:
        amnt_divisible = _div_by(length, base)


def _reduce_mul(b: List[UInt], out res: UInt):
    res = UInt(1)
    for base in b:
        res *= base


def _uint_gcd(a: UInt, b: UInt) -> UInt:
    var x = a
    var y = b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


@always_inline
def _reg_dft_radix[base: UInt]() -> Bool:
    """Register DFT: pow2, mixed `[3, 8]`, Bailey `6`/`9`/`10`/`15`/`20`."""
    return (
        base.is_power_of_two()
        or (base >= 3 and base <= 8)
        or base == 9
        or base == 10
        or base == 15
        or base == 20
    )


@always_inline
def _is_reg_dft_radix(base: UInt) -> Bool:
    """Runtime form of `_reg_dft_radix` (dynamic radix values)."""
    return (
        base.is_power_of_two()
        or (base >= 3 and base <= 8)
        or base == 9
        or base == 10
        or base == 15
        or base == 20
    )


@always_inline
def _stage_use_reg_bfly[base: UInt, length: UInt, tpt: UInt]() -> Bool:
    """Register Stockham butterfly when `tpt` exactly covers `n_bfly` groups."""
    comptime n_bfly = length // base
    return (
        _reg_dft_radix[base]()
        and length % base == 0
        and tpt > 0
        and n_bfly % tpt == 0
    )


@always_inline
def _stage_use_reg_bfly_partial[base: UInt, length: UInt, tpt: UInt]() -> Bool:
    """Like `_stage_use_reg_bfly`, but allow uneven cover via runtime ceildiv + guard.

    Used by the multi-ept intra path (e.g. 480@96) so R=2/3/4 stages can use
    closed butterflies without comptime-unrolling partial `n_work` into regs.
    """
    comptime n_bfly = length // base
    return (
        _reg_dft_radix[base]()
        and length % base == 0
        and tpt > 0
        and n_bfly > 0
    )


def _stockham_intra_block[
    dim: Int,
    bases: List[UInt],
    *,
    warp_size: UInt,
    occupancy_block: UInt,
]() -> UInt:
    """Intra-block thread count for Stockham.

    When every radix has a register DFT, pick the largest warp-multiple that
    divides every stage's group count `dim/R` and is at most `occupancy_block`.
    Otherwise keep the occupancy target when it divides `dim`. Returns 0 to
    mean one thread per sample.
    """
    # 480+[8,6,10]: tpt=80 (R=6 n_bfly=80 exact) — ~13.38 vs tpt=96 ~13.62.
    # Must precede GCD path (GCD of 60/80/48 is 4).
    if UInt(dim) == 480:
        return UInt(80)

    comptime ordered = _build_ordered_bases[UInt(dim), bases]()
    var ordered_var = materialize[ordered]()
    var all_reg = len(ordered_var) > 0
    for i in range(len(ordered_var)):
        if not _is_reg_dft_radix(ordered_var[i]):
            all_reg = False

    if all_reg:
        var g = UInt(dim) // ordered_var[0]
        for i in range(1, len(ordered_var)):
            g = _uint_gcd(g, UInt(dim) // ordered_var[i])
        if g >= warp_size:
            var block = min(occupancy_block, g)
            block = (block // warp_size) * warp_size
            while block >= warp_size and g % block != 0:
                block -= warp_size
            if (
                block >= warp_size
                and UInt(dim) > block
                and UInt(dim) % block == 0
            ):
                return block

    if (
        occupancy_block > 0
        and UInt(dim) > occupancy_block
        and UInt(dim) % occupancy_block == 0
    ):
        return occupancy_block
    return 0


def _build_ordered_bases[length: UInt, bases: List[UInt]]() -> List[UInt]:
    var existing_bases = materialize[bases]()
    # Preserve caller order when the product is already exact (GPU planning
    # picks stage order deliberately; sorting forced large-first).
    if _reduce_mul(existing_bases) == length:
        return existing_bases^
    sort(existing_bases)  # FIXME: this should just be ascending=False
    var new_bases = List[UInt](capacity=len(existing_bases))

    var processed = UInt(1)
    for i in reversed(range(len(existing_bases))):
        var base = existing_bases[i]
        var amnt_divisible = _times_divisible_by(length, base)
        new_bases.reserve(Int(amnt_divisible))
        for _ in range(amnt_divisible):
            new_bases.append(base)
            processed *= base

        if processed == length:
            break
    return new_bases^


def _get_ordered_bases_processed_list[
    length: UInt, bases: List[UInt]
]() -> Tuple[List[UInt], List[UInt]]:
    comptime assert len(bases) > 0, String(
        "The amount of bases is not enough: ", bases
    )
    comptime ordered_bases = _build_ordered_bases[length, bases]()

    def _build_processed_list() -> List[UInt]:
        var ordered_bases_var = materialize[ordered_bases]()
        var processed_list = List[UInt](capacity=len(ordered_bases_var))
        var processed = UInt(1)
        for base in ordered_bases_var:
            processed_list.append(processed)
            processed *= base
        return processed_list^

    comptime processed_list = _build_processed_list()
    comptime assert len(processed_list) == len(ordered_bases), "internal error"
    comptime assert (
        len(processed_list) > 0
        and processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length
    ), String(
        "powers of the bases must multiply together  to equal the sequence ",
        "length. The builtin algorithm was only able to produce: ",
        String(ordered_bases)
        .replace("SIMD[DType.uint, 1](", "")
        .replace(")", ""),
        " for the length: ",
        length,
    )
    comptime assert 1 not in ordered_bases, "Cannot do an fft with base 1."
    return materialize[ordered_bases](), materialize[processed_list]()


@always_inline
def _min(elems: List[UInt], out smallest: UInt):
    smallest = elems[0]
    for elem in elems[1:]:
        smallest = min(elem, smallest)


@always_inline
def _max(elems: IntTuple, out smallest: UInt):
    smallest = UInt(elems[0].value())
    for elem in elems[1:]:
        smallest = max(UInt(elem.value()), smallest)


@always_inline
def _max(elems: List[UInt], out smallest: UInt):
    smallest = elems[0]
    for elem in elems[1:]:
        smallest = max(elem, smallest)


@always_inline
def _max(elems: List[List[UInt]], out biggest: UInt):
    biggest = 0
    for bases in elems:
        for elem in bases:
            biggest = max(elem, biggest)


comptime _KeepSpatialDim[L: TensorLayout, T: CoordLike, idx: Int]: Bool = (
    idx >= 1 and idx < L.rank - 1
)

comptime _KeepBeforeLastDim[L: TensorLayout, T: CoordLike, idx: Int]: Bool = (
    idx < L.rank - 1
)

comptime _dims[L: TensorLayout] = RowMajorLayout[
    *L._shape_types.filter_idx[_KeepSpatialDim[L, _, _]]()
]

comptime _dims_from_tail[L: TensorLayout] = RowMajorLayout[
    *L._shape_types.filter_idx[_KeepBeforeLastDim[L, _, _]]()
]


@always_inline
def _product_of_dims[dims: TensorLayout]() -> Int:
    """Product of spatial axes in ``dims``."""
    var prod = 1
    comptime for i in range(dims.rank):
        prod *= dims.static_shape[i]
    return prod


@always_inline
def _product_of_dims_slice[
    dims: TensorLayout, start: Int, end: Int
]() -> Int:
    """Product of ``dims.static_shape[start:end]`` (empty → 1)."""
    var prod = 1
    comptime for i in range(start, end):
        prod *= dims.static_shape[i]
    return prod


@always_inline
def _axis_is_excluded[excluded: IntTuple, i: Int]() -> Bool:
    comptime for j in range(len(excluded)):
        comptime if i == excluded[j].value():
            return True
    return False


# NOTE: currently unused, but needed for future inplace variant
def _get_cascade_idxes[
    shape: IntTuple, excluded: IntTuple
](var flat_idx: Int, out idxes: IndexList[len(shape) - len(excluded)]):
    idxes = {fill = 0}

    def _idxes_i(i: Int, out amnt: Int):
        amnt = i

        comptime for j in range(len(excluded)):
            comptime val = excluded[j].value()
            amnt -= Int(i > val)

    comptime for i in range(len(shape)):
        comptime if _axis_is_excluded[excluded, i]():
            continue
        comptime curr_num = UInt(shape[i].value())
        comptime idxes_i = _idxes_i(i)
        idxes[idxes_i] = Int(UInt(flat_idx) % curr_num)
        flat_idx = Int(UInt(flat_idx) // curr_num)


@always_inline
def _unit_phasor_mul[twf: ComplexSIMD](val: type_of(twf)) -> type_of(twf):
    """Optimizes `twf * val`."""

    comptime if abs(twf.re - (1)) < EPSILON:  # Co(1, 0)
        return val
    elif abs(twf.im - (-1)) < EPSILON:  # Co(0, -1)
        return {val.im, -val.re}
    elif abs(twf.re - (-1)) < EPSILON:  # Co(-1, 0)
        return -val
    elif abs(twf.im - (1)) < EPSILON:  # Co(0, 1)
        return {-val.im, val.re}
    elif abs(abs(twf.re) - abs(twf.im)) < EPSILON:  # Co(1/√2, 1/√2)
        comptime factor = abs(twf.re)

        comptime if twf.re > 0 and twf.im > 0:  # Q1
            return {factor * (val.re - val.im), factor * (val.re + val.im)}
        elif twf.re < 0 and twf.im > 0:  # Q2
            return {factor * (-val.re - val.im), factor * (val.re - val.im)}
        elif twf.re < 0 and twf.im < 0:  # Q3
            return {factor * (-val.re + val.im), factor * (-val.re - val.im)}
        else:  # Q4
            return {factor * (val.re + val.im), factor * (-val.re + val.im)}
    else:
        return val * twf


@always_inline
def _unit_phasor_fma[
    twf: ComplexSIMD
](x_j: type_of(twf), acc: type_of(twf)) -> type_of(twf):
    comptime if abs(twf.re - (1)) < EPSILON:  # Co(1, 0)
        return acc + x_j
    elif abs(twf.im - (-1)) < EPSILON:  # Co(0, -1)
        return {acc.re + x_j.im, acc.im - x_j.re}
    elif abs(twf.re - (-1)) < EPSILON:  # Co(-1, 0)
        return acc - x_j
    elif abs(twf.im - (1)) < EPSILON:  # Co(0, 1)
        return {acc.re - x_j.im, acc.im + x_j.re}
    elif abs(abs(twf.re) - abs(twf.im)) < EPSILON:  # Co(1/√2, 1/√2)
        comptime factor = abs(twf.re)
        var re = x_j.re
        var im = x_j.im

        comptime if twf.re > 0 and twf.im > 0:  # Q1
            return acc + {factor * (re - im), factor * (re + im)}
        elif twf.re < 0 and twf.im > 0:  # Q2
            return acc + {factor * (-re - im), factor * (re - im)}
        elif twf.re < 0 and twf.im < 0:  # Q3
            return acc + {factor * (-re + im), factor * (-re - im)}
        else:  # Q4
            return acc + {factor * (re + im), factor * (-re + im)}
    else:
        return twf.fma(x_j, acc)


@always_inline
def _unit_phasor_fma[
    twf: ComplexSIMD, accum_is_real: Bool
](x_j: SIMD[twf.dtype, twf.length], acc: type_of(twf)) -> type_of(twf):
    comptime if abs(twf.re - 1) < EPSILON:  # Co(1, 0)
        return {acc.re + x_j, acc.im}
    elif abs(twf.im - (-1)) < EPSILON and accum_is_real:  # Co(0, -1)
        return {acc.re, -x_j}
    elif abs(twf.im - (-1)) < EPSILON:  # Co(0, -1)
        return {acc.re, acc.im - x_j}
    elif abs(twf.re - (-1)) < EPSILON:  # Co(-1, 0)
        return {acc.re - x_j, acc.im}
    elif abs(twf.im - (1)) < EPSILON and accum_is_real:  # Co(0, 1)
        return {acc.re, x_j}
    elif abs(twf.im - (1)) < EPSILON:  # Co(0, 1)
        return {acc.re, acc.im + x_j}
    elif accum_is_real:
        return {twf.re.fma(x_j, acc.re), twf.im * x_j}
    else:
        return {
            from_interleaved = twf.re.join(twf.im).fma(
                x_j.join(x_j), acc.re.join(acc.im)
            )
        }


@always_inline
def _false[a: Int]() -> Bool:
    return False


def _num_stages_end_of[
    bases: List[List[UInt]],
    dims: TensorLayout,
    dim_idx: Int,
    use_scratch_buffer: def[Int]() thin -> Bool = _false,
]() -> Int:
    comptime start_dim_idx = dims.rank - 1
    var num_stages = 0
    comptime for i in range(dim_idx, start_dim_idx + 1):
        comptime if use_scratch_buffer[i]():
            num_stages += 1
            continue
        comptime length = UInt(dims.static_shape[i])
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases[i]
        ]()
        comptime stages = len(bases_processed[0])
        num_stages += stages
    return num_stages


comptime _KeepTailDim[T: CoordLike, idx: Int]: Bool = idx >= 1

comptime _tail_tile_layout[L: TensorLayout] = RowMajorLayout[
    *L._shape_types.filter_idx[_KeepTailDim]()
]


def _calc_batches_M_N[
    dims: TensorLayout, into_: Int, from_: Int
]() -> Tuple[UInt, UInt, UInt]:
    """Transpose batch geometry from spatial ``dims`` layout."""
    comptime target_idx = min(into_, from_)
    comptime is_forward = into_ < from_

    var batch_val = UInt(1)
    comptime for i in range(target_idx):
        batch_val *= UInt(dims.static_shape[i])

    var m_val = UInt(dims.static_shape[target_idx])

    var n_val = UInt(1)
    comptime for i in range(target_idx + 1, dims.rank):
        n_val *= UInt(dims.static_shape[i])

    comptime if is_forward:
        return batch_val, m_val, n_val
    else:
        return batch_val, n_val, m_val


@always_inline
def _spatial_axis_scalar_stride[
    dims: TensorLayout, dim_idx: Int, *, complex_width: Int = 2
]() -> Int:
    """Scalars between consecutive samples along spatial axis `dim_idx`.

    Assumes row-major `[D0, …, Dn]` complexes stored with `complex_width`
    scalars each (2 for C2C).
    """
    var stride = complex_width
    comptime for j in range(dim_idx + 1, dims.rank):
        stride *= Int(dims.static_shape[j])
    return stride


@always_inline
def _spatial_complex_stride[dims: TensorLayout, dim_idx: Int]() -> Int:
    """Complex-index stride of spatial axis `dim_idx` (product of trailing dims)."""
    var stride = 1
    comptime for j in range(dim_idx + 1, dims.rank):
        stride *= Int(dims.static_shape[j])
    return stride


@always_inline
def _nd_line_base_scalar_offset[
    dims: TensorLayout, dim_idx: Int, *, complex_width: Int = 2
](batch_id: Int) -> Int:
    """Scalar offset to the first sample of FFT line `batch_id` along `dim_idx`.

    `batch_id` enumerates `outer_batches * (prod/dim)` lines in row-major order
    over all axes except `dim_idx` (including a leading outer batch that has
    already been folded into `batch_id` by the caller via `batches` count).
    Here `batch_id` is the full line id as used by the GPU kernel (`0 .. batches`).
    """
    comptime length = Int(dims.static_shape[dim_idx])
    comptime prod = _product_of_dims[dims]()
    comptime n_ortho = prod // length

    var outer = batch_id // n_ortho
    var rem = batch_id % n_ortho
    var complex_index = outer * prod

    # Decode orthogonal coords from the last spatial axis upward.
    comptime for i in reversed(range(dims.rank)):
        comptime if i == dim_idx:
            continue
        comptime di = Int(dims.static_shape[i])
        comptime cs = _spatial_complex_stride[dims, i]()
        var coord = rem % di
        rem = rem // di
        complex_index += coord * cs

    return complex_index * complex_width


@always_inline
def _isqrt_int[n: Int]() -> Int:
    """Integer square root of `n` (floor)."""
    comptime if n < 2:
        return n
    var lo = 1
    var hi = n
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if mid * mid <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


@always_inline
def _largest_divisor_le[N: Int, limit: Int]() -> Int:
    """Largest divisor of `N` in `[2, limit]`, or 1 if none."""
    var d = min(limit, N // 2)
    while d >= 2:
        if N % d == 0:
            return d
        d -= 1
    return 1


@always_inline
def _large_1d_can_split[N: Int, max_factor: Int]() -> Bool:
    """True when `N` factors as N1×N2 with both in `[2, max_factor]`."""
    comptime if N < 4 or max_factor < 2:
        return False
    comptime n2 = _largest_divisor_le[N, min(max_factor, _isqrt_int[N]())]()
    comptime if n2 >= 2 and (N // n2) <= max_factor and (N // n2) >= 2:
        return True
    comptime n1 = _largest_divisor_le[N, max_factor]()
    return n1 >= 2 and (N // n1) <= max_factor and (N // n1) >= 2


@always_inline
def _large_1d_factor_pair[N: Int, max_factor: Int]() -> Tuple[Int, Int]:
    """`(N1, N2)` with `N1*N2=N`, both in `[2, max_factor]`, near √N.

    `N2` is the contiguous first-upload length (matrix columns).
    """
    comptime assert _large_1d_can_split[N, max_factor](), (
        "length does not factor under max_factor for two-upload path"
    )
    comptime lim = min(max_factor, _isqrt_int[N]())
    comptime n2_try = _largest_divisor_le[N, lim]()
    comptime if n2_try >= 2 and (N // n2_try) <= max_factor:
        return (N // n2_try, n2_try)
    comptime n1 = _largest_divisor_le[N, max_factor]()
    return (n1, N // n1)


def _estimate_length_bases[length: Int]() -> List[UInt]:
    """Mixed-radix bases whose product equals `length` (small primes)."""
    # fmt: off
    var lower_primes: Array[Byte, 25] = [
        97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19,
        17, 13, 11, 7, 5, 3, 2
    ]
    # fmt: on
    var bases = List[UInt](capacity=len(lower_primes))
    var processed = 1
    for i in range(len(lower_primes)):
        var prime = UInt(lower_primes[i])
        var amnt_divisible = _times_divisible_by(
            UInt(length // processed), prime
        )
        for _ in range(amnt_divisible):
            bases.append(prime)
            processed *= Int(prime)
        if processed == length:
            bases.reverse()
            return bases^
    return bases^
