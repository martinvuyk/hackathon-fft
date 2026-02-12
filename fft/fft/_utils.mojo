from sys import is_compile_time
from sys.info import is_64bit, is_nvidia_gpu
from complex import ComplexScalar
from math import exp, pi, ceil, sin, cos, log2
from bit import count_trailing_zeros
from gpu.host.info import is_cpu
from layout import IntTuple
from utils.index import IndexList


fn _get_dtype[length: UInt]() -> DType:
    @parameter
    if length <= UInt(UInt8.MAX):
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


fn _mixed_radix_digit_reverse[
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

    @parameter
    if reverse:
        base_offset = 1
    else:
        base_offset = {length}

    @parameter
    for i in range(len(ordered_bases)):
        comptime base = type_of(idx)(
            ordered_bases[i if not reverse else (len(ordered_bases) - 1 - i)]
        )

        @parameter
        if not reverse:
            base_offset //= base
        reversed_idx += (current_val % base) * base_offset
        current_val //= base

        @parameter
        if reverse:
            base_offset *= base
    return reversed_idx


fn _get_twiddle_factor[
    dtype: DType, *, inverse: Bool, N: UInt
](n: UInt) -> ComplexScalar[dtype]:
    """Returns `exp((-j * 2 * pi * n) / N)`."""
    comptime assert dtype.is_floating_point()
    comptime `-2π/N` = Scalar[dtype](-2.0 * pi) / Scalar[dtype](N)
    var theta = `-2π/N` * Scalar[dtype](n)

    var num: ComplexScalar[dtype]

    if is_compile_time():
        var factor = 2 * n.cast[dtype]() / N.cast[dtype]()
        if factor < 1e-9:  # approx. 0
            num = {1, 0}
        elif factor == 0.5:
            num = {0, -1}
        elif factor == 1:
            num = {-1, 0}
        elif factor == 1.5:
            num = {0, 1}
        else:
            # FIXME: remove once comptime branch and comptime assert don't short circuit
            @parameter
            if is_nvidia_gpu():
                num = {
                    cos(theta.cast[DType.float32]()).cast[dtype](),
                    sin(theta.cast[DType.float32]()).cast[dtype](),
                }
            else:
                num = {cos(theta), sin(theta)}
    else:

        @parameter
        if is_nvidia_gpu():
            num = {
                cos(theta.cast[DType.float32]()).cast[dtype](),
                sin(theta.cast[DType.float32]()).cast[dtype](),
            }
        else:
            num = {cos(theta), sin(theta)}

    @parameter
    if not inverse:
        return num
    else:
        return num.conj()


fn _get_twiddle_factors[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: List[ComplexScalar[dtype]]):
    """Get all the twiddle factors for the length."""
    res = {unsafe_uninit_length = Int(length)}
    for n in range(length):
        res[n] = _get_twiddle_factor[dtype, inverse=inverse, N=length](n)


fn _get_twiddle_factors_inline[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: InlineArray[ComplexScalar[dtype], Int(length)]):
    """Get all the twiddle factors for the length."""
    res = {uninitialized = True}
    for n in range(length):
        res[n] = _get_twiddle_factor[dtype, inverse=inverse, N=length](n)


fn _div_by(x: UInt, base: UInt) -> UInt:
    # TODO: benchmark whether this performs better than doing branches
    return 1 if base == x else (
        0 if (base > x or x % base != 0) else (_div_by(x // base, base) + 1)
    )


fn _times_divisible_by(length: UInt, base: UInt, out amnt_divisible: UInt):
    debug_assert(base != 1, "The number 1 can infinitely divide any number")
    if UInt64(base).is_power_of_two():
        # FIXME(#5003): this should work
        # amnt_divisible = UInt(
        #     count_trailing_zeros(Scalar[DType.uint](length))
        #     // log2(Float64(base)).cast[DType.uint]()
        # )

        @parameter
        if is_64bit():
            amnt_divisible = UInt(
                count_trailing_zeros(UInt64(length))
                // log2(Float64(base)).cast[DType.uint64]()
            )
        else:
            amnt_divisible = UInt(
                count_trailing_zeros(UInt32(length))
                // log2(Float64(base)).cast[DType.uint32]()
            )
    else:
        amnt_divisible = _div_by(length, base)


@parameter
fn _reduce_mul(b: List[UInt], out res: UInt):
    res = UInt(1)
    for base in b:
        res *= base


@parameter
fn _build_ordered_bases[
    length: UInt
](bases: List[UInt], out new_bases: List[UInt]):
    if _reduce_mul(bases) == length:
        new_bases = bases.copy()
        sort(new_bases)  # FIXME: this should just be ascending=False
        new_bases.reverse()
    else:
        var existing_bases = bases.copy()
        sort(existing_bases)
        new_bases = List[UInt](capacity=len(existing_bases))

        var processed = UInt(1)
        for i in reversed(range(len(existing_bases))):
            var base = existing_bases[i]
            var amnt_divisible = _times_divisible_by(length, base)
            new_bases.reserve(Int(amnt_divisible))
            for _ in range(amnt_divisible):
                new_bases.append(base)
                processed *= base

            if processed == length:
                return


fn _get_ordered_bases_processed_list[
    length: UInt, bases: List[UInt]
]() -> Tuple[List[UInt], List[UInt]]:
    comptime ordered_bases = _build_ordered_bases[length](materialize[bases]())

    @parameter
    fn _build_processed_list() -> List[UInt]:
        var ordered_bases_var = materialize[ordered_bases]()
        var processed_list = List[UInt](capacity=len(ordered_bases_var))
        var processed = UInt(1)
        for base in ordered_bases_var:
            processed_list.append(processed)
            processed *= base
        return processed_list^

    comptime processed_list = _build_processed_list()
    constrained[
        processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length,
        "powers of the bases must multiply together  to equal the sequence ",
        "length. The builtin algorithm was only able to produce: ",
        ordered_bases.__str__().replace("UInt(", "").replace(")", ""),
        " for the length: ",
        String(length),
    ]()
    comptime assert 1 not in ordered_bases, "Cannot do an fft with base 1."
    return materialize[ordered_bases](), materialize[processed_list]()


@always_inline
fn _min(elems: List[UInt], out smallest: UInt):
    smallest = elems[0]
    for elem in elems[1:]:
        smallest = min(elem, smallest)


@always_inline
fn _max(elems: List[UInt], out smallest: UInt):
    smallest = elems[0]
    for elem in elems[1:]:
        smallest = max(elem, smallest)


@always_inline
fn _max(elems: List[List[UInt]], out biggest: UInt):
    biggest = 0
    for bases in elems:
        for elem in bases:
            biggest = max(elem, biggest)


@always_inline
fn _product_of_dims(dims: IntTuple) -> Int:
    """Calculates the product of a tuple of dimensions."""
    var prod = 1
    for i in range(len(dims)):
        prod *= dims[i].value()
    return prod


fn _get_cascade_idxes[
    shape: IntTuple, excluded: IntTuple
](var flat_idx: Int, out idxes: IndexList[len(shape) - len(excluded)]):
    idxes = {fill = 0}

    @parameter
    fn _idxes_i(i: Int, out amnt: Int):
        amnt = i

        @parameter
        for j in range(len(excluded)):
            comptime val = excluded[j].value()
            amnt -= Int(i > val)

    @parameter
    fn _is_excluded[i: Int]() -> Bool:
        @parameter
        for j in range(len(excluded)):

            @parameter
            if i == excluded[j].value():
                return True
        return False

    @parameter
    for i in range(len(shape)):

        @parameter
        if _is_excluded[i]():
            continue
        comptime curr_num = UInt(shape[i].value())
        comptime idxes_i = _idxes_i(i)
        idxes[idxes_i] = Int(UInt(flat_idx) % curr_num)
        flat_idx = Int(UInt(flat_idx) // curr_num)


@always_inline
fn _unit_phasor_mul[
    phasor: ComplexScalar
](twf: type_of(phasor)) -> type_of(phasor):
    """Optimizes `phasor * twf`."""

    @parameter
    if phasor.re == 1:  # Co(1, 0)
        return twf
    elif phasor.im == -1:  # Co(0, -1)
        return {twf.im, -twf.re}
    elif phasor.re == -1:  # Co(-1, 0)
        return -twf
    elif phasor.im == 1:  # Co(0, 1)
        return {-twf.im, twf.re}
    elif abs(phasor.re) == abs(phasor.im):  # Co(1/√2, 1/√2)
        comptime factor = abs(phasor.re)

        @parameter
        if phasor.re > 0 and phasor.im > 0:  # Q1
            return {factor * (twf.re - twf.im), factor * (twf.re + twf.im)}
        elif phasor.re < 0 and phasor.im > 0:  # Q2
            return {factor * (-twf.re - twf.im), factor * (twf.re - twf.im)}
        elif phasor.re < 0 and phasor.im < 0:  # Q3
            return {factor * (-twf.re + twf.im), factor * (-twf.re - twf.im)}
        else:  # Q4
            return {factor * (twf.re + twf.im), factor * (-twf.re + twf.im)}
    else:
        return twf * phasor


@always_inline
fn _unit_phasor_fma[
    dtype: DType, //, twf: ComplexScalar[dtype]
](x_j: ComplexScalar[dtype], acc: ComplexScalar[dtype]) -> ComplexScalar[dtype]:
    @parameter
    if twf.re == 1:  # Co(1, 0)
        return acc + x_j
    elif twf.im == -1:  # Co(0, -1)
        return {acc.re + x_j.im, acc.im - x_j.re}
    elif twf.re == -1:  # Co(-1, 0)
        return acc - x_j
    elif twf.im == 1:  # Co(0, 1)
        return {acc.re - x_j.im, acc.im + x_j.re}
    elif abs(twf.re) == abs(twf.im):  # Co(1/√2, 1/√2)
        comptime factor = abs(twf.re)
        var re = x_j.re
        var im = x_j.im

        @parameter
        if twf.re > 0 and twf.im > 0:  # Q1
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
fn _unit_phasor_fma[
    dtype: DType,
    //,
    twf: ComplexScalar[dtype],
    accum_is_real: Bool,
](x_j: Scalar[dtype], acc: ComplexScalar[dtype]) -> ComplexScalar[dtype]:
    @parameter
    if twf.re == 1:  # Co(1, 0)
        return {acc.re + x_j, acc.im}
    elif twf.im == -1 and accum_is_real:  # Co(0, -1)
        return {acc.re, -x_j}
    elif twf.im == -1:  # Co(0, -1)
        return {acc.re, acc.im - x_j}
    elif twf.re == -1:  # Co(-1, 0)
        return {acc.re - x_j, acc.im}
    elif twf.im == 1 and accum_is_real:  # Co(0, 1)
        return {acc.re, x_j}
    elif twf.im == 1:  # Co(0, 1)
        return {acc.re, acc.im + x_j}
    elif accum_is_real:
        return {twf.re.fma(x_j, acc.re), twf.im * x_j}
    else:
        return {
            from_interleaved = twf.re.join(twf.im).fma(
                x_j.join(x_j), acc.re.join(acc.im)
            )
        }
