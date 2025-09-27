from sys.info import is_64bit
from complex import ComplexSIMD
from math import exp, pi, ceil
from bit import (
    bit_reverse,
    count_trailing_zeros,
    prev_power_of_two,
    next_power_of_two,
)


fn _get_dtype[length: UInt]() -> DType:
    @parameter
    if length < UInt(UInt8.MAX):
        return DType.uint8
    elif length < UInt(UInt16.MAX):
        return DType.uint16
    elif length < UInt(UInt32.MAX):
        return DType.uint32
    elif UInt64(length) < UInt64.MAX:
        return DType.uint64
    elif UInt128(length) < UInt128.MAX:
        return DType.uint128
    else:
        return DType.uint256


fn _mixed_radix_digit_reverse[
    length: UInt, ordered_bases: List[UInt]
](idx: UInt) -> UInt:
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
    var reversed_idx = UInt(0)
    var current_val = idx
    # var base_offset = length

    # @parameter
    # for i in reversed(range(len(ordered_bases))):
    #     alias base = ordered_bases[i]
    #     base_offset //= base
    #     reversed_idx += (current_val % base) * base_offset
    #     current_val //= base
    # return reversed_idx

    var base_offset = length

    @parameter
    for i in range(len(ordered_bases)):
        alias base = ordered_bases[i]
        base_offset //= base
        reversed_idx += (current_val % base) * base_offset
        current_val //= base
    return reversed_idx


fn _get_ordered_items[
    length: UInt, ordered_bases: List[UInt]
](out res: InlineArray[Scalar[_get_dtype[length]()], length]):
    """The Butterfly diagram orders indexes by digit."""
    res = {uninitialized = True}
    alias E = __type_of(res).ElementType

    @parameter
    fn _is_all_two() -> Bool:
        for base in materialize[ordered_bases]():
            if base != 2:
                return False
        return True

    @parameter
    if _is_all_two():
        var values = List[E](capacity=length)
        for i in range(E(length)):
            values.append(bit_reverse(i))
        sort(values)
        for i in range(length):
            res[bit_reverse(values[i])] = i
    else:
        for i in range(length):
            res[_mixed_radix_digit_reverse[length, ordered_bases](i)] = i


# FIXME: mojo limitation. cos and sin can't run at compile time
fn _approx_sin(theta: Scalar) -> __type_of(theta):
    t = theta.cast[DType.float64]()
    return (
        t
        - (t**3) / 6.0
        + (t**5) / 120.0
        - (t**7) / 5040.0
        + (t**9) / 362880.0
        - (t**11) / 39916800.0
        + (t**13) / 6227020800.0
        - (t**15) / 1307674368000.0
        + (t**17) / 355687428096000.0
        - (t**19) / 121645100408832000.0
        + (t**21) / 51090942171709440000.0
        - (t**23) / 25852016738884976640000.0
        + (t**25) / 15511210043330985984000000.0
        - (t**27) / 10888869450418352160768000000.0
        + (t**29) / 8841761993739701954543616000000.0
        - (t**31) / 8222838654177922817725562880000000.0
        + (t**33) / 8683317618811886495518194401280000000.0
        - (t**35) / 10333147966386144929666651337523200000000.0
        + (t**37) / 13763753091226345046315979581580902400000000.0
        - (t**39) / 20397882081197443358640281739902897356800000000.0
        + (t**41) / 33452526613163807108170062053440751665152000000000.0
        - (t**43) / 60415263063373835637355132068513997507264512000000000.0
        + (t**45)
        / 119622220865480194561963161495657715064383733760000000000.0
        - (t**47)
        / 258623241511168180642964355153611979969197632389120000000000.0
    ).cast[theta.dtype]()


# FIXME: mojo limitation. cos and sin can't run at compile time
fn _approx_cos(theta: Scalar) -> __type_of(theta):
    t = theta.cast[DType.float64]()
    return (
        1.0
        - (t**2) / 2.0
        + (t**4) / 24.0
        - (t**6) / 720.0
        + (t**8) / 40320.0
        - (t**10) / 3628800.0
        + (t**12) / 479001600.0
        - (t**14) / 87178291200.0
        + (t**16) / 20922789888000.0
        - (t**18) / 6402373705728000.0
        + (t**20) / 2432902008176640000.0
        - (t**22) / 1124000727777607680000.0
        + (t**24) / 620448401733239439360000.0
        - (t**26) / 403291461126605635584000000.0
        + (t**28) / 304888344611713860501504000000.0
        - (t**30) / 265252859812191058636308480000000.0
        + (t**32) / 263130836933693530167218012160000000.0
        - (t**34) / 295232799039604140847618609643520000000.0
        + (t**36) / 371993326789901217467999448150835200000000.0
        - (t**38) / 523022617466601111760007224100074291200000000.0
        + (t**40) / 815915283247897734345611269596115894272000000000.0
        - (t**42) / 1405006117752879898543142606244511569936384000000000.0
        + (t**44) / 2658271574788448768043625811014615890319638528000000000.0
        - (t**46)
        / 5502622159812088949850305428800254892961651752960000000000.0
    ).cast[theta.dtype]()


fn _get_twiddle_factors[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: InlineArray[ComplexSIMD[dtype, 1], length - 1]):
    """Get the twiddle factors for the length.

    Examples:
        for a signal with 8 datapoints:
        the result is: [W_1_8, W_2_8, W_3_8, W_4_8, W_5_8, W_6_8, W_7_8]
    """
    alias C = ComplexSIMD[dtype, 1]
    res = __type_of(res)(uninitialized=True)
    alias N = length
    for n in range(1, N):
        # exp((-j * 2 * pi * n) / N)
        var factor = 2 * n / N

        var num: C

        if factor == 0:
            num = C(1, 0)
        elif factor == 0.5:
            num = C(0, -1)
        elif factor == 1:
            num = C(-1, 0)
        elif factor == 1.5:
            num = C(0, 1)
        else:
            var theta = Float64(-factor * pi)
            num = C(
                _approx_cos(theta).cast[dtype](),
                _approx_sin(theta).cast[dtype](),
            )

        @parameter
        if not inverse:
            res[n - 1] = num
        else:
            res[n - 1] = C(num.re, -num.im)


fn _prep_twiddle_factors[
    length: UInt, base: UInt, processed: UInt, dtype: DType, inverse: Bool
](
    out res: InlineArray[
        InlineArray[ComplexSIMD[dtype, 1], base - 1], length // base
    ]
):
    alias twiddle_factors = _get_twiddle_factors[length, dtype, inverse]()
    res = {uninitialized = True}
    alias Sc = Scalar[_get_dtype[length * base]()]
    alias offset = Sc(processed)

    alias next_offset = offset * Sc(base)
    alias ratio = Sc(length) // next_offset
    for local_i in range(length // base):
        res[local_i] = {uninitialized = True}
        for j in range(1, base):
            var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (
                offset * Sc(base)
            )
            var twiddle_idx = ((Sc(j) * n) % next_offset) * ratio
            res[local_i][j - 1] = twiddle_factors[
                twiddle_idx - 1
            ] if twiddle_idx != 0 else {1, 0}


@parameter
fn _get_flat_twfs[
    dtype: DType,
    length: UInt,
    total_twfs: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    inverse: Bool,
](out res: InlineArray[Scalar[dtype], total_twfs * 2]):
    res = {uninitialized = True}
    var idx = 0

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base
        var base_twfs = _prep_twiddle_factors[
            length, base, processed, dtype, inverse
        ]()

        for i in range(base_twfs.size):
            for j in range(base_twfs[0].size):
                var t = base_twfs[i][j]
                res[idx] = t.re
                idx += 1
                res[idx] = t.im
                idx += 1


fn _log_mod[base: UInt](x: UInt) -> (UInt, UInt):
    """Get the maximum exponent of base that fully divides x and the
    remainder.
    """
    var div = x // base

    @parameter
    fn _run() -> (UInt, UInt):
        ref res = _log_mod[base](div)
        res[0] += 1
        return res

    # TODO: benchmark whether this performs better than doing branches
    return (UInt(0), x) if x % base != 0 else (
        (UInt(1), UInt(0)) if div == 1 else _run()
    )


fn _get_ordered_bases_processed_list[
    length: UInt, bases: List[UInt]
]() -> (List[UInt], List[UInt]):
    @parameter
    fn _reduce_mul[b: List[UInt]](out res: UInt):
        res = UInt(1)
        for base in materialize[b]():
            res *= base

    @parameter
    fn _is_all_two() -> Bool:
        for base in materialize[bases]():
            if base != 2:
                return False
        return True

    @parameter
    fn _build_ordered_bases() -> List[UInt]:
        var new_bases: List[UInt]

        @parameter
        if _reduce_mul[bases]() == length:
            new_bases = materialize[bases]()
        else:
            var processed = UInt(1)
            new_bases = List[UInt](capacity=len(materialize[bases]()))

            @parameter
            for base in bases:
                var amnt_divisible: UInt

                @parameter
                if _is_all_two() and length.is_power_of_two() and base == 2:
                    # FIXME(#5003): this should just be Scalar[DType.index]
                    @parameter
                    if is_64bit():
                        amnt_divisible = UInt(
                            count_trailing_zeros(UInt64(length))
                        )
                    else:
                        amnt_divisible = UInt(
                            count_trailing_zeros(UInt32(length))
                        )
                else:
                    amnt_divisible = _log_mod[base](length // processed)[0]
                for _ in range(amnt_divisible):
                    new_bases.append(base)
                    processed *= base
        sort(new_bases)  # FIXME: this should just be ascending=False
        new_bases.reverse()
        return new_bases^

    alias ordered_bases = _build_ordered_bases()

    @parameter
    fn _build_processed_list() -> List[UInt]:
        var processed_list = List[UInt](
            capacity=len(materialize[ordered_bases]())
        )
        var processed = 1
        for base in materialize[ordered_bases]():
            processed_list.append(processed)
            processed *= base
        return processed_list^

    alias processed_list = _build_processed_list()
    constrained[
        processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length,
        "powers of the bases must multiply together  to equal the sequence ",
        "length. The builtin algorithm was only able to produce: ",
        ordered_bases.__str__(),
    ]()
    constrained[1 not in ordered_bases, "Cannot do an fft with base 1."]()
    return materialize[ordered_bases](), materialize[processed_list]()
