from bit import bit_reverse
from complex import ComplexSIMD
from math import pi


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


fn _mixed_radix_digit_reverse[bases: List[UInt]](idx: UInt) -> UInt:
    """Performs mixed-radix digit reversal for an index `idx` based on a
    sequence of `bases`.

    Given N = R_0 * R_1 * ... * R_{M-1}, an input index `k` is represented as:
    k = d_0 + d_1*R_0 + d_2*R_0*R_1 + ... + d_{M-1}*R_0*...*R_{M-2}
    where d_i is the digit for radix R_i.

    The reversed index k' is:
    k' = d_{M-1} + d_{M-2}*R_{M-1} + ... + d_1*R_{M-1}*...*R_2 + d_0*R_{M-1}*...*R_1

    Parameters:
        bases: A List of UInt representing the radices in the order
            R_0, R_1, ..., R_{M-1}.

    Args:
        idx: The input index to be reversed.

    Returns:
        The digit-reversed index.
    """
    var reversed_idx = UInt(0)
    var current_val = idx
    var digits = InlineArray[UInt, len(bases)](uninitialized=True)

    for i in range(len(bases)):
        var current_base = bases[i]
        digits[i] = current_val % current_base
        current_val //= current_base

    var current_product_term_base = UInt(1)
    for i in reversed(range(len(bases))):
        var digit_to_add = digits[i]
        reversed_idx += digit_to_add * current_product_term_base
        current_product_term_base *= bases[i]
    return reversed_idx


fn _get_ordered_items[
    length: UInt, bases: List[UInt]
](out res: InlineArray[Scalar[_get_dtype[length]()], length]):
    """The Butterfly diagram orders indexes by bit-reversed size."""
    res = __type_of(res)(uninitialized=True)

    @parameter
    fn _reversed() -> List[UInt]:
        var rev = List[UInt](capacity=len(bases))
        for item in reversed(bases):
            rev.append(item)
        return rev

    for i in range(length):
        res[i] = _mixed_radix_digit_reverse[_reversed()](i)


# FIXME: mojo limitation. cos and sin can't run at compile time
fn _approx_sin(theta: Scalar) -> __type_of(theta):
    return (
        theta
        - (theta**3) / 6
        + (theta**5) / 120
        - (theta**7) / 5040
        + (theta**9) / 362880
        - (theta**11) / 39916800
        + (theta**13) / 6227020800
        - (theta**15) / 1307674368000
        + (theta**17) / 355687428096000
        - (theta**19) / 121645100408832000
        + (theta**21) / 51090942171709440000
        - (theta**23) / 25852016738884976640000
        + (theta**25) / 15511210043330985984000000
        - (theta**27) / 10888869450418352160768000000
        + (theta**29) / 8841761993739701954543616000000
        - (theta**31) / 8222838654177922817725562880000000
    )


# FIXME: mojo limitation. cos and sin can't run at compile time
fn _approx_cos(theta: Scalar) -> __type_of(theta):
    return (
        1
        - (theta**2) / 2
        + (theta**4) / 24
        - (theta**6) / 720
        + (theta**8) / 40320
        - (theta**10) / 3628800
        + (theta**12) / 479001600
        - (theta**14) / 87178291200
        + (theta**16) / 20922789888000
        - (theta**18) / 6402373705728000
        + (theta**20) / 2432902008176640000
        - (theta**22) / 1124000727777607680000
        + (theta**24) / 620448401733239439360000
        - (theta**26) / 403291461126605635584000000
        + (theta**28) / 304888344611713860501504000000
        - (theta**30) / 265252859812191058636308480000000
    )


fn _get_twiddle_factors[
    length: UInt, dtype: DType
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
        theta = Scalar[dtype]((-2 * pi * n) / N)
        # TODO: this could be more generic using fputils
        # 0.002275657645967044
        # 0.002275657645956386
        res[n - 1] = C(
            _approx_cos(theta).__round__(15), _approx_sin(theta).__round__(15)
        )


fn main():
    # alias items = _get_ordered_items[8, List[UInt](4, 2)]()
    alias items = _get_twiddle_factors[8, DType.float64]()
    print(0, "", ComplexSIMD[DType.float64, 1](1, 0))
    for i in range(len(items)):
        print(i + 1, "", items[i])
