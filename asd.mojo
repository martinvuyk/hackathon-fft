from bit import bit_reverse


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


fn main():
    alias items = _get_ordered_items[8, List[UInt](4, 2)]()
    for i in range(len(items)):
        print(items[i])
