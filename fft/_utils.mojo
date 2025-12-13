from sys.info import is_64bit
from complex import ComplexScalar
from math import exp, pi, ceil, sin, cos
from bit import count_trailing_zeros
from gpu.host.info import is_cpu
from layout import IntTuple
from utils.index import IndexList


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
    var base_offset: UInt

    @parameter
    if reverse:
        base_offset = 1
    else:
        base_offset = length

    @parameter
    for i in range(len(ordered_bases)):
        comptime base = ordered_bases[
            i if not reverse else (len(ordered_bases) - 1 - i)
        ]

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
    dtype: DType, *, inverse: Bool = False
](n: UInt, N: UInt) -> ComplexScalar[dtype]:
    """Returns `exp((-j * 2 * pi * n) / N)`."""
    var factor = (2 * n) / N

    var num: ComplexScalar[dtype]

    if factor == 0:
        num = {1, 0}
    elif factor == 0.5:
        num = {0, -1}
    elif factor == 1:
        num = {-1, 0}
    elif factor == 1.5:
        num = {0, 1}
    else:
        var theta = Float64(-factor * pi)
        num = {cos(theta).cast[dtype](), sin(theta).cast[dtype]()}

    @parameter
    if not inverse:
        return num
    else:
        return {num.re, -num.im}


fn _get_twiddle_factors[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: InlineArray[ComplexScalar[dtype], Int(length - 1)]):
    """Get the twiddle factors for the length.

    Examples:
        for a signal with 8 datapoints:
        the result is: [W_1_8, W_2_8, W_3_8, W_4_8, W_5_8, W_6_8, W_7_8]
    """
    res = type_of(res)(uninitialized=True)
    comptime N = length
    for n in range(UInt(1), N):
        res[n - 1] = _get_twiddle_factor[dtype, inverse=inverse](n, N)


fn _prep_twiddle_factors[
    length: UInt, base: UInt, processed: UInt, dtype: DType, inverse: Bool
](
    out res: InlineArray[
        InlineArray[ComplexScalar[dtype], Int(base - 1)], Int(length // base)
    ]
):
    comptime twiddle_factors = _get_twiddle_factors[length, dtype, inverse]()
    res = {uninitialized = True}
    comptime Sc = Scalar[_get_dtype[length * base]()]
    comptime offset = Sc(processed)

    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset
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
](out res: InlineArray[Scalar[dtype], Int(total_twfs * 2)]):
    res = {uninitialized = True}
    var idx = 0

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
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


fn _log_mod(x: UInt, base: UInt) -> Tuple[UInt, UInt]:
    """Get the maximum exponent of base that fully divides x and the
    remainder.
    """
    var div = x // base

    @parameter
    fn _run() -> Tuple[UInt, UInt]:
        ref res = _log_mod(div, base)
        res[0] += 1
        return res

    # TODO: benchmark whether this performs better than doing branches
    return (UInt(0), x) if x % base != 0 else (
        (UInt(1), UInt(0)) if div == 1 else _run()
    )


@parameter
fn _reduce_mul(b: List[UInt], out res: UInt):
    res = UInt(1)
    for base in b:
        res *= base


@parameter
fn _is_all_two(existing_bases: List[UInt]) -> Bool:
    for base in existing_bases:
        if base != 2:
            return False
    return True


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
        for base in existing_bases:
            var amnt_divisible: UInt

            if (
                _is_all_two(existing_bases)
                and length.is_power_of_two()
                and base == 2
            ):
                # FIXME(#5003): this should just be Scalar[DType.index]
                @parameter
                if is_64bit():
                    amnt_divisible = UInt(count_trailing_zeros(UInt64(length)))
                else:
                    amnt_divisible = UInt(count_trailing_zeros(UInt32(length)))
            else:
                amnt_divisible = _log_mod(length // processed, base)[0]
            for _ in range(amnt_divisible):
                new_bases.append(base)
                processed *= base

        new_bases.reverse()


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
        ordered_bases.__str__(),
    ]()
    constrained[1 not in ordered_bases, "Cannot do an fft with base 1."]()
    return materialize[ordered_bases](), materialize[processed_list]()


@always_inline
fn _min(elems: List[UInt], out smallest: UInt):
    smallest = elems[0]
    for elem in elems[1:]:
        smallest = min(elem, smallest)


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
    shape: IntTuple, excluded: Tuple
](var flat_idx: Int, out idxes: IndexList[len(shape) - len(excluded)],):
    idxes = 0
    var j_idxes = 0
    for j in range(len(shape)):
        if j in excluded:
            continue
        while flat_idx > 0 and idxes[j_idxes] < shape[j].value() - 1:
            var i_idxes = j_idxes + 1
            for i in range((j + 1), len(shape)):
                if i in excluded:
                    continue
                var max_dim_idx = shape[i].value() - 1
                if idxes[i_idxes] + flat_idx <= max_dim_idx:
                    idxes[i_idxes] += flat_idx
                    flat_idx = 0
                    break
                idxes[i_idxes] = max_dim_idx
                flat_idx -= max_dim_idx
                i_idxes += 1

            if flat_idx > 0:
                idxes[j_idxes] += 1
                i_idxes = j_idxes + 1
                for i in range((j + 1), len(shape)):
                    if i in excluded:
                        continue
                    idxes[i_idxes] = 0
                    i_idxes += 1
                flat_idx -= 1
        j_idxes += 1
