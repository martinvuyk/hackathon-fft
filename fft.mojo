from bit import bit_reverse, count_trailing_zeros, prev_power_of_two
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.host.info import _get_info_from_target
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp, pi
from sys.info import has_accelerator, is_64bit, _accelerator_arch, simdwidthof
from memory.pointer import _GPUAddressSpace


def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt],
    inverse: Bool = False,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
):
    """Calculate the Discrete Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.
        inverse: Whether to run the inverse fourier transform.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.

    Notes:
        If the given bases list does not multiply together to equal the length,
        the builtin algorithm duplicates the biggest values that can still
        divide the length until reaching it.

        This function automatically runs the rfft if the input is real-valued.
        Then copies the symetric results into their corresponding slots in the
        output tensor.
    """
    constrained[
        has_accelerator(), "The current FFT implementation is for GPU only"
    ]()
    constrained[
        1 <= len(in_layout) <= 2, "in_layout must have only 1 or 2 axis"
    ]()

    @parameter
    if len(in_layout) == 2:
        constrained[
            in_layout.shape[1].value() == 2,
            "input must be a complex value tensor i.e. (sequence_length, 2)",
            " or a real valued one (sequence_length,)",
        ]()
    alias length = in_layout.shape[0].value()
    constrained[
        out_layout.shape == IntTuple(length, 2),
        "out_layout shape must be (in_layout.shape[0], 2)",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    @parameter
    fn _reduce_mul[b: List[UInt]](out res: UInt):
        res = UInt(1)
        for base in b:
            res *= base

    @parameter
    fn _is_all_two() -> Bool:
        for base in bases:
            if base != 2:
                return False
        return True

    @parameter
    fn _build_ordered_bases() -> List[UInt]:
        var new_bases: List[UInt]

        @parameter
        if _reduce_mul[bases]() == length:
            new_bases = bases
        else:
            var processed = UInt(1)
            new_bases = List[UInt](capacity=len(bases))

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
        return new_bases

    alias ordered_bases = _build_ordered_bases()

    @parameter
    fn _build_processed_list() -> List[UInt]:
        var processed_list = List[UInt](capacity=len(ordered_bases))
        var processed = 1
        for base in ordered_bases:
            processed_list.append(processed)
            processed *= base
        return processed_list

    alias processed_list = _build_processed_list()
    constrained[
        processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length,
        "powers of the bases must multiply together  to equal the sequence ",
        "length. The builtin algorithm was only able to produce: ",
        ordered_bases.__str__(),
    ]()

    alias gpu_info = _get_info_from_target[_accelerator_arch()]()
    alias max_threads_available = (
        gpu_info.threads_per_multiprocessor * gpu_info.sm_count
    )
    alias max_threads_per_block = (
        max_threads_available // gpu_info.thread_blocks_per_multiprocessor
    )

    @parameter
    if length <= max_threads_per_block:
        ctx.enqueue_function[
            _intra_block_fft_kernel_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                inverse=inverse,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
            ]
        ](
            output,
            x,
            grid_dim=1,
            block_dim=length // ordered_bases[len(ordered_bases) - 1],
        )
        _ = processed_list  # origin bug
    elif length <= max_threads_available:
        # TODO: Implement for sequences <= max_threads_available
        constrained[
            False,
            "inter_block_fft is not implemented yet. ",
            "max_threads_per_block: ",
            String(max_threads_per_block),
        ]()
    else:
        # TODO: Implement for sequences > max_threads_available
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


@always_inline
def ifft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt],
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
):
    """Calculate the Discrete Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.

    Notes:
        If the given bases list does not multiply together to equal the length,
        the builtin algorithm duplicates the biggest values that can still
        divide the length until reaching it.

        This function automatically runs the rfft if the input is real-valued.
        Then copies the symetric results into their corresponding slots in the
        output tensor.
    """
    fft[bases=bases, inverse=True](output, x, ctx)


# ===-----------------------------------------------------------------------===#
# inter_block
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# intra_block
# ===-----------------------------------------------------------------------===#


fn _intra_block_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """An FFT that assumes `sequence_length <= max_threads_per_block` and that
    there is only one block."""
    alias length = in_layout.shape[0].value()
    alias twiddle_factors = _get_twiddle_factors[length, out_dtype, inverse]()
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var shared_f = tb[out_dtype]().row_major[length, 2]().shared().alloc()

    alias do_rfft = len(in_layout) == 1

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base
        var is_execution_thread = local_i < amnt_threads

        @parameter
        if processed == 1:
            # reorder input x(global_i) items to match F(current_item) layout
            if is_execution_thread:
                alias ordered_items = _get_ordered_items[
                    length, ordered_bases
                ]()

                @parameter
                for i in range(base):
                    alias offset = i * amnt_threads
                    var g_idx = global_i + offset

                    var current_item: UInt

                    @parameter
                    if base == length:  # do a DFT on the inputs
                        current_item = g_idx
                    else:
                        current_item = UInt(ordered_items[g_idx])

                    @parameter
                    if do_rfft:
                        shared_f[current_item, 0] = x[g_idx].cast[out_dtype]()
                        # NOTE: filling the imaginary part with 0 is not necessary
                        # because the _radix_n_fft_kernel already sets it to 0
                        # when do_rfft and processed == 1
                    else:
                        alias msg = "in_layout must be complex valued"
                        constrained[len(in_layout) == 2, msg]()
                        constrained[in_layout.shape[1].value() == 2, msg]()
                        # TODO: make sure this is the most efficient
                        shared_f.store(
                            current_item,
                            0,
                            x.load[width=2](g_idx, 0).cast[out_dtype](),
                        )

        barrier()

        alias I = InlineArray[ComplexSIMD[out_dtype, 1], length // base]

        @parameter
        fn _prep_twiddle_factors(out res: InlineArray[I, base - 1]):
            res = __type_of(res)(uninitialized=True)
            alias Sc = Scalar[_get_dtype[length * base]()]
            alias offset = Sc(processed)

            alias next_offset = offset * Sc(base)
            alias ratio = Sc(length) // next_offset
            for j in range(1, base):
                res[j - 1] = I(uninitialized=True)
                for local_i in range(length // base):
                    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (
                        offset * Sc(base)
                    )
                    var twiddle_idx = ((Sc(j) * n) % next_offset) * ratio
                    res[j - 1][local_i] = twiddle_factors[
                        twiddle_idx - 1
                    ] if twiddle_idx != 0 else ComplexSIMD[out_dtype, 1](1, 0)

        if is_execution_thread:
            alias twf = _prep_twiddle_factors()
            _radix_n_fft_kernel[
                out_dtype=out_dtype,
                out_layout = shared_f.layout,
                address_space = shared_f.address_space,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                length_base = length // base,
                twiddle_factors=twf,
            ](shared_f, local_i)
        barrier()

    # when in the last stage, copy back to global memory
    # TODO: make sure this is the most efficient

    @parameter
    for i in range(ordered_bases[len(ordered_bases) - 1]):
        alias offset = i * length // ordered_bases[len(ordered_bases) - 1]
        var res = shared_f.load[width=2](local_i + offset, 0)
        output.store(global_i + offset, 0, res)
    barrier()


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


fn _radix_n_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    length_base: UInt,
    twiddle_factors: InlineArray[
        InlineArray[ComplexSIMD[out_dtype, 1], length_base], base - 1
    ],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    local_i: UInt,
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations, at the cost of a bit of branching."""
    constrained[length >= base, "length must be >= base"]()
    alias Sc = Scalar[_get_dtype[length * base]()]
    alias is_rfft_final_stage = do_rfft and processed * base == length
    alias rfft_idx_limit = length // 2
    alias is_first_rfft_stage = do_rfft and processed == 1

    @parameter
    if is_rfft_final_stage:
        if local_i > rfft_idx_limit:
            return

    alias offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    alias Co = ComplexSIMD[out_dtype, output.element_size]
    alias is_even = length % 2 == 0  # avoid evaluating for uneven
    alias base_twf = _get_twiddle_factors[base, out_dtype, inverse]()

    @parameter
    fn _base_phasor[i: UInt, j: UInt]() -> ComplexSIMD[out_dtype, 1]:
        var val = ComplexSIMD[out_dtype, 1](1, 0)

        @parameter
        for _ in range(i):
            val *= base_twf[j - 1]
        return val

    @parameter
    @always_inline
    fn _twf_fma(twf: Co, x_j: Co, acc: Co, out x_i: Co):
        if twf.re == 1:  # Co(1, 0)
            x_i = x_j + acc
        elif is_even and twf.im == -1:  # Co(0, -1)
            x_i = Co(acc.re + x_j.im, acc.im - x_j.re)
        elif is_even and twf.re == -1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, acc.im - x_j.im)
        elif is_even and twf.im == 1:  # Co(0, 1)
            x_i = Co(acc.re - x_j.im, acc.im + x_j.re)
        else:
            x_i = twf.fma(x_j, acc)

    @parameter
    @always_inline
    fn _twf_fma[twf: Co, is_j1: Bool](x_j: Co, acc: Co, out x_i: Co):
        @parameter
        if do_rfft and twf.re == 1 and is_j1:  # Co(1, 0)
            x_i = Co(acc.re + x_j.re, 0)
        elif do_rfft and twf.re == 1:  # Co(1, 0)
            x_i = Co(acc.re + x_j.re, acc.im)
        elif twf.re == 1:  # Co(1, 0)
            x_i = x_j + acc
        elif is_even and do_rfft and twf.im == -1 and is_j1:  # Co(0, -1)
            x_i = Co(acc.re, -x_j.re)
        elif is_even and do_rfft and twf.im == -1:  # Co(0, -1)
            x_i = Co(acc.re, acc.im - x_j.re)
        elif is_even and twf.im == -1:  # Co(0, -1)
            x_i = Co(acc.re + x_j.im, acc.im - x_j.re)
        elif is_even and do_rfft and twf.re == -1 and is_j1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, 0)
        elif is_even and do_rfft and twf.re == -1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, acc.im)
        elif is_even and twf.re == -1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, acc.im - x_j.im)
        elif is_even and do_rfft and twf.im == 1 and is_j1:  # Co(0, 1)
            x_i = Co(acc.re, x_j.re)
        elif is_even and do_rfft and twf.im == 1:  # Co(0, 1)
            x_i = Co(acc.re, acc.im + x_j.re)
        elif is_even and twf.im == 1:  # Co(0, 1)
            x_i = Co(acc.re - x_j.im, acc.im + x_j.re)
        elif do_rfft:
            x_i = Co(twf.re.fma(x_j.re, acc.re), twf.im.fma(x_j.re, acc.im))
        else:
            x_i = twf.fma(x_j, acc)

    var x = InlineArray[Co, base](uninitialized=True)
    var x_out = InlineArray[Co, base](uninitialized=True)

    @parameter
    for i in range(base):
        var idx = Int(n + i * offset)

        @parameter
        if do_rfft and processed == 1:
            x[i] = Co(output[idx, 0], 0)
        else:
            var data = output.load[2 * output.element_size](idx, 0)
            x[i] = UnsafePointer(to=data).bitcast[Co]()[]

    @parameter
    for j in range(1, base):

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                alias base_phasor = rebind[Co](_base_phasor[i, j]())

                @parameter
                if j == 1:
                    x_out[i] = _twf_fma[base_phasor, True](x[j], x[0])
                else:
                    x_out[i] = _twf_fma[base_phasor, False](x[j], x_out[i])
            continue

        alias j_array = twiddle_factors[j - 1]
        ref i0_j_twf = j_array.unsafe_get(local_i)

        @parameter
        for i in range(base):
            alias base_phasor = rebind[Co](_base_phasor[i, j]())
            var twf: Co

            @parameter
            if base_phasor.re == 1:  # Co(1, 0)
                twf = rebind[Co](i0_j_twf)
            elif base_phasor.im == -1:  # Co(0, -1)
                twf = Co(i0_j_twf.im, -i0_j_twf.re)
            elif base_phasor.re == -1:  # Co(-1, 0)
                twf = -rebind[Co](i0_j_twf)
            elif base_phasor.im == 1:  # Co(0, 1)
                twf = Co(-i0_j_twf.im, i0_j_twf.re)
            else:
                twf = rebind[Co](i0_j_twf) * base_phasor

            @parameter
            if j == 1:
                x_out[i] = _twf_fma(twf, x[j], x[0])
            else:
                x_out[i] = _twf_fma(twf, x[j], x_out[i])

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        alias factor = Scalar[out_dtype](1 / length)

        @parameter
        for i in range(base):
            x_out[i].re *= factor
            x_out[i].im *= factor

    @parameter
    for i in range(base):
        # TODO: make sure this is the most efficient
        var idx = n + i * offset
        var ptr = x_out.unsafe_ptr().bitcast[Scalar[out_dtype]]() + 2 * i
        var res = ptr.load[width=2]()
        output.store(Int(idx), 0, res)

        @parameter
        if is_rfft_final_stage:  # copy the symmetric conjugates
            # when idx == 0 its conjugate is idx == length + 1
            # when the sequence length is even then the next_idx can be idx
            # when idx == rfft_idx_limit
            var next_idx = Sc(length) * Int(idx != 0) - idx
            if next_idx != idx:
                output.store(Int(next_idx), 0, res[0].join(-res[1]))


# ===-----------------------------------------------------------------------===#
# utils
# ===-----------------------------------------------------------------------===#


# I know this seems overkill but it's better to save cache. and some
# operations like modulo can become faster depending on the micro-architecture.
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

    Parameters:
        bases: A List of UInt representing the radices in the order
            `R_0, R_1, ..., R_{M-1}`.

    Args:
        idx: The input index to be reversed.

    Returns:
        The digit-reversed index.

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
    fn _is_all_two() -> Bool:
        for base in bases:
            if base != 2:
                return False
        return True

    @parameter
    if _is_all_two():
        values = List[__type_of(res).ElementType](capacity=length)
        for i in range(__type_of(res).ElementType(length)):
            values.append(bit_reverse(i))
        sort(values)
        for i in range(length):
            res[i] = bit_reverse(values[i])
    else:

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
            var theta = Scalar[dtype](-factor * pi)
            # TODO: Rounding to 15 is very arbitrary, find a good value and
            # justify it
            num = C(
                _approx_cos(theta).__round__(15),
                _approx_sin(theta).__round__(15),
            )

        @parameter
        if not inverse:
            res[n - 1] = num
        else:
            res[n - 1] = C(num.re, -num.im)


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
