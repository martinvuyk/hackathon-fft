from algorithm import parallelize, vectorize
from builtin.globals import global_constant
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext
from gpu.host.info import Vendor, is_cpu
from layout import Layout, LayoutTensor, IntTuple
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import simd_width_of

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
)

# ===-----------------------------------------------------------------------===#
# inter_multiprocessor_fft
# ===-----------------------------------------------------------------------===#


fn _inter_multiprocessor_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    twf_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    base: UInt,
    ordered_bases: List[UInt],
    processed: UInt,
    do_rfft: Bool,
    inverse: Bool,
    twf_offset: UInt,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y

    comptime amnt_threads = length // base
    comptime x_out_layout = Layout.row_major(Int(base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutExternalOrigin
    ].stack_allocation()

    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime func = _radix_n_fft_kernel[
        do_rfft=do_rfft,
        base=base,
        length=length,
        processed=processed,
        inverse=inverse,
        twf_offset=twf_offset,
        ordered_bases=ordered_bases,
    ]

    if global_i < amnt_threads:  # is execution thread
        func(output, x, global_i, twiddle_factors, x_out)


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


@always_inline
fn _radix_n_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    in_address_space: AddressSpace,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    x_out_layout: Layout,
    x_out_address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    twf_offset: UInt,
    ordered_bases: List[UInt],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space, ...
    ],
    x: LayoutTensor[
        in_dtype, in_layout, in_origin, address_space=in_address_space, ...
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space, ...
    ],
    x_out: LayoutTensor[
        mut=True,
        out_dtype,
        x_out_layout,
        address_space=x_out_address_space,
        ...,
    ],
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    __comptime_assert length >= base, "length must be >= base"
    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]
    comptime is_even = length % 2 == 0

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twfs = _get_twiddle_factors[base, out_dtype, inverse]()
        comptime base_twf = base_twfs[j - 1]
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

    @parameter
    @always_inline
    fn _twf_fma[twf: Co, is_j1: Bool](x_j: Co, acc: Co) -> CoV:
        var x_i: Co

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
        return to_CoV(x_i)

    @parameter
    fn _get_x[i: UInt]() -> Co:
        @parameter
        if processed == 1:
            # Reorder input x(local_i) items to match F(current_item) layout.
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            @parameter
            if base == length:
                copy_from = idx  # do a DFT on the inputs
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            @parameter
            if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return to_Co(x.load[2](Int(copy_from), 0).cast[out_dtype]())
        else:
            comptime step = Sc(i) * offset
            return to_Co(output.load[2](Int(n + step), 0))

    var x_0 = _get_x[0]()

    @parameter
    for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: Co

                @parameter
                if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.load[2](Int(i), 0))

                x_out.store(Int(i), 0, _twf_fma[base_phasor, j == 1](x_j, acc))
            continue

        var twf_index = Int(twf_offset + local_i * (base - 1) + (j - 1))

        var i0_j_twf_vec = twiddle_factors.load[2](twf_index, 0)
        var i0_j_twf = to_Co(i0_j_twf_vec)

        @parameter
        for i in range(base):
            comptime base_phasor = _base_phasor[i, j]()
            var twf: Co

            @parameter
            if base_phasor.re == 1:  # Co(1, 0)
                twf = i0_j_twf
            elif base_phasor.im == -1:  # Co(0, -1)
                twf = Co(i0_j_twf.im, -i0_j_twf.re)
            elif base_phasor.re == -1:  # Co(-1, 0)
                twf = -i0_j_twf
            elif base_phasor.im == 1:  # Co(0, 1)
                twf = Co(-i0_j_twf.im, i0_j_twf.re)
            else:
                twf = i0_j_twf * base_phasor

            var acc: Co

            @parameter
            if j == 1:
                acc = x_0
            else:
                acc = to_Co(x_out.load[2](Int(i), 0))

            x_out.store(Int(i), 0, to_CoV(twf.fma(x_j, acc)))

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        comptime factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        if base.is_power_of_two():
            x_out.ptr.store(x_out.load[Int(base * 2)](0, 0) * factor)
        else:

            @parameter
            for i in range(base):
                x_out.store(Int(i), 0, x_out.load[2](Int(i), 0) * factor)

    @parameter
    for i in range(base):
        comptime step = Sc(i * offset)
        output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))


@always_inline
fn _radix_n_fft_kernel_exp[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    in_address_space: AddressSpace,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    x_out_layout: Layout,
    x_out_address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    twf_offset: UInt,
    ordered_bases: List[UInt],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space, ...
    ],
    x: LayoutTensor[
        in_dtype, in_layout, in_origin, address_space=in_address_space, ...
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
    x_out: LayoutTensor[
        mut=True, out_dtype, x_out_layout, address_space=x_out_address_space
    ],
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    __comptime_assert length >= base, "length must be >= base"
    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]
    comptime is_even = length % 2 == 0

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _get_x(i: UInt) -> Co:
        @parameter
        if processed == 1:
            # Reorder input x(local_i) items to match F(current_item) layout.
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            @parameter
            if base == length:
                copy_from = idx  # do a DFT on the inputs
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            @parameter
            if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return to_Co(x.load[2](Int(copy_from), 0).cast[out_dtype]())
        else:
            var step = Sc(i) * offset
            return to_Co(output.load[2](Int(n + step), 0))

    var x_0 = _get_x(0)

    for j in range(UInt(1), base):
        var x_j = _get_x(j)

        @parameter
        if processed == 1:
            for i in range(base):
                var acc: Co

                if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.load[2](Int(i), 0))

                x_out.store(Int(i), 0, to_CoV(x_j + acc))
            continue

        var twf_index = Int(twf_offset + local_i * (base - 1) + (j - 1))

        var i0_j_twf_vec = twiddle_factors.load[2](twf_index, 0)
        var i0_j_twf = to_Co(i0_j_twf_vec)

        for i in range(base):
            var twf = i0_j_twf

            var acc: Co

            if j == 1:
                acc = x_0
            else:
                acc = to_Co(x_out.load[2](Int(i), 0))

            x_out.store(Int(i), 0, to_CoV(twf.fma(x_j, acc)))

    if inverse and processed * base == length:  # last ifft stage
        comptime factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        if base.is_power_of_two():
            x_out.ptr.store(x_out.load[Int(base * 2)](0, 0) * factor)
        else:
            for i in range(base):
                x_out.store(Int(i), 0, x_out.load[2](Int(i), 0) * factor)

    for i in range(base):
        var step = Sc(i * offset)
        output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))


@always_inline
fn _radix_n_fft_kernel_exp2[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    in_address_space: AddressSpace,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    x_out_layout: Layout,
    x_out_address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    twf_offset: UInt,
    ordered_bases: List[UInt],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space, ...
    ],
    x: LayoutTensor[
        in_dtype, in_layout, in_origin, address_space=in_address_space, ...
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space, ...
    ],
    x_out: LayoutTensor[
        mut=True,
        out_dtype,
        x_out_layout,
        address_space=x_out_address_space,
        ...,
    ],
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    __comptime_assert length >= base, "length must be >= base"
    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]
    comptime is_even = length % 2 == 0

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twfs = _get_twiddle_factors[base, out_dtype, inverse]()
        comptime base_twf = base_twfs[j - 1]
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

    @parameter
    @always_inline
    fn _twf_fma[twf: Co, is_j1: Bool](x_j: Co, acc: Co) -> CoV:
        var x_i: Co

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
        return to_CoV(x_i)

    @parameter
    fn _get_x[i: UInt]() -> Co:
        @parameter
        if processed == 1:
            # Reorder input x(local_i) items to match F(current_item) layout.
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            @parameter
            if base == length:
                copy_from = idx  # do a DFT on the inputs
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            @parameter
            if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return to_Co(x.load[2](Int(copy_from), 0).cast[out_dtype]())
        else:
            comptime step = Sc(i) * offset
            return to_Co(output.load[2](Int(n + step), 0))

    var x_0 = _get_x[0]()

    @parameter
    for j in range(UInt(1), base):
        var x_j = _get_x[j]()
        var x_j = Co(1, 0)

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: Co

                @parameter
                if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.load[2](Int(i), 0))

                x_out.store(Int(i), 0, _twf_fma[base_phasor, j == 1](x_j, acc))
            continue

        var twf_index = Int(twf_offset + local_i * (base - 1) + (j - 1))

        var i0_j_twf_vec = twiddle_factors.load[2](twf_index, 0)
        var i0_j_twf = to_Co(i0_j_twf_vec)

        @parameter
        for i in range(base):
            comptime base_phasor = _base_phasor[i, j]()
            var twf: Co

            @parameter
            if base_phasor.re == 1:  # Co(1, 0)
                twf = i0_j_twf
            elif base_phasor.im == -1:  # Co(0, -1)
                twf = Co(i0_j_twf.im, -i0_j_twf.re)
            elif base_phasor.re == -1:  # Co(-1, 0)
                twf = -i0_j_twf
            elif base_phasor.im == 1:  # Co(0, 1)
                twf = Co(-i0_j_twf.im, i0_j_twf.re)
            else:
                twf = i0_j_twf * base_phasor

            var acc: Co

            @parameter
            if j == 1:
                acc = x_0
            else:
                acc = to_Co(x_out.load[2](Int(i), 0))

            x_out.store(Int(i), 0, to_CoV(twf.fma(x_j, acc)))

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        comptime factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        if base.is_power_of_two():
            x_out.ptr.store(x_out.load[Int(base * 2)](0, 0) * factor)
        else:
            for i in range(base):
                x_out.store(Int(i), 0, x_out.load[2](Int(i), 0) * factor)

    @parameter
    for i in range(base):
        comptime step = Sc(i * offset)
        output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))
