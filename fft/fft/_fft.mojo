from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from layout import Layout, LayoutTensor

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _mixed_radix_digit_reverse,
    _unit_phasor_fma,
    _get_twiddle_factor,
    _get_twiddle_factors_inline,
    _unit_phasor_mul,
)

# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


@always_inline
def _radix_n_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutOrigin,
    out_address_space: AddressSpace,
    in_dtype: DType,
    in_layout: Layout,
    in_origin: ImmutOrigin,
    in_address_space: AddressSpace,
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
    ordered_bases: List[UInt],
    inline_twfs: Bool,
    runtime_twfs: Bool,
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
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * next_offset

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    @always_inline
    def to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    def to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @always_inline
    @parameter
    def _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=base
        ](j)
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

    @always_inline
    @parameter
    def _get_x[i: UInt]() -> Co:
        comptime if processed == 1:
            # Reorder input x(local_i) items to match F(current_item) layout.
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            comptime if base == length:
                copy_from = idx  # do a DFT on the inputs
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            comptime if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return to_Co(x.load[2](Int(copy_from), 0).cast[out_dtype]())
        else:
            comptime step = Sc(i) * offset
            return to_Co(output.load[2](Int(n + step), 0))

    var x_0 = _get_x[0]()

    comptime for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        comptime if processed == 1:
            comptime for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.load[CoV.size](Int(i), 0))

                comptime if do_rfft:
                    var res = _unit_phasor_fma[base_phasor, j == 1](x_j.re, acc)
                    x_out.store(Int(i), 0, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[base_phasor](x_j, acc)
                    x_out.store(Int(i), 0, to_CoV(res))
            continue

        var twf_index = Sc(j) * (Sc(local_i) % offset) * ratio
        var i0_j_twf: Co

        comptime if inline_twfs:
            comptime twfs = _get_twiddle_factors_inline[
                length, out_dtype, inverse
            ]()
            ref twfs_runtime = global_constant[twfs]()
            i0_j_twf = twfs_runtime[twf_index]
        elif runtime_twfs:
            i0_j_twf = _get_twiddle_factor[
                out_dtype, inverse=inverse, N=Sc(length)
            ](twf_index)
        else:
            i0_j_twf = to_Co(twiddle_factors.load[2](Int(twf_index), 0))

        comptime for i in range(base):
            var twf = _unit_phasor_mul[_base_phasor[i, j]()](i0_j_twf)

            var acc: Co

            comptime if j == 1:
                acc = x_0
            else:
                acc = to_Co(x_out.load[CoV.size](Int(i), 0))

            x_out.store(Int(i), 0, to_CoV(twf.fma(x_j, acc)))

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()

        comptime if UInt64(base).is_power_of_two():
            x_out.ptr.store(x_out.ptr.load[Int(base) * CoV.size]() * `1 / N`)
        else:
            comptime for i in range(base):
                var res = x_out.load[CoV.size](Int(i), 0) * `1 / N`
                x_out.store(Int(i), 0, res)

    comptime for i in range(base):
        comptime step = Sc(i) * offset
        output.store(Int(n + step), 0, x_out.load[CoV.size](Int(i), 0))


@always_inline
def _radix_n_fft_kernel_spread[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutOrigin,
    out_address_space: AddressSpace,
    in_dtype: DType,
    in_layout: Layout,
    in_origin: ImmutOrigin,
    in_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    ordered_bases: List[UInt],
    inline_twfs: Bool,
    runtime_twfs: Bool,
    run_inplace: Bool,
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
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    # cooley-tukey fft indexing
    var cooley_tukey_n = (Sc(local_i) // next_offset) * next_offset + (
        Sc(local_i) % offset
    )
    # stockham fft indexing
    var stockham_n = (Sc(local_i) // next_offset) * offset + (
        Sc(local_i) % next_offset
    ) % offset

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    @always_inline
    def to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    def to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @always_inline
    @parameter
    def _get_x[i: UInt]() -> Co:
        comptime if run_inplace:
            comptime step = Sc(i) * offset
            return to_Co(output.load[2](Int(cooley_tukey_n + step), 0))
        else:
            comptime step = Sc(i) * (Sc(length) // Sc(base))
            var src_idx = Int(stockham_n + step)

            comptime if processed == 1 and do_rfft:
                return Co(x.load[1](src_idx, 0).cast[out_dtype](), 0)
            else:
                return to_Co(x.load[2](src_idx, 0).cast[out_dtype]())

    var x_out = Co(0, 0)

    comptime for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        comptime max_twf_idx = (base - 1) * (
            (length - 1) % UInt(next_offset)
        ) * UInt(ratio)
        comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
        var twf_index = (
            Sc_twf(j) * Sc_twf(Sc(local_i) % next_offset) * Sc_twf(ratio)
        ) % Sc_twf(length)

        comptime if inline_twfs:
            comptime twfs = _get_twiddle_factors_inline[
                length, out_dtype, inverse
            ]()
            ref twfs_runtime = global_constant[twfs]()
            twf = twfs_runtime.unsafe_get(twf_index)
        elif runtime_twfs:
            twf = _get_twiddle_factor[out_dtype, inverse=inverse, N=Sc(length)](
                twf_index
            )
        else:
            twf = to_Co(twiddle_factors.load[2](Int(twf_index), 0))

        var acc: Co

        comptime if j == 1:
            acc = _get_x[0]()
        else:
            acc = x_out

        x_out = twf.fma(x_j, acc)

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()
        x_out *= `1 / N`

    output.store(Int(local_i), 0, to_CoV(x_out))
