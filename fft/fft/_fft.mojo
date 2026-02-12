from algorithm import parallelize, vectorize
from builtin.globals import global_constant
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from gpu.host.info import Vendor, is_cpu
from math import sqrt
from layout import Layout, LayoutTensor, IntTuple
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import simd_width_of, size_of

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
    ordered_bases: List[UInt],
    runtime_twfs: Bool,
    inline_twfs: Bool,
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
    comptime Sc = Scalar[_get_dtype[length * base]()]
    comptime offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=base
        ](j)
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

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

                @parameter
                if do_rfft:
                    var res = _unit_phasor_fma[base_phasor, j == 1](x_j.re, acc)
                    x_out.store(Int(i), 0, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[base_phasor](x_j, acc)
                    x_out.store(Int(i), 0, to_CoV(res))
            continue

        comptime next_offset = offset * Sc(base)
        comptime ratio = Sc(length) // next_offset
        var twf_index = ((Sc(j) * n) % next_offset) * ratio
        var i0_j_twf: Co

        @parameter
        if runtime_twfs:
            i0_j_twf = _get_twiddle_factor[
                out_dtype, inverse=inverse, N=length
            ](UInt(twf_index))
        elif inline_twfs:
            comptime twfs = _get_twiddle_factors_inline[
                length, out_dtype, inverse
            ]()
            ref twfs_runtime = global_constant[twfs]()
            i0_j_twf = twfs_runtime[twf_index]
        else:
            i0_j_twf = to_Co(twiddle_factors.load[2](Int(twf_index), 0))

        @parameter
        for i in range(base):
            var twf = _unit_phasor_mul[_base_phasor[i, j]()](i0_j_twf)

            var acc: Co

            @parameter
            if j == 1:
                acc = x_0
            else:
                acc = to_Co(x_out.load[2](Int(i), 0))

            x_out.store(Int(i), 0, to_CoV(twf.fma(x_j, acc)))

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()

        @parameter
        if UInt64(base).is_power_of_two():
            x_out.ptr.store(x_out.load[Int(base) * 2](0, 0) * `1 / N`)
        else:

            @parameter
            for i in range(base):
                x_out.store(Int(i), 0, x_out.load[2](Int(i), 0) * `1 / N`)

    @parameter
    for i in range(base):
        comptime step = Sc(i) * offset
        output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))
