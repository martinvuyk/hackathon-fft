from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from layout import TileTensor

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
def to_Co(v: SIMD[_, 2]) -> ComplexScalar[v.dtype]:
    return UnsafePointer(to=v).bitcast[ComplexScalar[v.dtype]]()[]


@always_inline
def to_CoV(c: ComplexScalar) -> SIMD[c.dtype, 2]:
    return UnsafePointer(to=c).bitcast[SIMD[c.dtype, 2]]()[]


@always_inline
def _get_x[
    i: UInt,
    out_dtype: DType,
    length: UInt,
    base: UInt,
    processed: UInt,
    do_rfft: Bool,
    ordered_bases: List[UInt],
    run_inplace: Bool = False,
](
    output: TileTensor[out_dtype, ...],
    x: TileTensor,
    n: Scalar,
    local_i: UInt,
) -> ComplexScalar[out_dtype]:
    comptime Sc = type_of(n)
    comptime if not run_inplace:
        comptime step = Sc(i) * (Sc(length) // Sc(base))
        var src_idx = Int(n + step)

        comptime if processed == 1 and do_rfft:
            return {x.raw_load[1](src_idx).cast[out_dtype](), 0}
        else:
            return to_Co(x.raw_load[2](src_idx * 2).cast[out_dtype]())
    elif processed == 1:
        # Reorder input x(local_i) items to match F(current_item) layout.
        var idx = Sc(local_i) * Sc(base) + Sc(i)

        var copy_from: Sc

        comptime if base == length:
            copy_from = idx  # do a DFT on the inputs
        else:
            copy_from = _mixed_radix_digit_reverse[length, ordered_bases](idx)

        comptime if do_rfft:
            return {x.raw_load[1](Int(copy_from)).cast[out_dtype](), 0}
        else:
            return to_Co(x.raw_load[2](Int(copy_from) * 2).cast[out_dtype]())
    else:
        comptime step = Sc(i * processed)
        return to_Co(output.raw_load[2](Int(n + step) * 2))


@always_inline
def _radix_n_fft_kernel_butterfly[
    out_dtype: DType,
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
    phase: Optional[UInt] = None,
](
    output: TileTensor[mut=True, out_dtype, _, _, ...],
    x: TileTensor[_, _, _, ...],
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, out_dtype, _, _, ...],
    x_out: TileTensor[mut=True, out_dtype, _, _, ...],
):
    """A generic Butterfly algorithm. It has most of the generalizable radix
    optimizations. Can run inplace by reordering the input (Cooley Tukey) or
    out of place (Stockham)."""
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    var k = Sc(phase.or_else(UInt(Sc(local_i) % offset)))
    var n = k + (Sc(local_i) // offset) * next_offset

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    @always_inline
    @parameter
    def _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=base
        ](j)
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

    comptime get[i: UInt] = _get_x[
        i,
        out_dtype,
        length,
        base,
        processed,
        do_rfft,
        ordered_bases,
        run_inplace,
    ]
    var indexing_n = n if run_inplace else Sc(local_i)
    var x_0 = get[0](output, x, indexing_n, local_i)

    comptime for j in range(UInt(1), base):
        var x_j = get[j](output, x, indexing_n, local_i)

        comptime if processed == 1:
            comptime for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                comptime if do_rfft:
                    var res = _unit_phasor_fma[base_phasor, j == 1](x_j.re, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[base_phasor](x_j, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
            continue

        comptime i0_j_twf_comptime = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=length
        ](j * UInt(ratio) * phase.or_else(0))
        var twf_index = Sc(j) * k * ratio
        var i0_j_twf: Co
        comptime if phase:
            i0_j_twf = Co(0, 0)
        else:
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
                i0_j_twf = to_Co(
                    twiddle_factors.raw_load[2](Int(twf_index) * 2)
                )

        var x_j_i0: Co
        comptime if phase:
            x_j_i0 = Co(0, 0)
        else:
            x_j_i0 = i0_j_twf * x_j

        comptime if base % 2 == 0:
            comptime for i in range(base // 2):
                var acc_top: Co
                var acc_bot: Co

                comptime complement = i + base // 2
                comptime if j == 1:
                    acc_top = x_0
                    acc_bot = x_0
                else:
                    acc_top = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))
                    acc_bot = to_Co(
                        x_out.raw_load[CoV.size](Int(complement) * 2)
                    )

                var term: Co
                comptime if phase:
                    comptime twf = _unit_phasor_mul[_base_phasor[i, j]()](
                        i0_j_twf_comptime
                    )
                    term = _unit_phasor_mul[twf](x_j)
                else:
                    term = _unit_phasor_mul[_base_phasor[i, j]()](x_j_i0)
                x_out.raw_store(Int(i) * 2, to_CoV(acc_top + term))
                comptime if j % 2 == 0:
                    x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot + term))
                else:
                    x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot - term))
        else:
            comptime for i in range(base):
                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                var res: Co
                comptime if phase:
                    comptime twf = _unit_phasor_mul[_base_phasor[i, j]()](
                        i0_j_twf_comptime
                    )
                    res = _unit_phasor_fma[twf](x_j, acc)
                else:
                    res = _unit_phasor_fma[_base_phasor[i, j]()](x_j_i0, acc)
                x_out.raw_store(Int(i) * 2, to_CoV(res))

    comptime base_is_pow2 = Bool(UInt64(base).is_power_of_two())

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()

        comptime if base_is_pow2:
            x_out.ptr.store(x_out.ptr.load[Int(base) * CoV.size]() * `1 / N`)
        else:
            comptime for i in range(base):
                var res = x_out.raw_load[CoV.size](Int(i) * 2) * `1 / N`
                x_out.raw_store(Int(i) * 2, res)

    comptime if run_inplace and base_is_pow2 and processed == 1:
        output.raw_store(Int(n) * 2, x_out.raw_load[Int(base) * 2](0))
    else:
        comptime for i in range(base):
            comptime step = Sc(i) * offset
            output.raw_store(Int(n + step) * 2, x_out.raw_load[CoV.size](Int(i) * 2))


@always_inline
def _radix_n_fft_kernel_butterfly_comptime[
    out_dtype: DType,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    ordered_bases: List[UInt],
    run_inplace: Bool,
    local_i: UInt,
](
    output: TileTensor[mut=True, out_dtype, _, _, ...],
    x: TileTensor[_, _, _, ...],
    x_out: TileTensor[mut=True, out_dtype, _, _, ...],
):
    """A generic Butterfly algorithm. It has most of the generalizable radix
    optimizations. Can run inplace by reordering the input (Cooley Tukey) or
    out of place (Stockham)."""
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    comptime n = (
        Sc(local_i) % offset + (Sc(local_i) // offset) * next_offset
    ) if run_inplace else Sc(local_i)

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    @always_inline
    @parameter
    def _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=base
        ](j)
        res = {1, 0}

        for _ in range(i):
            res *= base_twf

    comptime get[i: UInt] = _get_x[
        i,
        out_dtype,
        length,
        base,
        processed,
        do_rfft,
        ordered_bases,
        run_inplace,
    ]
    var x_0 = get[0](output, x, n, local_i)

    comptime for j in range(UInt(1), base):
        var x_j = get[j](output, x, n, local_i)

        comptime twf_index = Sc(j) * (Sc(local_i) % offset) * ratio
        comptime i0_j_twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=Sc(length)
        ](twf_index)

        comptime if base % 2 == 0:
            comptime for i in range(base // 2):
                comptime twf = _unit_phasor_mul[_base_phasor[i, j]()](i0_j_twf)
                var acc_top: Co
                var acc_bot: Co

                comptime complement = i + base // 2
                comptime if j == 1:
                    acc_top = x_0
                    acc_bot = x_0
                else:
                    acc_top = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))
                    acc_bot = to_Co(
                        x_out.raw_load[CoV.size](Int(complement) * 2)
                    )

                var term = _unit_phasor_mul[twf](x_j)
                x_out.raw_store(Int(i) * 2, to_CoV(acc_top + term))
                comptime if j % 2 == 0:
                    x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot + term))
                else:
                    x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot - term))
        else:
            comptime for i in range(base):
                comptime twf = _unit_phasor_mul[_base_phasor[i, j]()](i0_j_twf)

                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                comptime if processed == 1 and do_rfft:
                    var res = _unit_phasor_fma[twf, j == 1](x_j.re, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[twf](x_j, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))

    comptime base_is_pow2 = Bool(UInt64(base).is_power_of_two())

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()

        comptime if base_is_pow2:
            x_out.ptr.store(x_out.ptr.load[Int(base) * CoV.size]() * `1 / N`)
        else:
            comptime for i in range(base):
                var res = x_out.raw_load[CoV.size](Int(i) * 2) * `1 / N`
                x_out.raw_store(Int(i) * 2, res)

    comptime if run_inplace and base_is_pow2 and processed == 1:
        output.raw_store(Int(n) * 2, x_out.raw_load[Int(base) * 2](0))
    else:
        comptime out_n = n if run_inplace else (
            (Sc(local_i) // offset) * next_offset + (Sc(local_i) % offset)
        )
        comptime for i in range(base):
            comptime step = Sc(i) * offset
            output.raw_store(Int(out_n + step) * 2, x_out.raw_load[CoV.size](Int(i) * 2))


@always_inline
def _radix_n_fft_kernel_elem_per_thread[
    out_dtype: DType,
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
    output: TileTensor[mut=True, out_dtype, _, _, ...],
    x: TileTensor[_, _, _, ...],
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, out_dtype, _, _, ...],
):
    """A generic Stockham algorithm. It has most of the generalizable radix
    optimizations. Can't run inplace, but has better memory access patterns."""
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    var n = (Sc(local_i) // next_offset) * offset + (
        Sc(local_i) % next_offset
    ) % offset

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    comptime get[i: UInt] = _get_x[
        i,
        out_dtype,
        length,
        base,
        processed,
        do_rfft,
        ordered_bases,
    ]
    var x_out = get[0](output, x, n, local_i)

    comptime for j in range(UInt(1), base):
        var x_j = get[j](output, x, n, local_i)

        var twf_index: Sc
        comptime max_twf_idx = (base - 1) * UInt(next_offset)
        comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
        var base_idx = Sc_twf(j) * Sc_twf(Sc(local_i) % next_offset)
        twf_index = Sc(base_idx % Sc_twf(next_offset)) * ratio

        var twf: Co

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
            twf = to_Co(twiddle_factors.raw_load[2](Int(twf_index) * 2))

        x_out = twf.fma(x_j, x_out)

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()
        x_out *= `1 / N`

    output.raw_store(Int(local_i) * 2, to_CoV(x_out))


@always_inline
def _radix_n_fft_kernel_elem_per_thread_comptime[
    out_dtype: DType,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    ordered_bases: List[UInt],
    local_i: UInt,
](
    output: TileTensor[mut=True, out_dtype, _, _, ...],
    x: TileTensor[_, _, _, ...],
):
    """A generic Stockham algorithm. It has most of the generalizable radix
    optimizations. Can't run inplace, but has better memory access patterns."""
    comptime assert length >= base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()

    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset

    comptime n = (Sc(local_i) // next_offset) * offset + (
        Sc(local_i) % next_offset
    ) % offset

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    comptime get[i: UInt] = _get_x[
        i,
        out_dtype,
        length,
        base,
        processed,
        do_rfft,
        ordered_bases,
    ]
    var x_out = get[0](output, x, n, local_i)

    comptime for j in range(UInt(1), base):
        var x_j = get[j](output, x, n, local_i)

        comptime max_twf_idx = (base - 1) * UInt(next_offset)
        comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
        comptime base_idx = Sc_twf(j) * Sc_twf(Sc(local_i) % next_offset)
        comptime twf_index = Sc(base_idx % Sc_twf(next_offset)) * ratio
        comptime twf = _get_twiddle_factor[
            out_dtype, inverse=inverse, N=length
        ](twf_index)

        comptime if processed == 1 and do_rfft:
            x_out = _unit_phasor_fma[twf, accum_is_real=j == 1](x_j.re, x_out)
        else:
            x_out = _unit_phasor_fma[twf](x_j, x_out)

    comptime if inverse and processed * base == length:  # last ifft stage
        comptime `1 / N` = (1.0 / Float64(length)).cast[out_dtype]()
        x_out *= `1 / N`

    output.raw_store(Int(local_i) * 2, to_CoV(x_out))
