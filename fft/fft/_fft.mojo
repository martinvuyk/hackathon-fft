from std.bit import count_trailing_zeros
from std.builtin.globals import global_constant
from std.collections import Array
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
    _reg_dft_radix,
)


@fieldwise_init
struct _FFTKernelExecConfig[
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    ordered_bases: List[UInt],
    inline_twfs: Bool,
    runtime_twfs: Bool,
    run_inplace: Bool,
    in_complex_stride: Int = 2,
    out_complex_stride: Int = 2,
](TrivialRegisterPassable):
    """Comptime execution parameters and shared Stockham/butterfly indexing.

    `in_complex_stride` / `out_complex_stride` are scalar elements between
    consecutive complexes in the read/write tiles (2 = packed; GPU shared may
    use a padded stride).
    """

    comptime Sc = Scalar[_get_dtype[Self.length]()]
    comptime offset = Self.Sc(Self.processed)
    comptime next_offset = Self.offset * Self.Sc(Self.base)
    comptime ratio = Self.Sc(Self.length) // Self.next_offset
    comptime base_is_pow2 = Bool(Self.base.is_power_of_two())
    comptime is_last_ifft = Self.inverse and (
        Self.processed * Self.base == Self.length
    )
    comptime `1 / N`[dtype: DType] = (
        (1.0 / Float64(Self.length)).cast[dtype]()
    )

    @staticmethod
    @always_inline
    def butterfly_n(local_i: UInt, phase: Optional[UInt] = None) -> Self.Sc:
        """Cooley-Tukey / butterfly output index for `local_i`."""
        var k = Self.Sc(phase.or_else(UInt(Self.Sc(local_i) % Self.offset)))
        return k + (Self.Sc(local_i) // Self.offset) * Self.next_offset

    @staticmethod
    @always_inline
    def stockham_n(local_i: UInt) -> Self.Sc:
        """Stockham read index for `local_i`.

        `(local_i % next_offset) % offset == local_i % offset` because
        `next_offset = offset * base`.
        """
        return (Self.Sc(local_i) // Self.next_offset) * Self.offset + (
            Self.Sc(local_i) % Self.offset
        )


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


@always_inline
def _assert_radix_kernel_cfg[out_dtype: DType, cfg: _FFTKernelExecConfig]():
    comptime assert cfg.length >= cfg.base, "length must be >= base"
    comptime assert out_dtype.is_floating_point()


@always_inline
def to_Co(v: SIMD[_, 2]) -> ComplexScalar[v.dtype]:
    return UnsafePointer(to=v).unsafe_bitcast[ComplexScalar[v.dtype]]()[]


@always_inline
def to_CoV(c: ComplexScalar) -> SIMD[c.dtype, 2]:
    return UnsafePointer(to=c).unsafe_bitcast[SIMD[c.dtype, 2]]()[]


@always_inline
def _get_x[
    i: UInt, out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[out_dtype, ...],
    x: TileTensor,
    n: Scalar,
    local_i: UInt,
) -> ComplexScalar[out_dtype]:
    comptime Sc = type_of(n)
    comptime if not cfg.run_inplace:
        comptime step = Sc(i) * (Sc(cfg.length) // Sc(cfg.base))
        var src_idx = Int(n + step)

        comptime if cfg.processed == 1 and cfg.do_rfft:
            return {x.raw_load[1](src_idx).cast[out_dtype](), 0}
        else:
            return to_Co(
                x.raw_load[2](
                    src_idx * cfg.in_complex_stride
                ).cast[out_dtype]()
            )
    elif cfg.processed == 1:
        # Reorder input x(local_i) items to match F(current_item) layout.
        var idx = Sc(local_i) * Sc(cfg.base) + Sc(i)

        var copy_from: Sc

        comptime if cfg.base == cfg.length:
            copy_from = idx  # do a DFT on the inputs
        else:
            copy_from = _mixed_radix_digit_reverse[
                cfg.length, cfg.ordered_bases
            ](idx)

        comptime if cfg.do_rfft:
            return {x.raw_load[1](Int(copy_from)).cast[out_dtype](), 0}
        else:
            return to_Co(
                x.raw_load[2](
                    Int(copy_from) * cfg.in_complex_stride
                ).cast[out_dtype]()
            )
    else:
        comptime step = Sc(i * cfg.processed)
        return to_Co(
            output.raw_load[2](
                Int(n + step) * cfg.out_complex_stride
            )
        )


@always_inline
def _base_phasor[
    out_dtype: DType, cfg: _FFTKernelExecConfig, i: UInt, j: UInt
]() -> ComplexScalar[out_dtype]:
    comptime base_twf = _get_twiddle_factor[
        out_dtype, inverse=cfg.inverse, N=cfg.base
    ](j)
    var res = ComplexScalar[out_dtype](1, 0)
    for _ in range(i):
        res *= base_twf
    return res


@always_inline
def _load_twiddle[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    twf_index: Scalar,
    twiddle_factors: TileTensor[mut=False, ...],
) -> ComplexScalar[out_dtype]:
    comptime if cfg.inline_twfs:
        comptime twfs = _get_twiddle_factors_inline[
            cfg.length, out_dtype, cfg.inverse
        ]()
        ref twfs_runtime = global_constant[twfs]()
        return twfs_runtime.unsafe_get(twf_index)
    elif cfg.runtime_twfs:
        return _get_twiddle_factor[
            out_dtype, inverse=cfg.inverse, N=cfg.Sc(cfg.length)
        ](twf_index)
    else:
        return to_Co(
            twiddle_factors.raw_load[2](Int(twf_index) * 2).cast[out_dtype]()
        )


@always_inline
def _scale_ifft_tile[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](x_out: TileTensor[mut=True, out_dtype, ...]):
    comptime if cfg.is_last_ifft:
        comptime CoV = SIMD[out_dtype, 2]

        comptime if cfg.base_is_pow2:
            x_out.ptr.store(
                x_out.ptr.load[Int(cfg.base) * CoV.size]()
                * cfg.`1 / N`[out_dtype]
            )
        else:
            comptime for i in range(cfg.base):
                var res = (
                    x_out.raw_load[CoV.size](Int(i) * 2)
                    * cfg.`1 / N`[out_dtype]
                )
                x_out.raw_store(Int(i) * 2, res)


@always_inline
def _store_butterfly_outputs[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=True, out_dtype, ...],
    x_out: TileTensor[mut=True, out_dtype, ...],
    n: cfg.Sc,
):
    """Scatter butterfly tile `x_out` into `output` at Stockham/CT index `n`."""
    comptime CoV = SIMD[out_dtype, 2]
    comptime if cfg.run_inplace and cfg.base_is_pow2 and cfg.processed == 1:
        output.raw_store(Int(n) * 2, x_out.raw_load[Int(cfg.base) * 2](0))
    else:
        comptime for i in range(cfg.base):
            comptime step = cfg.Sc(i) * cfg.offset
            output.raw_store(
                Int(n + step) * cfg.out_complex_stride,
                x_out.raw_load[CoV.size](Int(i) * 2),
            )


@always_inline
def _even_radix_accumulate[
    out_dtype: DType, cfg: _FFTKernelExecConfig, i: UInt, j: UInt
](
    x_out: TileTensor[mut=True, out_dtype, ...],
    x_0: ComplexScalar[out_dtype],
    term: ComplexScalar[out_dtype],
):
    """Even-base butterfly: add/sub `term` into complementary output slots."""
    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]
    comptime complement = i + cfg.base // 2
    var acc_top: Co
    var acc_bot: Co
    comptime if j == 1:
        acc_top = x_0
        acc_bot = x_0
    else:
        acc_top = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))
        acc_bot = to_Co(x_out.raw_load[CoV.size](Int(complement) * 2))
    x_out.raw_store(Int(i) * 2, to_CoV(acc_top + term))
    comptime if j % 2 == 0:
        x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot + term))
    else:
        x_out.raw_store(Int(complement) * 2, to_CoV(acc_bot - term))


@always_inline
def _radix_n_fft_kernel_butterfly[
    out_dtype: DType, cfg: _FFTKernelExecConfig, phase: Optional[UInt] = None
](
    output: TileTensor[mut=True, out_dtype, ...],
    x: TileTensor,
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
    x_out: TileTensor[mut=True, out_dtype, ...],
):
    """A generic Butterfly algorithm. It has most of the generalizable radix
    optimizations. Can run inplace by reordering the input (Cooley Tukey) or
    out of place (Stockham)."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()

    var n = cfg.butterfly_n(local_i, phase)
    var k = cfg.Sc(phase.or_else(UInt(cfg.Sc(local_i) % cfg.offset)))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    var indexing_n = n if cfg.run_inplace else cfg.Sc(local_i)
    var x_0 = _get_x[0, out_dtype, cfg](output, x, indexing_n, local_i)

    comptime for j in range(UInt(1), cfg.base):
        var x_j = _get_x[j, out_dtype, cfg](output, x, indexing_n, local_i)

        comptime if cfg.processed == 1:
            comptime for i in range(cfg.base):
                comptime base_phasor = _base_phasor[out_dtype, cfg, i, j]()
                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                comptime if cfg.do_rfft:
                    var res = _unit_phasor_fma[base_phasor, j == 1](x_j.re, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[base_phasor](x_j, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
            continue

        comptime i0_j_twf_comptime = _get_twiddle_factor[
            out_dtype, inverse=cfg.inverse, N=cfg.length
        ](j * UInt(cfg.ratio) * phase.or_else(0))
        var twf_index = cfg.Sc(j) * k * cfg.ratio
        var i0_j_twf: Co
        comptime if phase:
            i0_j_twf = Co(0, 0)
        else:
            i0_j_twf = _load_twiddle[out_dtype, cfg](twf_index, twiddle_factors)

        var x_j_i0: Co
        comptime if phase:
            x_j_i0 = Co(0, 0)
        else:
            x_j_i0 = i0_j_twf * x_j

        comptime if cfg.base % 2 == 0:
            comptime for i in range(cfg.base // 2):
                comptime base_phasor = _base_phasor[out_dtype, cfg, i, j]()
                var term: Co
                comptime if phase:
                    comptime twf = _unit_phasor_mul[base_phasor](
                        i0_j_twf_comptime
                    )
                    term = _unit_phasor_mul[twf](x_j)
                else:
                    term = _unit_phasor_mul[base_phasor](x_j_i0)
                _even_radix_accumulate[out_dtype, cfg, i, j](x_out, x_0, term)
        else:
            comptime for i in range(cfg.base):
                comptime base_phasor = _base_phasor[out_dtype, cfg, i, j]()
                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                var res: Co
                comptime if phase:
                    comptime twf = _unit_phasor_mul[base_phasor](
                        i0_j_twf_comptime
                    )
                    res = _unit_phasor_fma[twf](x_j, acc)
                else:
                    res = _unit_phasor_fma[base_phasor](x_j_i0, acc)
                x_out.raw_store(Int(i) * 2, to_CoV(res))

    _scale_ifft_tile[out_dtype, cfg](x_out)
    _store_butterfly_outputs[out_dtype, cfg](output, x_out, n)


@always_inline
def _radix_n_fft_kernel_butterfly_comptime[
    out_dtype: DType, cfg: _FFTKernelExecConfig, local_i: UInt
](
    output: TileTensor[mut=True, out_dtype, ...],
    x: TileTensor,
    x_out: TileTensor[mut=True, out_dtype, ...],
):
    """A generic Butterfly algorithm. It has most of the generalizable radix
    optimizations. Can run inplace by reordering the input (Cooley Tukey) or
    out of place (Stockham)."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()

    comptime n = (
        cfg.butterfly_n(local_i) if cfg.run_inplace else cfg.Sc(local_i)
    )

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    var x_0 = _get_x[0, out_dtype, cfg](output, x, n, local_i)

    comptime for j in range(UInt(1), cfg.base):
        var x_j = _get_x[j, out_dtype, cfg](output, x, n, local_i)

        comptime twf_index = (
            cfg.Sc(j) * (cfg.Sc(local_i) % cfg.offset) * cfg.ratio
        )
        comptime i0_j_twf = _get_twiddle_factor[
            out_dtype, inverse=cfg.inverse, N=cfg.Sc(cfg.length)
        ](twf_index)

        comptime if cfg.base % 2 == 0:
            comptime for i in range(cfg.base // 2):
                comptime base_phasor = _base_phasor[out_dtype, cfg, i, j]()
                comptime twf = _unit_phasor_mul[base_phasor](i0_j_twf)
                var term = _unit_phasor_mul[twf](x_j)
                _even_radix_accumulate[out_dtype, cfg, i, j](x_out, x_0, term)
        else:
            comptime for i in range(cfg.base):
                comptime base_phasor = _base_phasor[out_dtype, cfg, i, j]()
                comptime twf = _unit_phasor_mul[base_phasor](i0_j_twf)

                var acc: Co

                comptime if j == 1:
                    acc = x_0
                else:
                    acc = to_Co(x_out.raw_load[CoV.size](Int(i) * 2))

                comptime if cfg.processed == 1 and cfg.do_rfft:
                    var res = _unit_phasor_fma[twf, j == 1](x_j.re, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))
                else:
                    var res = _unit_phasor_fma[twf](x_j, acc)
                    x_out.raw_store(Int(i) * 2, to_CoV(res))

    _scale_ifft_tile[out_dtype, cfg](x_out)
    comptime out_n = n if cfg.run_inplace else cfg.butterfly_n(local_i)
    _store_butterfly_outputs[out_dtype, cfg](output, x_out, out_n)


@always_inline
def _radix_n_fft_kernel_elem_to_reg[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[out_dtype, ...],
    x: TileTensor,
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
) -> SIMD[out_dtype, 2]:
    """Stockham per-element stage into a register (no store)."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()

    var n = cfg.stockham_n(local_i)

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    var x_out = _get_x[0, out_dtype, cfg](output, x, n, local_i)

    comptime for j in range(UInt(1), cfg.base):
        var x_j = _get_x[j, out_dtype, cfg](output, x, n, local_i)

        comptime if j == 1:
            var rem = cfg.Sc(local_i) % cfg.next_offset
            var twf = _load_twiddle[out_dtype, cfg](
                rem * cfg.ratio, twiddle_factors
            )
            x_out = twf.fma(x_j, x_out)
        else:
            comptime max_twf_idx = (cfg.base - 1) * UInt(cfg.next_offset)
            comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
            var base_idx = Sc_twf(j) * Sc_twf(cfg.Sc(local_i) % cfg.next_offset)
            var twf_index = (
                cfg.Sc(base_idx % Sc_twf(cfg.next_offset)) * cfg.ratio
            )
            var twf = _load_twiddle[out_dtype, cfg](twf_index, twiddle_factors)
            x_out = twf.fma(x_j, x_out)

    comptime if cfg.is_last_ifft:
        x_out *= cfg.`1 / N`[out_dtype]

    return to_CoV(x_out)


@always_inline
def _stockham_elem_twiddle[
    out_dtype: DType, cfg: _FFTKernelExecConfig, j: UInt
](
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
) -> ComplexScalar[out_dtype]:
    """Twiddle for Stockham output `local_i` and input arm `j`."""
    comptime if j == 1:
        # (1 * rem) % next == rem when rem < next.
        var rem = cfg.Sc(local_i) % cfg.next_offset
        return _load_twiddle[out_dtype, cfg](rem * cfg.ratio, twiddle_factors)
    comptime max_twf_idx = (cfg.base - 1) * UInt(cfg.next_offset)
    comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
    var base_idx = Sc_twf(j) * Sc_twf(cfg.Sc(local_i) % cfg.next_offset)
    var twf_index = cfg.Sc(base_idx % Sc_twf(cfg.next_offset)) * cfg.ratio
    return _load_twiddle[out_dtype, cfg](twf_index, twiddle_factors)


@always_inline
def _bit_reverse[bits: Int](i: Int) -> Int:
    var x = i
    var y = 0
    comptime for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


@always_inline
def _fft_len_r[
    dtype: DType, R: Int, inverse: Bool
](mut x: Array[ComplexScalar[dtype], R]):
    """Length-`R` in-register DFT.

    Power-of-two R: in-place Cooley–Tukey with comptime twiddles.
    Small primes (3, 5, 7): direct DFT with comptime twiddles.
    General R: naive O(R²) DFT with comptime twiddles.
    """
    comptime Co = ComplexScalar[dtype]
    comptime if R <= 1:
        return

    comptime if R == 2:
        var t = x[1]
        var u = x[0]
        x[0] = u + t
        x[1] = u - t
        return
    comptime if R == 4:
        # Closed radix-4 (DIT); sg flips for inverse.
        comptime sg = Scalar[dtype](-1.0) if inverse else Scalar[dtype](1.0)
        var a = x[0] + x[2]
        var b = x[1] + x[3]
        var c = x[0] - x[2]
        var d = x[1] - x[3]
        # (-i * sg) * d → re = sg * d.im, im = -sg * d.re
        var di_re = sg * d.im
        var di_im = -sg * d.re
        x[0] = a + b
        x[2] = a - b
        x[1] = Co(c.re + di_re, c.im + di_im)
        x[3] = Co(c.re - di_re, c.im - di_im)
        return
    comptime if R == 3:
        # Closed radix-3 (ω = e^{∓2πi/3}): y0=x0+u, y1/y2 = x0-u/2 ∓ i(√3/2)v.
        comptime half = Scalar[dtype](0.5)
        comptime tau = Scalar[dtype](0.8660254037844386)  # √3/2
        comptime sgn = Scalar[dtype](-1.0) if inverse else Scalar[dtype](1.0)
        var u = x[1] + x[2]
        var v = x[1] - x[2]
        var a = x[0] - u * half
        # (-i * sgn * tau) * v
        var b_re = sgn * tau * v.im
        var b_im = -sgn * tau * v.re
        x[0] = x[0] + u
        x[1] = Co(a.re + b_re, a.im + b_im)
        x[2] = Co(a.re - b_re, a.im - b_im)
        return
    comptime if R == 5:
        # Closed radix-5; flip sin signs for inverse.
        comptime c1 = Scalar[dtype](0.30901699437494745)  # cos(2π/5)
        comptime c2 = Scalar[dtype](-0.8090169943749473)  # cos(4π/5)
        comptime s1 = Scalar[dtype](0.9510565162951535)  # sin(2π/5)
        comptime s2 = Scalar[dtype](0.5877852522924731)  # sin(4π/5)
        comptime sg = Scalar[dtype](-1.0) if inverse else Scalar[dtype](1.0)
        var t0 = x[1] + x[4]
        var t1 = x[2] + x[3]
        var t2 = x[1] - x[4]
        var t3 = x[2] - x[3]
        var y0 = x[0] + t0 + t1
        var a = x[0] + t0 * c1 + t1 * c2
        var b = x[0] + t0 * c2 + t1 * c1
        # w = (-i * sg) * (s1*t2 ± s2*t3) → re = sg * s·im, im = -sg * s·re
        var w1_re = sg * (s1 * t2.im + s2 * t3.im)
        var w1_im = -sg * (s1 * t2.re + s2 * t3.re)
        var w2_re = sg * (s2 * t2.im - s1 * t3.im)
        var w2_im = -sg * (s2 * t2.re - s1 * t3.re)
        x[0] = y0
        x[1] = Co(a.re + w1_re, a.im + w1_im)
        x[4] = Co(a.re - w1_re, a.im - w1_im)
        x[2] = Co(b.re + w2_re, b.im + w2_im)
        x[3] = Co(b.re - w2_re, b.im - w2_im)
        return
    comptime if R == 6:
        # Even/odd radix-2 split into two length-3 DFTs (fewer temps than 2×3 Bailey).
        var e = Array[Co, 3](fill=Co(0, 0))
        var o = Array[Co, 3](fill=Co(0, 0))
        e[0] = x[0]
        e[1] = x[2]
        e[2] = x[4]
        o[0] = x[1]
        o[1] = x[3]
        o[2] = x[5]
        _fft_len_r[dtype, 3, inverse](e)
        _fft_len_r[dtype, 3, inverse](o)
        comptime w1 = _get_twiddle_factor[dtype, inverse=inverse, N = UInt(6)](UInt(1))
        comptime w2 = _get_twiddle_factor[dtype, inverse=inverse, N = UInt(6)](UInt(2))
        o[1] = w1 * o[1]
        o[2] = w2 * o[2]
        x[0] = e[0] + o[0]
        x[3] = e[0] - o[0]
        x[1] = e[1] + o[1]
        x[4] = e[1] - o[1]
        x[2] = e[2] + o[2]
        x[5] = e[2] - o[2]
        return
    comptime if R == 8:
        # Even/odd radix-2 split into two length-4 DFTs (fewer temps than CT-8).
        var e = Array[Co, 4](fill=Co(0, 0))
        var o = Array[Co, 4](fill=Co(0, 0))
        e[0] = x[0]
        e[1] = x[2]
        e[2] = x[4]
        e[3] = x[6]
        o[0] = x[1]
        o[1] = x[3]
        o[2] = x[5]
        o[3] = x[7]
        _fft_len_r[dtype, 4, inverse](e)
        _fft_len_r[dtype, 4, inverse](o)
        comptime w1 = _get_twiddle_factor[dtype, inverse=inverse, N = UInt(8)](UInt(1))
        comptime w2 = _get_twiddle_factor[dtype, inverse=inverse, N = UInt(8)](UInt(2))
        comptime w3 = _get_twiddle_factor[dtype, inverse=inverse, N = UInt(8)](UInt(3))
        o[1] = w1 * o[1]
        o[2] = w2 * o[2]
        o[3] = w3 * o[3]
        x[0] = e[0] + o[0]
        x[4] = e[0] - o[0]
        x[1] = e[1] + o[1]
        x[5] = e[1] - o[1]
        x[2] = e[2] + o[2]
        x[6] = e[2] - o[2]
        x[3] = e[3] + o[3]
        x[7] = e[3] - o[3]
        return
    comptime if R == 7:
        # Closed radix-7 via factored DFT (naive O(R²) is too heavy in-reg).
        # Coefficients: 2*cos(2πk/7), 2*sin(2πk/7) for k=1,2,3.
        comptime c1 = Scalar[dtype](0.6234898018587335)  # cos(2π/7)
        comptime c2 = Scalar[dtype](-0.2225209339563144)  # cos(4π/7)
        comptime c3 = Scalar[dtype](-0.9009688679024191)  # cos(6π/7)
        comptime s1 = Scalar[dtype](0.7818314824680298)  # sin(2π/7)
        comptime s2 = Scalar[dtype](0.9749279121818236)  # sin(4π/7)
        comptime s3 = Scalar[dtype](0.4338837391175581)  # sin(6π/7)
        comptime sg = Scalar[dtype](-1.0) if inverse else Scalar[dtype](1.0)
        var u1 = x[1] + x[6]
        var u2 = x[2] + x[5]
        var u3 = x[3] + x[4]
        var v1 = x[1] - x[6]
        var v2 = x[2] - x[5]
        var v3 = x[3] - x[4]
        var y0 = x[0] + u1 + u2 + u3
        var a1 = x[0] + u1 * c1 + u2 * c2 + u3 * c3
        var a2 = x[0] + u1 * c2 + u2 * c3 + u3 * c1
        var a3 = x[0] + u1 * c3 + u2 * c1 + u3 * c2
        # (-i*sg) * (s1*v1 + s2*v2 + s3*v3) etc. with cyclic sin signs.
        var w1_re = sg * (s1 * v1.im + s2 * v2.im + s3 * v3.im)
        var w1_im = -sg * (s1 * v1.re + s2 * v2.re + s3 * v3.re)
        var w2_re = sg * (s2 * v1.im - s3 * v2.im - s1 * v3.im)
        var w2_im = -sg * (s2 * v1.re - s3 * v2.re - s1 * v3.re)
        var w3_re = sg * (s3 * v1.im - s1 * v2.im + s2 * v3.im)
        var w3_im = -sg * (s3 * v1.re - s1 * v2.re + s2 * v3.re)
        x[0] = y0
        x[1] = Co(a1.re + w1_re, a1.im + w1_im)
        x[6] = Co(a1.re - w1_re, a1.im - w1_im)
        x[2] = Co(a2.re + w2_re, a2.im + w2_im)
        x[5] = Co(a2.re - w2_re, a2.im - w2_im)
        x[3] = Co(a3.re + w3_re, a3.im + w3_im)
        x[4] = Co(a3.re - w3_re, a3.im - w3_im)
        return
    comptime if R == 9:
        # 9 = 3×3 Bailey four-step (FFT along n1=3 first).
        var tmp = Array[Co, 9](fill=Co(0, 0))
        comptime for k in range(3):
            var col = Array[Co, 3](fill=Co(0, 0))
            comptime for j in range(3):
                col[j] = x[j * 3 + k]
            _fft_len_r[dtype, 3, inverse](col)
            comptime for j in range(3):
                comptime if j > 0 and k > 0:
                    var w = _get_twiddle_factor[
                        dtype, inverse=inverse, N = UInt(9)
                    ](UInt(j * k))
                    col[j] = w * col[j]
                tmp[k * 3 + j] = col[j]
        comptime for j in range(3):
            var col3 = Array[Co, 3](fill=Co(0, 0))
            comptime for k in range(3):
                col3[k] = tmp[k * 3 + j]
            _fft_len_r[dtype, 3, inverse](col3)
            comptime for k in range(3):
                x[k * 3 + j] = col3[k]
        return
    comptime if R == 10:
        # Even/odd radix-2 split into two length-5 DFTs.
        var e = Array[Co, 5](fill=Co(0, 0))
        var o = Array[Co, 5](fill=Co(0, 0))
        comptime for k in range(5):
            e[k] = x[2 * k]
            o[k] = x[2 * k + 1]
        _fft_len_r[dtype, 5, inverse](e)
        _fft_len_r[dtype, 5, inverse](o)
        comptime for k in range(1, 5):
            var w = _get_twiddle_factor[
                dtype, inverse=inverse, N = UInt(10)
            ](UInt(k))
            o[k] = w * o[k]
        comptime for k in range(5):
            x[k] = e[k] + o[k]
            x[k + 5] = e[k] - o[k]
        return
    comptime if R == 15:
        # 15 = 3×5 Bailey four-step (FFT along n1=3 first).
        var tmp = Array[Co, 15](fill=Co(0, 0))
        comptime for k in range(5):
            var col = Array[Co, 3](fill=Co(0, 0))
            comptime for j in range(3):
                col[j] = x[j * 5 + k]
            _fft_len_r[dtype, 3, inverse](col)
            comptime for j in range(3):
                comptime if j > 0 and k > 0:
                    var w = _get_twiddle_factor[
                        dtype, inverse=inverse, N = UInt(15)
                    ](UInt(j * k))
                    col[j] = w * col[j]
                tmp[k * 3 + j] = col[j]
        comptime for j in range(3):
            var col5 = Array[Co, 5](fill=Co(0, 0))
            comptime for k in range(5):
                col5[k] = tmp[k * 3 + j]
            _fft_len_r[dtype, 5, inverse](col5)
            comptime for k in range(5):
                x[k * 3 + j] = col5[k]
        return
    comptime if R == 20:
        # 20 = 4×5 Bailey four-step (FFT along n1=4 first).
        var tmp = Array[Co, 20](fill=Co(0, 0))
        comptime for k in range(5):
            var col = Array[Co, 4](fill=Co(0, 0))
            comptime for j in range(4):
                col[j] = x[j * 5 + k]
            _fft_len_r[dtype, 4, inverse](col)
            comptime for j in range(4):
                comptime if j > 0 and k > 0:
                    var w = _get_twiddle_factor[
                        dtype, inverse=inverse, N = UInt(20)
                    ](UInt(j * k))
                    col[j] = w * col[j]
                tmp[k * 4 + j] = col[j]
        comptime for k in range(5):
            var row = Array[Co, 4](fill=Co(0, 0))
            comptime for j in range(4):
                row[j] = tmp[k * 4 + j]
            # Length-5 FFT on indices (k + t*5)? After transpose shape is (5,4)
            # with layout tmp[k*4+j]; FFT along k needs gathering k for fixed j.
            _ = row
        comptime for j in range(4):
            var col5 = Array[Co, 5](fill=Co(0, 0))
            comptime for k in range(5):
                col5[k] = tmp[k * 4 + j]
            _fft_len_r[dtype, 5, inverse](col5)
            comptime for k in range(5):
                x[k * 4 + j] = col5[k]
        return

    comptime if UInt(R).is_power_of_two():
        comptime LOG = Int(count_trailing_zeros(Scalar[DType.uint](R)))
        comptime for i in range(R):
            comptime j = _bit_reverse[LOG](i)
            comptime if j > i:
                var tmp = x[i]
                x[i] = x[j]
                x[j] = tmp

        comptime for s in range(1, LOG + 1):
            comptime m = 1 << s
            comptime mh = m // 2
            comptime for k0 in range(R // m):
                comptime k = k0 * m
                comptime for j in range(mh):
                    comptime w = _get_twiddle_factor[
                        dtype, inverse=inverse, N=UInt(m)
                    ](UInt(j))
                    var t = _unit_phasor_mul[w](x[k + j + mh])
                    var u = x[k + j]
                    x[k + j] = u + t
                    x[k + j + mh] = u - t
    else:
        # Naive O(R²) DFT with comptime twiddles for small primes.
        var y = Array[Co, R](fill=Co(0, 0))
        comptime for k in range(R):
            comptime for n in range(R):
                comptime w = _get_twiddle_factor[
                    dtype, inverse=inverse, N=UInt(R)
                ](UInt(k * n % R))
                y[k] = y[k] + _unit_phasor_mul[w](x[n])
        comptime for k in range(R):
            x[k] = y[k]


@always_inline
def _radix_n_stockham_butterfly_reg[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=True, out_dtype, ...],
    x: TileTensor,
    b: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
):
    """Closed radix-R Stockham group: twiddle inputs, length-R DFT, store.

    `b` is the butterfly id in `0 .. N/R - 1`. `R = cfg.base` is a parameter.
    """
    _assert_radix_kernel_cfg[out_dtype, cfg]()
    comptime R = Int(cfg.base)
    comptime P = UInt(cfg.processed)
    var k0 = (b % P) + (b // P) * (P * UInt(R))

    comptime if _reg_dft_radix[UInt(R)]():
        var xs = _stockham_bfly_load_dft[out_dtype, cfg](
            output.as_immut(), x, b, twiddle_factors
        )
        _stockham_bfly_store[out_dtype, cfg](output, b, xs)
    else:
        comptime for i in range(UInt(R)):
            _radix_n_fft_kernel_elem_per_thread[out_dtype, cfg](
                output, x, k0 + i * P, twiddle_factors
            )


@always_inline
def _stockham_bfly_load_dft[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=False, out_dtype, ...],
    x: TileTensor,
    b: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
) -> Array[ComplexScalar[out_dtype], Int(cfg.base)]:
    """Load + twiddle + length-R DFT into registers (no SM store)."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()
    comptime R = Int(cfg.base)
    comptime Co = ComplexScalar[out_dtype]
    comptime P = UInt(cfg.processed)
    var k0 = (b % P) + (b // P) * (P * UInt(R))
    var n = cfg.stockham_n(k0)
    var xs = Array[Co, R](fill=Co(0, 0))
    comptime for j in range(UInt(R)):
        var xj = _get_x[j, out_dtype, cfg](output, x, n, k0)
        comptime if j > 0:
            xj = (
                _stockham_elem_twiddle[out_dtype, cfg, j](k0, twiddle_factors)
                * xj
            )
        xs[j] = xj
    _fft_len_r[out_dtype, R, cfg.inverse](xs)
    return xs^


@always_inline
def _stockham_bfly_store[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=True, out_dtype, ...],
    b: UInt,
    xs: Array[ComplexScalar[out_dtype], Int(cfg.base)],
):
    """Store a register DFT tile to Stockham output positions."""
    comptime P = UInt(cfg.processed)
    comptime R = Int(cfg.base)
    var k0 = (b % P) + (b // P) * (P * UInt(R))
    comptime for i in range(UInt(R)):
        var yi = xs[i]
        comptime if cfg.is_last_ifft:
            yi *= cfg.`1 / N`[out_dtype]
        output.raw_store(
            Int(k0 + i * P) * cfg.out_complex_stride, to_CoV(yi)
        )


@always_inline
def _radix_n_fft_kernel_elem_per_thread[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=True, out_dtype, ...],
    x: TileTensor,
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
):
    """Stockham per-element stage (out-of-place)."""
    var v = _radix_n_fft_kernel_elem_to_reg[out_dtype, cfg](
        output, x, local_i, twiddle_factors
    )
    output.raw_store(Int(local_i) * cfg.out_complex_stride, v)


@always_inline
def _radix_n_fft_kernel_elem_per_thread_comptime[
    out_dtype: DType, cfg: _FFTKernelExecConfig, local_i: UInt
](output: TileTensor[mut=True, out_dtype, ...], x: TileTensor):
    """Comptime-unrolled Stockham per-element stage."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()

    comptime n = cfg.stockham_n(local_i)

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    var x_out = _get_x[0, out_dtype, cfg](output, x, n, local_i)

    comptime for j in range(UInt(1), cfg.base):
        var x_j = _get_x[j, out_dtype, cfg](output, x, n, local_i)

        comptime max_twf_idx = (cfg.base - 1) * UInt(cfg.next_offset)
        comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
        comptime base_idx = Sc_twf(j) * Sc_twf(
            cfg.Sc(local_i) % cfg.next_offset
        )
        comptime twf_index = (
            cfg.Sc(base_idx % Sc_twf(cfg.next_offset)) * cfg.ratio
        )
        comptime twf = _get_twiddle_factor[
            out_dtype, inverse=cfg.inverse, N=cfg.length
        ](twf_index)

        comptime if cfg.processed == 1 and cfg.do_rfft:
            x_out = _unit_phasor_fma[twf, accum_is_real=j == 1](x_j.re, x_out)
        else:
            x_out = _unit_phasor_fma[twf](x_j, x_out)

    comptime if cfg.is_last_ifft:
        x_out *= cfg.`1 / N`[out_dtype]

    output.raw_store(Int(local_i) * cfg.out_complex_stride, to_CoV(x_out))
