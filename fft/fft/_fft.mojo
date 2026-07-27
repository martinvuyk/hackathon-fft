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
](TrivialRegisterPassable):
    """Comptime execution parameters and shared Stockham/butterfly indexing."""

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
        """Stockham read index for `local_i`."""
        return (Self.Sc(local_i) // Self.next_offset) * Self.offset + (
            Self.Sc(local_i) % Self.next_offset
        ) % Self.offset


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
            return to_Co(x.raw_load[2](src_idx * 2).cast[out_dtype]())
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
            return to_Co(x.raw_load[2](Int(copy_from) * 2).cast[out_dtype]())
    else:
        comptime step = Sc(i * cfg.processed)
        return to_Co(output.raw_load[2](Int(n + step) * 2))


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
                Int(n + step) * 2, x_out.raw_load[CoV.size](Int(i) * 2)
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
def _radix_n_fft_kernel_elem_per_thread[
    out_dtype: DType, cfg: _FFTKernelExecConfig
](
    output: TileTensor[mut=True, out_dtype, ...],
    x: TileTensor,
    local_i: UInt,
    twiddle_factors: TileTensor[mut=False, ...],
):
    """Stockham per-element stage (out-of-place)."""
    _assert_radix_kernel_cfg[out_dtype, cfg]()

    var n = cfg.stockham_n(local_i)

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]

    var x_out = _get_x[0, out_dtype, cfg](output, x, n, local_i)

    comptime for j in range(UInt(1), cfg.base):
        var x_j = _get_x[j, out_dtype, cfg](output, x, n, local_i)

        var twf_index: cfg.Sc
        comptime max_twf_idx = (cfg.base - 1) * UInt(cfg.next_offset)
        comptime Sc_twf = Scalar[_get_dtype[max_twf_idx]()]
        var base_idx = Sc_twf(j) * Sc_twf(cfg.Sc(local_i) % cfg.next_offset)
        twf_index = cfg.Sc(base_idx % Sc_twf(cfg.next_offset)) * cfg.ratio
        var twf = _load_twiddle[out_dtype, cfg](twf_index, twiddle_factors)

        x_out = twf.fma(x_j, x_out)

    comptime if cfg.is_last_ifft:
        x_out *= cfg.`1 / N`[out_dtype]

    output.raw_store(Int(local_i) * 2, to_CoV(x_out))


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

    output.raw_store(Int(local_i) * 2, to_CoV(x_out))
