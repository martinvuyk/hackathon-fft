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
        out_dtype, x_out_layout, MutOrigin.external
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
# inter_block_fft
# ===-----------------------------------------------------------------------===#


fn _inter_block_fft_kernel_radix_n[
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
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    twf_offsets: List[UInt],
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_per_block_cluster`."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_layout = Layout.row_major(
        Int(length), in_layout.shape[2].value()
    )
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    # TODO: this should use distributed shared memory for the intermediate output
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime x_out_layout = Layout.row_major(Int(ordered_bases[0]), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime amnt_threads = length // base
        comptime func = _radix_n_fft_kernel[
            do_rfft=do_rfft,
            base=base,
            length=length,
            processed=processed,
            inverse=inverse,
            twf_offset = twf_offsets[b],
            ordered_bases=ordered_bases,
        ]

        @parameter
        if amnt_threads == total_threads:
            func(output, x, global_i, twiddle_factors, x_out)
        else:
            if global_i < amnt_threads:  # is execution thread
                func(output, x, global_i, twiddle_factors, x_out)

        cluster_arrive_relaxed()
        cluster_wait()


# ===-----------------------------------------------------------------------===#
# intra_block_fft
# ===-----------------------------------------------------------------------===#


fn _intra_block_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    total_twfs: UInt,
    twf_offsets: List[UInt],
    warp_exec: Bool,
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `sequence_length // smallest_base <=
    max_threads_per_block` and that `sequence_length` out_dtype items fit in
    a block's shared memory."""

    var local_i = thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_complex = in_layout.shape[2].value()
    comptime x_layout = Layout.row_major(Int(length), x_complex)
    var x = LayoutTensor[in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    var output = LayoutTensor[out_dtype, block_out_layout, batch_output.origin](
        batch_output.ptr + batch_output.stride[0]() * Int(block_num)
    )
    var shared_f = LayoutTensor[
        out_dtype,
        block_out_layout,
        MutOrigin.external,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
    ref twfs_array_runtime = global_constant[twfs_array]()
    var twfs = LayoutTensor[out_dtype, twfs_layout](
        twfs_array_runtime.unsafe_ptr().as_immutable()
    )
    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime x_out_layout = Layout.row_major(Int(ordered_bases[0]), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime amnt_threads = length // base
        comptime func = _radix_n_fft_kernel[
            do_rfft=do_rfft,
            base=base,
            length=length,
            processed=processed,
            inverse=inverse,
            twf_offset = twf_offsets[b],
            ordered_bases=ordered_bases,
        ]

        @parameter
        if amnt_threads == total_threads:
            func(shared_f, x, local_i, twfs, x_out)
        else:
            if local_i < amnt_threads:  # is execution thread
                func(shared_f, x, local_i, twfs, x_out)

        @parameter
        if not warp_exec:
            barrier()

    @parameter
    for i in range(last_base):
        comptime offset = i * total_threads
        var idx = Int(local_i + offset)
        output.store(idx, 0, shared_f.load[width=2](idx, 0))

    @parameter
    if not warp_exec:
        barrier()


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
        out_dtype, out_layout, out_origin, address_space=out_address_space, **_
    ],
    x: LayoutTensor[
        in_dtype, in_layout, in_origin, address_space=in_address_space, **_
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
    x_out: LayoutTensor[
        mut=True, out_dtype, x_out_layout, address_space=x_out_address_space
    ],
    enable_debug: Bool = False,
    i_to_debug: UInt = 0,
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
    fn _scatter_offsets(out res: SIMD[Sc.dtype, Int(base * 2)]):
        res = {}
        var idx = 0
        for i in range(base):
            res[idx] = i * offset * 2
            idx += 1
            res[idx] = i * offset * 2 + 1
            idx += 1

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
