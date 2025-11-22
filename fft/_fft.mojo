from algorithm import parallelize, vectorize
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
    _get_ordered_bases_processed_list,
    _max,
    _min,
)

# ===-----------------------------------------------------------------------===#
# inter_multiprocessor_fft
# ===-----------------------------------------------------------------------===#


fn _launch_inter_or_intra_multiprocessor_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    length: UInt,
    processed_list: List[UInt],
    ordered_bases: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    num_blocks: UInt,
    block_dim: UInt,
    batches: UInt,
    total_twfs: UInt,
    twf_offsets: List[UInt],
    max_cluster_size: UInt,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    ctx: DeviceContext,
) raises:
    comptime twf_layout = Layout.row_major(Int(total_twfs), 2)
    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    var twfs = ctx.enqueue_create_buffer[out_dtype](twf_layout.size())
    ctx.enqueue_copy(twfs, twfs_array.unsafe_ptr())
    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
    )

    comptime grid_dim = (Int(num_blocks), Int(batches))
    comptime gpu_info = ctx.default_device_info
    comptime is_sm_90_or_newer = (
        gpu_info.vendor == Vendor.NVIDIA_GPU and gpu_info.compute >= 9.0
    )

    @parameter
    if is_sm_90_or_newer and num_blocks <= max_cluster_size:
        comptime func = _inter_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors.layout,
            x.origin,
            output.origin,
            twiddle_factors.origin,
            twiddle_factors.address_space,
            length=length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            twf_offsets=twf_offsets,
        ]
        ctx.enqueue_function_checked[func, func](
            output,
            x,
            twiddle_factors,
            grid_dim=grid_dim,
            block_dim=Int(block_dim),
        )
    else:
        comptime func[b: Int] = _inter_multiprocessor_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors.layout,
            x.origin,
            output.origin,
            twiddle_factors.origin,
            twiddle_factors.address_space,
            length=length,
            base = ordered_bases[b],
            ordered_bases=ordered_bases,
            processed = processed_list[b],
            do_rfft=do_rfft,
            inverse=inverse,
            twf_offset = twf_offsets[b],
        ]

        @parameter
        for b in range(len(ordered_bases)):
            ctx.enqueue_function_checked[func[b], func[b]](
                output,
                x,
                twiddle_factors,
                grid_dim=grid_dim,
                block_dim=Int(block_dim),
            )


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
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    comptime amnt_threads = length // base
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_layout = Layout.row_major(
        Int(length), in_layout.shape[2].value()
    )
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
    comptime x_out_layout = Layout.row_major(Int(base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutOrigin.external
    ].stack_allocation()

    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime func = _radix_n_fft_kernel[
        do_rfft=do_rfft,
        base=base,
        length=length,
        processed=processed,
        inverse=inverse,
        twf_offset=twf_offset,
        ordered_bases=ordered_bases,
    ]

    @parameter
    if amnt_threads == total_threads:
        func(output, x, global_i, twiddle_factors, x_out)
    else:
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
        out_dtype, block_out_layout, MutOrigin.external
    ].stack_allocation()
    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = LayoutTensor[out_dtype, twfs_layout](
        twfs_array.unsafe_ptr().as_immutable()
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
        var res = shared_f.load[width=2](Int(local_i + offset), 0)
        output.store(Int(local_i + offset), 0, res)

    @parameter
    if not warp_exec:
        barrier()


# ===-----------------------------------------------------------------------===#
# _cpu_fft_kernel_radix_n
# ===-----------------------------------------------------------------------===#


fn _cpu_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    total_twfs: UInt,
    twf_offsets: List[UInt],
](
    batch_output: LayoutTensor[mut=True, out_dtype, out_layout],
    batch_x: LayoutTensor[mut=False, in_dtype, in_layout],
    cpu_workers: Optional[UInt] = None,
):
    """An FFT that runs on the CPU."""

    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        twfs_array.unsafe_ptr()
    )
    comptime amnt_threads = in_layout.shape[0].value()

    @parameter
    fn _inner_cpu_kernel(global_i: Int):
        var block_num = UInt(global_i)
        comptime block_out_layout = Layout.row_major(Int(length), 2)
        var output = LayoutTensor[
            mut=True, out_dtype, block_out_layout, batch_output.origin
        ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
        comptime x_layout = Layout.row_major(
            Int(length), in_layout.shape[2].value()
        )
        var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
            batch_x.ptr + batch_x.stride[0]() * Int(block_num)
        )
        comptime max_base = Int(ordered_bases[0])
        var x_out_array = InlineArray[Scalar[out_dtype], max_base * 2](
            uninitialized=True
        )
        var x_out = LayoutTensor[
            mut=True, out_dtype, Layout.row_major(max_base, 2)
        ](x_out_array.unsafe_ptr())

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
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
            fn _run[width: Int](local_i: Int):
                func(output, x, UInt(local_i), twfs, x_out)

            comptime width = UInt(simd_width_of[out_dtype]())
            comptime unroll_factor = Int(base) if base <= width else 1
            vectorize[_run, 1, size = Int(base), unroll_factor=unroll_factor]()

    parallelize[func=_inner_cpu_kernel](
        amnt_threads,
        Int(cpu_workers.or_else(UInt(parallelism_level()))),
    )


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


@always_inline
fn _radix_n_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    out_layout: Layout,
    in_layout: Layout,
    in_origin: ImmutOrigin,
    in_address_space: AddressSpace,
    out_origin: MutOrigin,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    x_out_layout: Layout,
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
        out_dtype, out_layout, out_origin, address_space=out_address_space
    ],
    x: LayoutTensor[
        in_dtype, in_layout, in_origin, address_space=in_address_space
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
    x_out: LayoutTensor[mut=True, out_dtype, x_out_layout],
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    constrained[length >= base, "length must be >= base"]()
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
        comptime base_twf = _get_twiddle_factors[base, out_dtype, inverse]()
        res = {1, 0}

        @parameter
        for _ in range(i):
            res *= base_twf[j - 1]

    @parameter
    @always_inline
    fn _twf_fma[twf: Co, is_j1: Bool](x_j_v: CoV, acc_v: CoV) -> CoV:
        var x_j = to_Co(x_j_v)
        var acc = to_Co(acc_v)
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
    fn _get_x[i: UInt]() -> SIMD[out_dtype, 2]:
        @parameter
        if processed == 1:
            # reorder input x(local_i) items to match F(current_item) layout
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            @parameter
            if base == length:  # do a DFT on the inputs
                copy_from = idx
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            @parameter
            if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return x.load[2](Int(copy_from), 0).cast[out_dtype]()
        else:
            comptime step = Sc(i * offset)
            return output.load[2](Int(n + step), 0)

    var x_0 = _get_x[0]()

    @parameter
    for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: CoV

                @parameter
                if j == 1:
                    acc = x_0
                else:
                    acc = x_out.load[2](Int(i), 0)
                x_out.store(Int(i), 0, _twf_fma[base_phasor, j == 1](x_j, acc))
            continue

        var i0_j_twf_vec = twiddle_factors.load[2](
            Int(twf_offset + local_i * (base - 1) + (j - 1)), 0
        )
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

            var acc: CoV

            @parameter
            if j == 1:
                acc = x_0
            else:
                acc = x_out.load[2](Int(i), 0)

            x_out.store(Int(i), 0, to_CoV(twf.fma(to_Co(x_j), to_Co(acc))))

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
    if base.is_power_of_two() and processed == 1:
        output.store(Int(n), 0, x_out.load[Int(base * 2)](0, 0))
    elif base.is_power_of_two() and out_address_space is AddressSpace.GENERIC:
        comptime offsets = _scatter_offsets()
        var v = x_out.load[Int(base * 2)](0, 0)
        output.ptr.offset(n * 2).scatter(offsets, v)
    else:

        @parameter
        for i in range(base):
            comptime step = Sc(i * offset)
            output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))
