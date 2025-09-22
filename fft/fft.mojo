from algorithm import parallelize
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceil
from bit import next_power_of_two
from sys.info import has_accelerator, simd_width_of

from fft.utils import (
    _get_dtype,
    _get_ordered_items,
    _approx_sin,
    _approx_cos,
    _get_twiddle_factors,
    _prep_twiddle_factors,
    _log_mod,
    _get_ordered_bases_processed_list,
)

# TODO: benchmark whether adding numbers like 6, 8 or 10 is worth it
alias _DEFAULT_BASES: List[UInt] = [7, 5, 4, 3, 2]

# TODO: specialized fft_2d and fft_3d. They can have better memory reuse.


# TODO: batched fft for multiple sequences intra_block and inter_multiprocessor
# - For intra_block: the amount of sequences can be scheduled as the amount of
# blocks. Multiple calls to enqueue_function can be made if bigger than
# max_amnt_blocks.
# - For inter_multiprocessor: _inter_multiprocessor_fft_kernel_radix_n could
# probably be parametrized to use global_i = global_i % sequence_length.
# Multiple calls to enqueue_function can be made if bigger than max_threads
fn fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt] = _DEFAULT_BASES,
    inverse: Bool = False,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    """Calculate the Discrete Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.
        inverse: Whether to run the inverse fourier transform.
        target: Target device ("cpu" or "gpu").

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
        1 <= len(in_layout) <= 2, "in_layout must have only 1 or 2 axis"
    ]()

    alias do_rfft = len(in_layout) == 1

    @parameter
    if not do_rfft:
        constrained[
            len(in_layout) == 2 and in_layout.shape[1].value() == 2,
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

    alias bases_processed = _get_ordered_bases_processed_list[length, bases]()
    alias ordered_bases = bases_processed[0]
    alias processed_list = bases_processed[1]

    @parameter
    if is_cpu[target]():
        _cpu_fft_kernel_radix_n[
            length=length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
        ](output, x)
        return
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    alias gpu_info = ctx.default_device_info
    alias max_threads_per_block = gpu_info.max_thread_block_size
    alias max_threads_available = gpu_info.threads_per_sm * gpu_info.sm_count
    alias num_threads = length // ordered_bases[len(ordered_bases) - 1]
    alias num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.index]()
    )
    alias shared_mem_size = gpu_info.shared_memory_per_multiprocessor

    @parameter
    if (
        num_threads <= max_threads_per_block
        and out_dtype.size_of() * length * 2 <= shared_mem_size
    ):
        ctx.enqueue_function[
            _intra_block_fft_kernel_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                length=length,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
                do_rfft=do_rfft,
                inverse=inverse,
            ]
        ](output, x, grid_dim=1, block_dim=num_threads)
    elif num_threads <= max_threads_available:
        alias block_dim = UInt(ceil(num_threads / num_blocks))
        _launch_inter_multiprocessor_fft[
            length=length,
            processed_list=processed_list,
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
            inverse=inverse,
            block_dim=block_dim,
            num_blocks=num_blocks,
        ](output, x, ctx)
    else:
        # TODO: Implement for sequences > max_threads_available in the same GPU
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


@always_inline
fn ifft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt] = _DEFAULT_BASES,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    """Calculate the Discrete Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        bases: The list of bases for which to build the mixed-radix algorithm.
        target: Target device ("cpu" or "gpu").

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
    fft[bases=bases, inverse=True, target=target](output, x, ctx)


# ===-----------------------------------------------------------------------===#
# inter_multiprocessor_fft
# ===-----------------------------------------------------------------------===#


fn _launch_inter_multiprocessor_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    length: UInt,
    processed_list: List[UInt],
    ordered_bases: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    num_blocks: UInt,
    block_dim: UInt,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    alias last_base = ordered_bases[len(ordered_bases) - 1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        var bases = materialize[ordered_bases]()
        var offsets = List[UInt](
            capacity=(length // last_base) * (last_base - 1) * len(bases)
        )
        var val = UInt(0)
        for i in range(len(bases)):
            offsets.append(val)
            val += (length // bases[i]) * (bases[i] - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]
    alias twf_layout = Layout.row_major(total_twfs, 2)
    twfs = ctx.enqueue_create_buffer[out_dtype](total_twfs * 2)
    with twfs.map_to_host() as twfs_host:

        @parameter
        for b in range(len(ordered_bases)):
            alias base = ordered_bases[b]
            alias processed = processed_list[b]
            alias amnt_threads = length // base
            alias base_twfs = _prep_twiddle_factors[
                length, base, processed, out_dtype, inverse
            ]()

            @parameter
            for i in range(amnt_threads):

                @parameter
                for j in range(base - 1):
                    alias t = base_twfs[i][j]
                    alias idx = twf_offsets[b] + i * (base - 1) + j
                    twfs_host[2 * idx] = t.re
                    twfs_host[2 * idx + 1] = t.im

    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
    )

    ctx.enqueue_function[
        _inter_multiprocessor_reorder_kernel[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            length=length,
            base = ordered_bases[0],
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
        ]
    ](output, x, grid_dim=num_blocks, block_dim=block_dim)

    @parameter
    for b in range(len(ordered_bases)):
        ctx.enqueue_function[
            _inter_multiprocessor_fft_kernel_radix_n[
                out_dtype,
                out_layout,
                twiddle_factors.layout,
                twiddle_factors.origin,
                twiddle_factors.address_space,
                length=length,
                base = ordered_bases[b],
                processed = processed_list[b],
                do_rfft=do_rfft,
                inverse=inverse,
                twf_offset = twf_offsets[b],
            ]
        ](output, twiddle_factors, grid_dim=num_blocks, block_dim=block_dim)


fn _inter_multiprocessor_reorder_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    length: UInt,
    base: UInt,
    ordered_bases: List[UInt],
    do_rfft: Bool,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """A reorder kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    alias amnt_threads = length // base
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var is_execution_thread = global_i < amnt_threads

    # reorder input x(global_i) items to match F(current_item) layout
    if is_execution_thread:
        _reorder_kernel[
            length=length,
            do_rfft=do_rfft,
            base=base,
            ordered_bases=ordered_bases,
        ](output, x, global_i)


fn _inter_multiprocessor_fft_kernel_radix_n[
    out_dtype: DType,
    out_layout: Layout,
    twf_layout: Layout,
    twf_origin: ImmutableOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    base: UInt,
    processed: UInt,
    do_rfft: Bool,
    inverse: Bool,
    twf_offset: UInt,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """An FFT that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    alias amnt_threads = length // base
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var is_execution_thread = global_i < amnt_threads

    if is_execution_thread:
        _radix_n_fft_kernel[
            out_dtype=out_dtype,
            out_layout = output.layout,
            out_address_space = output.address_space,
            twf_layout = twiddle_factors.layout,
            twf_origin = twiddle_factors.origin,
            twf_address_space = twiddle_factors.address_space,
            do_rfft=do_rfft,
            base=base,
            length=length,
            processed=processed,
            inverse=inverse,
            twf_offset=twf_offset,
        ](output, global_i, twiddle_factors)


# ===-----------------------------------------------------------------------===#
# intra_block_fft
# ===-----------------------------------------------------------------------===#


fn _intra_block_fft_kernel_radix_n[
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
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """An FFT that assumes `sequence_length // smallest_base <=
    max_threads_per_block` and that `sequence_length` out_dtype items fit in
    a block's shared memory."""

    var local_i = thread_idx.x
    var shared_f = tb[out_dtype]().row_major[length, 2]().shared().alloc()

    alias last_base = ordered_bases[len(ordered_bases) - 1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        var bases = materialize[ordered_bases]()
        var offsets = List[UInt](
            capacity=(length // last_base) * (last_base - 1) * len(bases)
        )
        var val = UInt(0)
        for i in range(len(bases)):
            offsets.append(val)
            val += (length // bases[i]) * (bases[i] - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]
    var twfs = tb[out_dtype]().row_major[total_twfs, 2]().shared().alloc()

    if local_i == 0:

        @parameter
        for b in range(len(ordered_bases)):
            alias base = ordered_bases[b]
            alias processed = processed_list[b]
            alias amnt_threads = length // base
            alias base_twfs = _prep_twiddle_factors[
                length, base, processed, out_dtype, inverse
            ]()

            @parameter
            for i in range(amnt_threads):

                @parameter
                for j in range(base - 1):
                    alias t = base_twfs[i][j]
                    alias idx = twf_offsets[b] + i * (base - 1) + j
                    twfs[idx, 0] = t.re
                    twfs[idx, 1] = t.im
    barrier()

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
                _reorder_kernel[
                    length=length,
                    do_rfft=do_rfft,
                    base=base,
                    ordered_bases=ordered_bases,
                ](shared_f, x, local_i)

        # NOTE: no barrier is needed here when processed == 1 because each
        # thread copies what it needs to run

        if is_execution_thread:
            _radix_n_fft_kernel[
                out_dtype=out_dtype,
                out_layout = shared_f.layout,
                out_address_space = shared_f.address_space,
                twf_layout = twfs.layout,
                twf_address_space = twfs.address_space,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
            ](shared_f, local_i, twfs.get_immutable())
        barrier()

    # when in the last stage, copy back to global memory
    # TODO: make sure this is the most efficient

    @parameter
    for i in range(last_base):
        alias offset = i * length // last_base
        var res = shared_f.load[width=2](local_i + offset, 0)
        output.store(local_i + offset, 0, res)
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
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """An FFT that runs on the CPU."""

    alias last_base = ordered_bases[len(ordered_bases) - 1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        var bases = materialize[ordered_bases]()
        var offsets = List[UInt](
            capacity=(length // last_base) * (last_base - 1) * len(bases)
        )
        var val = UInt(0)
        for i in range(len(bases)):
            offsets.append(val)
            val += (length // bases[i]) * (bases[i] - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]
    # TODO: this could probably just be a comptime InlineArray LayoutTensor
    var twfs = tb[out_dtype]().row_major[total_twfs, 2]().alloc()

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base
        alias base_twfs = _prep_twiddle_factors[
            length, base, processed, out_dtype, inverse
        ]()

        @parameter
        for i in range(amnt_threads):

            @parameter
            for j in range(base - 1):
                alias t = base_twfs[i][j]
                alias idx = twf_offsets[b] + i * (base - 1) + j
                twfs[idx, 0] = t.re
                twfs[idx, 1] = t.im

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base

        @parameter
        fn _inner_kernel(local_i: Int):
            @parameter
            if processed == 1:
                # reorder input x(global_i) items to match F(current_item) layout
                _reorder_kernel[
                    length=length,
                    do_rfft=do_rfft,
                    base=base,
                    ordered_bases=ordered_bases,
                ](output, x, local_i)

            # NOTE: no barrier is needed here when processed == 1 because each
            # thread copies what it needs to run

            _radix_n_fft_kernel[
                out_dtype=out_dtype,
                out_layout = output.layout,
                out_address_space = output.address_space,
                twf_layout = twfs.layout,
                twf_address_space = twfs.address_space,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
            ](output, local_i, twfs.get_immutable())

        # parallelize[func=_inner_kernel](amnt_threads)
        for i in range(amnt_threads):
            _inner_kernel(i)


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


fn _radix_n_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutableOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    twf_offset: UInt,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space
    ],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
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

    alias Co = ComplexSIMD[out_dtype, 1]
    alias is_even = length % 2 == 0  # avoid evaluating for uneven

    @parameter
    fn _base_phasor[i: UInt, j: UInt]() -> Co:
        # NOTE: using Float64 to avoid multiplication error accumulation
        alias base_twf = _get_twiddle_factors[base, DType.float64, inverse]()
        var val = ComplexSIMD[DType.float64, 1](1, 0)

        @parameter
        for _ in range(i):
            val *= base_twf[j - 1]
        return Co(val.re.cast[out_dtype](), val.im.cast[out_dtype]())

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

    @parameter
    for i in range(base):
        alias step = i * offset
        var idx = Int(n + step)

        @parameter
        if do_rfft and processed == 1:
            x[i] = Co(rebind[Scalar[out_dtype]](output[idx, 0]), 0)
        else:
            var data = output.load[2](idx, 0)
            x[i] = Co(data[0], data[1])

    var x_out = InlineArray[Co, base](fill=x[0])

    @parameter
    for j in range(1, base):

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                alias base_phasor = _base_phasor[i, j]()
                x_out[i] = _twf_fma[base_phasor, j == 1](x[j], x_out[i])
            continue

        var i0_j_twf_vec = twiddle_factors.load[2](
            twf_offset + local_i * (base - 1) + (j - 1), 0
        )
        var i0_j_twf = Co(i0_j_twf_vec[0], i0_j_twf_vec[1])

        @parameter
        for i in range(base):
            alias base_phasor = _base_phasor[i, j]()
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

            x_out[i] = _twf_fma(twf, x[j], x_out[i])

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        alias factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        for i in range(base):
            x_out[i].re *= factor
            x_out[i].im *= factor

    @parameter
    for i in range(base):
        alias step = i * offset
        # TODO: make sure this is the most efficient
        var idx = n + step
        output[idx, 0] = x_out[i].re
        output[idx, 1] = x_out[i].im

        @parameter
        if is_rfft_final_stage:  # copy the symmetric conjugates
            # when idx == 0 its conjugate is idx == length + 1
            # when the sequence length is even then the next_idx can be idx
            # when idx == rfft_idx_limit
            var next_idx = Sc(length) * Sc(Int(idx != 0)) - idx
            if next_idx != idx:
                output[Int(next_idx), 0] = x_out[i].re
                output[Int(next_idx), 1] = -x_out[i].im


# ===-----------------------------------------------------------------------===#
# _reorder_kernel
# ===-----------------------------------------------------------------------===#


fn _reorder_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    ordered_bases: List[UInt],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    local_i: UInt,
):
    alias amnt_threads = length // base
    alias ordered_items = _get_ordered_items[length, ordered_bases]()

    @parameter
    for i in range(base):
        alias offset = i * amnt_threads
        var idx = local_i + offset

        var current_item: UInt

        @parameter
        if base == length:  # do a DFT on the inputs
            current_item = idx
        else:
            current_item = UInt(ordered_items.unsafe_get(idx))

        @parameter
        if do_rfft:
            output[current_item, 0] = x[idx].cast[out_dtype]()
            # NOTE: filling the imaginary part with 0 is not necessary
            # because the _radix_n_fft_kernel already sets it to 0
            # when do_rfft and processed == 1
        else:
            alias msg = "in_layout must be complex valued"
            constrained[len(in_layout) == 2, msg]()
            constrained[in_layout.shape[1].value() == 2, msg]()
            # TODO: make sure this is the most efficient
            output.store(
                current_item,
                0,
                x.load[width=2](idx, 0).cast[out_dtype](),
            )
