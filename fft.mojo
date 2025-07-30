from bit import bit_reverse
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import log2, exp, pi, cos, sin, iota, sqrt
from sys import sizeof, argv
from sys.info import is_gpu


def fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
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

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.

    Notes:
        This function automatically runs the rfft if the input is real-valued.
        Then copies the symetric results into their corresponding slots in the
        output tensor.
    """
    # TODO: maybe constraint layout on row major for read write perf.
    constrained[is_gpu(), "The current FFT implementation is for GPU only"]()
    constrained[
        1 <= len(in_layout) <= 2, "in_layout must have only 1 or 2 axis"
    ]()
    # TODO: implement radix-3
    # TODO: implement radix-5
    # TODO: implement radix-7
    # TODO: weave the implementations together for a composite signal 2x3x4x5x7
    # NOTE: highest power of 2 that divides a number: (n & (~(n - 1)))
    # for the other radixes `while n % radix_number == 0: ...`

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
    alias max_threads_per_block = 64  # TODO
    alias max_threads_available = 128  # TODO

    @parameter
    if length.is_power_of_two():

        @parameter
        if length <= max_threads_per_block:
            _intra_block_fft_launch[
                threads_per_block=length, blocks_per_grid=1
            ](output, x, ctx)
        elif length <= max_threads_available:
            _inter_block_fft_launch[
                threads_per_block=max_threads_per_block,
                block_dim = max_threads_available // max_threads_per_block,
            ](output, x, ctx)
        else:
            # TODO: Implement for sequences > max_threads_available
            constrained[
                False,
                "fft for sequences longer than max_threads_available",
                "is not implemented yet",
            ]()
    else:
        # TODO: implement a mixed-radix algorithm
        # TODO: implement slower path for prime number/other sequence lengths
        constrained[
            False,
            "FFT for non-power-of-two sequence lengths is not implemented yet",
        ]()


# ===-----------------------------------------------------------------------===#
# intra_block power of two
# ===-----------------------------------------------------------------------===#


def _intra_block_fft_launch[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    threads_per_block: Int,
    blocks_per_grid: Int,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
):
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    # so we just calculate the twiddle factors on the cpu at runtime
    alias length = in_layout.shape[0].value()
    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    var twiddle_factors = _get_pow2_twiddle_factors[stages, length, out_dtype]()
    ctx.enqueue_function[
        _intra_block_fft_kernel[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            threads_per_block=threads_per_block,
        ]
    ](
        output,
        x,
        twiddle_factors,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )


fn _intra_block_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    threads_per_block: Int,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[
        ComplexSIMD[out_dtype, 1], in_layout.shape[0].value() - 1
    ],
):
    """An FFT that assumes `num_threads_per_block == sequence_length` and that
    there is only one block."""
    alias length = in_layout.shape[0].value()
    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    _intra_block_fft_kernel_core[
        stages=stages, length=length, threads_per_block=threads_per_block
    ](output, x, twiddle_factors)


fn _intra_block_fft_kernel_core[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    stages: UInt,
    length: UInt,
    threads_per_block: UInt,
    *,
    stage_start: UInt = 0,
    stage_end: UInt = stages,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
):
    alias ordered_items = _get_ordered_items[length, List(UInt(2))]()
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # complex vectors for the frequencies
    var shared_f = (
        tb[out_dtype]().row_major[threads_per_block, 2]().shared().alloc()
    )

    # reorder input x(global_i) items to match F(current_item) layout
    var current_item = UInt(ordered_items[global_i])

    alias do_rfft = len(in_layout) == 1

    @parameter
    if do_rfft:
        shared_f[current_item, 0] = x[global_i].cast[out_dtype]()
        # NOTE: filling the imaginary part with 0 is not necessary
        # because the _radix_2_fft_kernel already sets it to 0
        # when the stage == 0
    else:
        alias msg = "in_layout must be complex valued"
        constrained[len(in_layout) == 2, msg]()
        constrained[in_layout.shape[1].value() == 2, msg]()
        shared_f[current_item, 0] = x[global_i, 0].cast[out_dtype]()
        shared_f[current_item, 1] = x[global_i, 1].cast[out_dtype]()

    barrier()

    @parameter
    for stage in range(stage_start, stage_end):
        _radix_2_fft_kernel[
            out_dtype=out_dtype,
            out_layout = shared_f.layout,
            address_space = shared_f.address_space,
            stages=stages,
            length=length,
            stage=stage,
            do_rfft=do_rfft,
        ](shared_f, twiddle_factors, local_i)
        barrier()

    output[global_i, 0] = shared_f[local_i, 0]
    output[global_i, 1] = shared_f[local_i, 1]


# ===-----------------------------------------------------------------------===#
# intra_block radix-n
# ===-----------------------------------------------------------------------===#


def _intra_block_fft_launch_radix_n[
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
    alias length = in_layout.shape[0].value()
    alias twiddle_factors = _get_twiddle_factors[length, out_dtype]()

    @parameter
    fn _build_ordered_bases() -> List[UInt]:
        var processed = UInt(1)
        var new_bases = List[UInt](capacity=len(bases))

        @parameter
        for base in bases:
            var amnt_divisible = _log_mod[base](length // processed)[0]
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
        "powers of the bases must multiply together",
        " to equal the sequence length",
    ]()

    ctx.enqueue_function[
        _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors=twiddle_factors,
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


fn _intra_block_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    twiddle_factors: InlineArray[
        ComplexSIMD[out_dtype, 1], in_layout.shape[0].value() - 1
    ],
    ordered_bases: List[UInt],
    processed_list: List[UInt],
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """An FFT that assumes `sequence_length <= max_threads_per_block` and that
    there is only one block."""
    alias length = in_layout.shape[0].value()
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var shared_f = tb[out_dtype]().row_major[length, 2]().shared().alloc()

    alias do_rfft = len(in_layout) == 1

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias ratio = length // base
        var is_execution_thread = local_i < ratio

        @parameter
        if processed == 1:
            if is_execution_thread:
                alias ordered_items = _get_ordered_items[
                    length, ordered_bases
                ]()

                # reorder input x(global_i) items to match F(current_item) layout
                @parameter
                for i in range(base):
                    alias offset = i * ratio
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
                        # FIXME: make this a single load operation
                        shared_f[current_item, 0] = x[g_idx, 0].cast[
                            out_dtype
                        ]()
                        shared_f[current_item, 1] = x[g_idx, 1].cast[
                            out_dtype
                        ]()
        barrier()
        if is_execution_thread:
            _radix_n_fft_kernel[
                out_dtype=out_dtype,
                out_layout = shared_f.layout,
                address_space = shared_f.address_space,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                twiddle_factors=twiddle_factors,
            ](shared_f, local_i)
        barrier()

        # when in the last stage, copy back to global memory
        @parameter
        if processed * base == length:  # all threads should execute
            # FIXME: each thread should copy a contiguous block of memory
            # or only e.g. thread 0 should copy the whole thing

            @parameter
            for i in range(base):
                alias offset = i * ratio
                output[global_i + offset, 0] = shared_f[local_i + offset, 0]
                output[global_i + offset, 1] = shared_f[local_i + offset, 1]
        barrier()


# ===-----------------------------------------------------------------------===#
# inter_block
# ===-----------------------------------------------------------------------===#


def _inter_block_fft_launch[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    threads_per_block: UInt,
    block_dim: UInt,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
):
    """An FFT that assumes `num_threads_per_block * amount_blocks ==
    sequence_length`.
    """
    alias length = in_layout.shape[0].value()
    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    # so we just calculate the twiddle factors on the cpu at runtime
    var twiddle_factors = _get_pow2_twiddle_factors[stages, length, out_dtype]()

    ctx.enqueue_function[
        _inter_block_fft[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            threads_per_block=threads_per_block,
        ]
    ](
        output,
        x,
        twiddle_factors,
        grid_dim=threads_per_block,
        block_dim=block_dim,
    )


def _inter_block_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    stages: UInt,
    length: UInt,
    *,
    threads_per_block: Int,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
):
    """An FFT that assumes `num_threads_per_block * amount_blocks ==
    sequence_length`.
    """
    alias stages_limit = UInt(
        log2(Float64(threads_per_block)).cast[DType.index]()
    )
    # run the _intra_block_fft_kernel over the first stages using shared memory
    # handles initial reordering as well
    _intra_block_fft_kernel_core[
        threads_per_block=threads_per_block,
        stages=stages,
        length=length,
        stage_end=stages_limit,
    ](output, x, twiddle_factors)
    # run the _inter_block_fft_kernel
    _inter_block_fft_kernel[
        out_dtype,
        out_layout,
        threads_per_block=threads_per_block,
        stages=stages,
        length=length,
        do_rfft = len(in_layout) == 1,
        stage_start=stages_limit,
    ](output, twiddle_factors)


fn _inter_block_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    *,
    threads_per_block: Int,
    stages: UInt,
    length: UInt,
    do_rfft: Bool,
    stage_start: UInt = 0,
    stage_end: UInt = stages,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
):
    """An FFT that assumes `num_threads_per_block * amount_blocks ==
    sequence_length`.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x

    @parameter
    for stage in range(stage_start, stage_end):
        _radix_2_fft_kernel[
            out_dtype=out_dtype,
            out_layout = output.layout,
            address_space = output.address_space,
            stages=stages,
            length=length,
            stage=stage,
            do_rfft=do_rfft,
        ](output, twiddle_factors, global_i)
        barrier()


# ===-----------------------------------------------------------------------===#
# radix implementations
# ===-----------------------------------------------------------------------===#


fn _radix_2_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    stages: UInt,
    length: UInt,
    stage: UInt,
    do_rfft: Bool,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
    local_i: UInt,
):
    # TODO: test this function with non-power-of-two sequence lengths. It might
    # work with a few fixes.
    alias idx_scalar = Scalar[_get_dtype[length * 2]()]
    alias offset = idx_scalar(2**stage)
    alias rfft_idx_limit = idx_scalar(length // 2)
    alias is_rfft_final_stage = do_rfft and 2 ** (stage + 1) == length
    var curr_idx = idx_scalar(local_i)

    @parameter
    if is_rfft_final_stage:
        if curr_idx > rfft_idx_limit:
            return
    # Run the Danielson-Lanczos Lemma if the current local_i is an "x_0"
    # line in the butterfly diagram. "x_0" would mean that the given
    # item is the lhs term that always gets added without any multiplication
    var next_idx = UInt(curr_idx + offset)
    var is_lim_thread = curr_idx == rfft_idx_limit

    var is_execution_thread: Bool
    var is_x0_line = (curr_idx // offset) % 2 == 0

    @parameter
    if not is_rfft_final_stage:
        is_execution_thread = is_x0_line
    else:
        is_execution_thread = is_x0_line and (
            length - next_idx >= local_i or is_lim_thread
        )

    if not is_execution_thread:
        return

    alias Co = ComplexSIMD[out_dtype, output.element_size]

    # get the upper and lower paths in the butterfly diagram
    # f_0 = x_0 + W * x_1
    # f_1 = x_0 - W * x_1
    var x_0: Co

    @parameter
    if stage == 0 and do_rfft:
        x_0 = Co(output[local_i, 0], 0)
    else:
        x_0 = Co(output[local_i, 0], output[local_i, 1])
    var res_0: Co
    var res_1: Co

    @parameter
    if stage == 0 and do_rfft:  # W = Co(1, 0)
        var re = output[next_idx, 0]
        res_0 = Co(x_0.re + re, x_0.im)
        res_1 = Co(x_0.re - re, x_0.im)
    elif stage == 0:
        var x_1 = Co(output[next_idx, 0], output[next_idx, 1])
        res_0 = x_0 + x_1
        res_1 = x_0 - x_1
    else:
        # get the twiddle factor W
        var twiddle_idx = curr_idx % offset
        if twiddle_idx == 0:  # W = Co(1, 0)
            var x_1 = Co(output[next_idx, 0], output[next_idx, 1])
            res_0 = x_0 + x_1
            res_1 = x_0 - x_1
        else:
            # base_idx is the offset to the end of the previous stage
            alias base_idx = offset - 1
            var twiddle_factor = twiddle_factors[base_idx + twiddle_idx]
            if twiddle_factor.im == -1:  # W = Co(0, -1)
                var x_1 = Co(output[next_idx, 1], -output[next_idx, 0])
                res_0 = x_0 + x_1
                res_1 = x_0 - x_1
            else:
                var x_1 = Co(output[next_idx, 0], output[next_idx, 1])
                var twiddle_factor_vec = Co(
                    twiddle_factor.re, twiddle_factor.im
                )

                res_0 = twiddle_factor_vec.fma(x_1, x_0)
                res_1 = (-twiddle_factor_vec).fma(x_1, x_0)

    output[local_i, 0] = res_0.re
    output[local_i, 1] = res_0.im

    output[next_idx, 0] = res_1.re
    output[next_idx, 1] = res_1.im

    @parameter
    if is_rfft_final_stage:  # copy the symmetric conjugates
        var idx = length - local_i
        if local_i != 0 and idx != local_i and idx != next_idx:
            output[idx, 0] = res_0.re
            output[idx, 1] = -res_0.im

        idx = length - next_idx
        if idx != local_i and idx != next_idx:
            output[idx, 0] = res_1.re
            output[idx, 1] = -res_1.im


fn _radix_3_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    stages: UInt,
    length: UInt,
    stage: UInt,
    do_rfft: Bool,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    local_i: UInt,
):
    # TODO
    # alias idx_scalar = Scalar[_get_dtype[length * 3]()]
    # alias offset = idx_scalar(3**stage)
    # var curr_idx = idx_scalar(local_i)
    # if (curr_idx // offset) % idx_scalar(3) != 0:  # execution thread
    #     return

    # alias Co = ComplexSIMD[out_dtype, output.element_size]
    # alias `√3` = sqrt(Scalar[out_dtype](3))
    # alias Nx = offset
    # var n = curr_idx

    # var x_0 = Co(output[UInt(n), 0], output[UInt(n), 1])
    # var x_1 = Co(output[UInt(n + Nx), 0], output[UInt(n + Nx), 1])
    # var x_2 = Co(output[UInt(n + 2 * Nx), 0], output[UInt(n + 2 * Nx), 1])

    # var v_0 = -(x_1 + x_2)
    # var v_1 = x_1 - x_2
    # output[UInt(n), 0] = x_0.re + x_1.re + x_2.re
    # output[UInt(n), 1] = x_0.im + x_1.im + x_2.im
    # output[UInt(n + Nx), 0] = (v_1.im.fma(`√3`, v_0.re)).fma(0.5, x_0.re)
    # output[UInt(n + Nx), 1] = (v_1.re.fma(-`√3`, v_0.im)).fma(0.5, x_0.im)
    # output[UInt(n + 2 * Nx), 0] = (v_1.im.fma(-`√3`, v_0.re)).fma(0.5, x_0.re)
    # output[UInt(n + 2 * Nx), 1] = (v_1.re.fma(`√3`, v_0.im)).fma(0.5, x_0.im)
    ...


fn _radix_5_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    stages: UInt,
    length: UInt,
    stage: UInt,
    do_rfft: Bool,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    local_i: UInt,
):
    # TODO
    # alias idx_scalar = Scalar[_get_dtype[length * 5]()]
    # alias offset = idx_scalar(5**stage)
    # var curr_idx = idx_scalar(local_i)
    # if (curr_idx // offset) % idx_scalar(5) != 0:  # execution thread
    #     return

    # alias Co = ComplexSIMD[out_dtype, output.element_size]
    # alias Nx = offset
    # var n = curr_idx

    # TODO: make these numbers more exact
    # alias W1_5 = Scalar[out_dtype](0.30901699437494)
    # alias W2_5 = Scalar[out_dtype](0.95105651629515)
    # alias W3_5 = Scalar[out_dtype](0.80901699437494)
    # alias W4_5 = Scalar[out_dtype](0.58778525229247)
    # var x_0 = Co(output[UInt(n), 0], output[UInt(n), 1])
    # var x_1 = Co(output[UInt(n + Nx), 0], output[UInt(n + Nx), 1])
    # var x_2 = Co(output[UInt(n + 2 * Nx), 0], output[UInt(n + 2 * Nx), 1])
    # var x_3 = Co(output[UInt(n + 3 * Nx), 0], output[UInt(n + 3 * Nx), 1])
    # var x_4 = Co(output[UInt(n + 4 * Nx), 0], output[UInt(n + 4 * Nx), 1])

    # var v_1 = W1_5 * (x_1 + x_4) - W3_5 * (x_2 + x_3)
    # var v_2 = W3_5 * (x_1 + x_4) - W1_5 * (x_2 + x_3)
    # var v_3 = W4_5 * (x_1 - x_4) - W2_5 * (x_2 - x_3)
    # var v_4 = W2_5 * (x_1 - x_4) + W4_5 * (x_2 - x_3)
    # x_0 = x_0 + x_1 + x_2 + x_3 + x_4
    # x_1 = x_0 + v_1 + Co(v_4.im, -v_4.re)
    # x_2 = x_0 - v_2 + Co(v_3.im, -v_3.re)
    # x_3 = x_0 - v_2 + Co(-v_3.im, v_3.re)
    # x_4 = x_0 + v_1 + Co(-v_4.im, v_4.re)

    # output[UInt(n), 0] = x_0.re
    # output[UInt(n), 1] = x_0.im
    # output[UInt(n + Nx), 0] = x_1.re
    # output[UInt(n + Nx), 1] = x_1.im
    # output[UInt(n + 2 * Nx), 0] = x_2.re
    # output[UInt(n + 2 * Nx), 1] = x_2.im
    # output[UInt(n + 3 * Nx), 0] = x_3.re
    # output[UInt(n + 3 * Nx), 1] = x_3.im
    # output[UInt(n + 4 * Nx), 0] = x_4.re
    # output[UInt(n + 4 * Nx), 1] = x_4.im
    ...


fn _radix_7_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    stages: UInt,
    length: UInt,
    stage: UInt,
    do_rfft: Bool,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    local_i: UInt,
):
    # TODO
    # alias idx_scalar = Scalar[_get_dtype[length * 7]()]
    # alias offset = idx_scalar(7**stage)
    # var curr_idx = idx_scalar(local_i)
    # if (curr_idx // offset) % idx_scalar(7) != 0:  # execution thread
    #     return

    # alias Co = ComplexSIMD[out_dtype, output.element_size]
    # alias Nx = offset
    # var n = curr_idx

    # # TODO: make these numbers more exact
    # alias W1_7 = Scalar[out_dtype](0.62348980185873)
    # alias W2_7 = Scalar[out_dtype](0.78183148246802)
    # alias W3_7 = Scalar[out_dtype](0.22252093395631)
    # alias W4_7 = Scalar[out_dtype](0.97492791218182)
    # alias W5_7 = Scalar[out_dtype](0.90096886790241)
    # alias W6_7 = Scalar[out_dtype](0.43388373911755)
    # var x_0 = Co(output[UInt(n), 0], output[UInt(n), 1])
    # var x_1 = Co(output[UInt(n + Nx), 0], output[UInt(n + Nx), 1])
    # var x_2 = Co(output[UInt(n + 2 * Nx), 0], output[UInt(n + 2 * Nx), 1])
    # var x_3 = Co(output[UInt(n + 3 * Nx), 0], output[UInt(n + 3 * Nx), 1])
    # var x_4 = Co(output[UInt(n + 4 * Nx), 0], output[UInt(n + 4 * Nx), 1])
    # var x_5 = Co(output[UInt(n + 5 * Nx), 0], output[UInt(n + 5 * Nx), 1])
    # var x_6 = Co(output[UInt(n + 6 * Nx), 0], output[UInt(n + 6 * Nx), 1])

    # var v_1 = W1_7 * (x_1 + x_6) - W3_7 * (x_2 + x_5) - W5_7 * (x_3 + x_4)
    # var v_2 = W3_7 * (x_1 + x_6) + W5_7 * (x_2 + x_5) - W1_7 * (x_3 + x_4)
    # var v_3 = W5_7 * (x_1 + x_6) - W1_7 * (x_2 + x_5) + W3_7 * (x_3 + x_4)
    # var v_4 = W2_7 * (x_1 - x_6) + W4_7 * (x_2 - x_5) + W5_7 * (x_3 - x_4)
    # var v_5 = W4_7 * (x_1 - x_6) - W5_7 * (x_2 - x_5) - W2_7 * (x_3 - x_4)
    # var v_6 = W5_7 * (x_1 - x_6) - W2_7 * (x_2 - x_5) + W4_7 * (x_3 - x_4)
    # x_0 = x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6
    # x_1 = x_0 + v_1 + Co(v_4.im, -v_4.re)
    # x_2 = x_0 - v_2 + Co(v_5.im, -v_5.re)
    # x_3 = x_0 - v_3 + Co(v_6.im, -v_6.re)
    # x_4 = x_0 - v_3 + Co(-v_6.im, v_6.re)
    # x_5 = x_0 - v_2 + Co(-v_5.im, v_5.re)
    # x_6 = x_0 + v_1 + Co(-v_4.im, v_4.re)

    # output[UInt(n), 0] = x_0.re
    # output[UInt(n), 1] = x_0.im
    # output[UInt(n + Nx), 0] = x_1.re
    # output[UInt(n + Nx), 1] = x_1.im
    # output[UInt(n + 2 * Nx), 0] = x_2.re
    # output[UInt(n + 2 * Nx), 1] = x_2.im
    # output[UInt(n + 3 * Nx), 0] = x_3.re
    # output[UInt(n + 3 * Nx), 1] = x_3.im
    # output[UInt(n + 4 * Nx), 0] = x_4.re
    # output[UInt(n + 4 * Nx), 1] = x_4.im
    # output[UInt(n + 5 * Nx), 0] = x_5.re
    # output[UInt(n + 5 * Nx), 1] = x_5.im
    # output[UInt(n + 6 * Nx), 0] = x_6.re
    # output[UInt(n + 6 * Nx), 1] = x_6.im
    ...


# TODO: benchmark this implementation thoroughly
fn _radix_n_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    length: UInt,
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
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
    for i in range(base):
        ref x_i = x_out[i]
        x_i = x[0]
        var idx = n + i * offset

        @parameter
        for j in range(1, base):
            ref x_j = x[j]

            @parameter
            if processed == 1 and i == 0:
                x_i.re += x_j.re

                @parameter
                if not do_rfft:
                    x_i.im += x_j.im
                continue

            alias is_even = length % 2 == 0  # avoid evaluating for uneven
            alias next_offset = offset * Sc(base)
            alias ratio = Sc(length) // next_offset
            var twiddle_idx = ((Sc(j) * idx) % next_offset) * ratio

            if twiddle_idx == 0:  # Co(1, 0)
                x_i.re += x_j.re

                @parameter
                if not is_first_rfft_stage:
                    x_i.im += x_j.im
                else:
                    x_i.im += 0
            elif is_even and 4 * twiddle_idx == length:  # Co(0, -1)

                @parameter
                if not is_first_rfft_stage:
                    x_i.re += x_j.im
                else:
                    x_i.re += 0
                x_i.im += -x_j.re
            elif is_even and 2 * twiddle_idx == length:  # Co(-1, 0)
                x_i.re += -x_j.re

                @parameter
                if not is_first_rfft_stage:
                    x_i.im += -x_j.im
                else:
                    x_i.im += 0
            elif is_even and 4 * twiddle_idx == 3 * length:  # Co(0, 1)

                @parameter
                if not is_first_rfft_stage:
                    x_i.re += -x_j.im
                else:
                    x_i.re += 0
                x_i.im += x_j.re
            else:
                var twf = twiddle_factors.unsafe_get(twiddle_idx - 1)

                @parameter
                if not is_first_rfft_stage:
                    # x_i = Co(twf.re, twf.im).fma(x_j, x_i)
                    var res = Co(twf.re, twf.im) * x_j
                    x_i.re += res.re
                    x_i.im += res.im
                else:
                    alias SM = SIMD[out_dtype, output.element_size]
                    # x_i.re = SM(twf.re).fma(x_j.re, x_i.re)
                    # x_i.im = SM(twf.im).fma(x_j.re, x_i.im)
                    x_i.re += SM(twf.re) * x_j.re
                    x_i.im += SM(twf.im) * x_j.re

    @parameter
    for i in range(base):
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

    Given N = R_0 * R_1 * ... * R_{M-1}, an input index `k` is represented as:
    k = d_0 + d_1*R_0 + d_2*R_0*R_1 + ... + d_{M-1}*R_0*...*R_{M-2}
    where d_i is the digit for radix R_i.

    The reversed index k' is:
    k' = d_{M-1} + d_{M-2}*R_{M-1} + ... + d_1*R_{M-1}*...*R_2 + d_0*R_{M-1}*...*R_1

    Parameters:
        bases: A List of UInt representing the radices in the order
            R_0, R_1, ..., R_{M-1}.

    Args:
        idx: The input index to be reversed.

    Returns:
        The digit-reversed index.
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


# TODO: this can be redesigned to skip over C(1, 0) and
# calculate twiddle factors only for the last stage and
# indexing into them thus using length // 2 -1 size
# instead of length -1
fn _get_pow2_twiddle_factors[
    stages: UInt, length: UInt, dtype: DType
](out res: InlineArray[ComplexSIMD[dtype, 1], length - 1]):
    """Twiddle factors are stored contiguously in memory but the
    logical layout is per stage + an offset.

    Examples:
        for a signal with 8 datapoints:
        stage 1: W_0_2
        stage 2: W_0_4, W_1_4
        stage 3: W_0_8, W_1_8, W_2_8, W_3_8
        the result is: [W_0_2, W_0_4, W_1_4, W_0_8, W_1_8, W_2_8, W_3_8]
    """
    alias C = ComplexSIMD[dtype, 1]
    res = __type_of(res)(uninitialized=True)
    var i = UInt(0)
    for N in [2**i for i in range(1, stages + 1)]:
        for n in range(N // 2):
            # exp((-j * 2 * pi * n) / N)
            theta = Scalar[dtype]((-2 * pi * n) / N)
            # TODO: this could be more generic using fputils
            res[i] = C(cos(theta).__round__(15), sin(theta).__round__(15))
            i += 1


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
    length: UInt, dtype: DType
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
        theta = Scalar[dtype]((-2 * pi * n) / N)
        # TODO: this could be more generic using fputils
        # 0.002275657645967044
        # 0.002275657645956386
        res[n - 1] = C(
            _approx_cos(theta).__round__(15), _approx_sin(theta).__round__(15)
        )


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
