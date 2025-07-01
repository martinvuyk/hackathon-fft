from bit import bit_reverse
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import log2, exp, pi, cos, sin, iota
from sys import sizeof, argv


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
    constrained[len(in_layout) == 1, "in_layout must have only 1 axis"]()
    alias length = in_layout.shape[0].value()
    constrained[
        out_layout.shape == IntTuple(length, 2),
        "out_layout shape must be (in_layout.shape[0], 2)",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()
    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    alias max_threads_per_block = 64
    alias max_threads_available = 128

    @parameter
    if length.is_power_of_two():

        @parameter
        if length <= max_threads_per_block:
            _intra_block_fft_launch[
                threads_per_block=length, blocks_per_grid=1
            ](output, x, ctx)
        elif length <= max_threads_available:
            _inter_block_fft[
                threads_per_block=max_threads_per_block,
                blocks_per_grid = max_threads_available
                // max_threads_per_block,
            ](output, x, ctx)
        else:
            constrained[
                False,
                "fft for sequences longer than max_threads_available",
                "is not implemented yet",
            ]()
    else:
        constrained[
            False,
            "FFT for non-power-of-two sequence lengths is not implemented yet",
        ]()


fn _get_ordered_items[length: UInt](out res: InlineArray[UInt, length]):
    """The Butterfly diagram orders indexes by bit-reversed size."""
    res = InlineArray[UInt, length](uninitialized=True)
    values = List[UInt](capacity=length)
    for i in range(length):
        values.append(bit_reverse(i))
    sort(values)
    for i in range(length):
        res[i] = bit_reverse(values[i])


fn _get_offsets[stages: UInt](out res: InlineArray[UInt, stages]):
    """Offsets are the numeric jump that the indexes make from the
    upper line to the lower line in the butterfly diagram. It is
    the same number for the whole stage."""
    res = InlineArray[UInt, stages](uninitialized=True)

    for i in reversed(range(Int(stages))):
        res[UInt(i)] = UInt(2**i)


fn _get_twiddle_factors[
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
            res[i] = C(cos(theta).__round__(16), sin(theta).__round__(16))
            i += 1


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
    var twiddle_factors = _get_twiddle_factors[stages, length, out_dtype]()
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
    constrained[len(in_layout) == 1, "in_layout must have only 1 axis"]()
    alias length = in_layout.shape[0].value()
    constrained[
        length.is_power_of_two(), "input sequence length must be a power of two"
    ]()
    constrained[
        out_layout.shape == IntTuple(length, 2),
        "out_layout shape must be (in_layout.shape[0], 2)",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()
    constrained[
        threads_per_block == length,
        "threads_per_block must be equal to sequence length",
    ]()
    # TODO: maybe constraint layout on row major for read write perf.

    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    alias offsets: InlineArray[UInt, stages] = _get_offsets[stages]()
    alias ordered_items = _get_ordered_items[length]()
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    # alias twiddle_factors: InlineArray[
    #     ComplexSIMD[out_dtype, 1], length - 1
    # ] = _get_twiddle_factors[stages, length, out_dtype]()
    _intra_block_fft_kernel_core[
        stages=stages,
        length=length,
        ordered_items_length=length,
        ordered_items=ordered_items,
        offsets=offsets,
        threads_per_block=threads_per_block,
    ](output, x, twiddle_factors)


fn _intra_block_fft_kernel_core[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    stages: UInt,
    length: UInt,
    ordered_items_length: UInt,
    ordered_items: InlineArray[UInt, ordered_items_length],
    offsets: InlineArray[UInt, stages],
    threads_per_block: UInt,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # complex vectors for the frequencies
    shared_f = (
        tb[out_dtype]().row_major[threads_per_block, 2]().shared().alloc()
    )

    # reorder input x(global_i) items to match F(current_item) layout
    current_item = ordered_items[global_i]
    shared_f[current_item, 0] = x[global_i].cast[out_dtype]()
    shared_f[current_item, 1] = 0  # imaginary part

    @parameter
    for stage in range(stages):
        _fft_kernel[
            out_dtype=out_dtype,
            out_layout = shared_f.layout,
            address_space = shared_f.address_space,
            stages=stages,
            length=length,
            ordered_items_length=ordered_items_length,
            ordered_items=ordered_items,
            offsets=offsets,
            stage=stage,
        ](shared_f, twiddle_factors, local_i)
        barrier()

    output[global_i, 0] = shared_f[local_i, 0]
    output[global_i, 1] = shared_f[local_i, 1]


fn _fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    *,
    stages: UInt,
    length: UInt,
    ordered_items_length: UInt,
    ordered_items: InlineArray[UInt, ordered_items_length],
    offsets: InlineArray[UInt, stages],
    stage: UInt,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
    local_i: UInt,
):
    alias Complex = ComplexSIMD[out_dtype, _]
    alias pow2 = 2**stage
    next_idx = local_i + offsets[stage]
    # Run the Danielson-Lanczos Lemma if the current local_i is an "x_0"
    # line in the butterfly diagram. "x_0" would mean that the given
    # item is the lhs term that always gets added without any multiplication
    is_execution_thread = False
    if next_idx < length:
        delta = ordered_items[local_i] + pow2 - ordered_items[next_idx]
        delta_0_reference = (
            ordered_items[0] + pow2 - ordered_items[offsets[stage]]
        )
        # only "x_0" paths get the same delta as the 0th line
        is_execution_thread = delta == delta_0_reference

    if is_execution_thread:
        # get the twiddle factor W
        twiddle_idx = UInt(local_i % pow2)
        # base_idx is the offset to the end of the previous stage
        alias base_idx: UInt = (2 ** (stage + 1)) // 2 - 1
        twiddle_factor = twiddle_factors[base_idx + twiddle_idx]
        twiddle_factor_vec = Complex[output.element_size](
            twiddle_factor.re, twiddle_factor.im
        )

        # get the upper and lower paths in the butterfly diagram
        x_0 = Complex(output[local_i, 0], output[local_i, 1])
        x_1 = Complex(output[next_idx, 0], output[next_idx, 1])

        # f_0 = x_0 + W * x_1
        res_0 = twiddle_factor_vec.fma(x_1, x_0)
        output[local_i, 0] = res_0.re
        output[local_i, 1] = res_0.im
        # f_1 = x_0 - W * x_1
        res_1 = (-twiddle_factor_vec).fma(x_1, x_0)
        output[next_idx, 0] = res_1.re
        output[next_idx, 1] = res_1.im


fn _inter_block_fft_kernel[
    out_dtype: DType,
    out_layout: Layout,
    *,
    threads_per_block: Int,
    stages: UInt,
    length: UInt,
    ordered_items: InlineArray[UInt, length],
    offsets: InlineArray[UInt, stages],
    skipped_stages: UInt = 0,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[ComplexSIMD[out_dtype, 1], length - 1],
):
    """An FFT that assumes `num_threads_per_block * amount_blocks ==
    sequence_length`.
    """
    alias Complex = ComplexSIMD[out_dtype, _]

    global_i = block_dim.x * block_idx.x + thread_idx.x

    @parameter
    for stage in range(skipped_stages, stages):
        _fft_kernel[
            out_dtype=out_dtype,
            out_layout = output.layout,
            address_space = output.address_space,
            stages=stages,
            length=length,
            ordered_items_length=length,
            ordered_items=ordered_items,
            offsets=offsets,
            stage=stage,
        ](output, twiddle_factors, global_i)
        barrier()


def _inter_block_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout, //,
    *,
    threads_per_block: Int,
    blocks_per_grid: Int,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
):
    """An FFT that assumes `num_threads_per_block * amount_blocks ==
    sequence_length`.
    """
    constrained[len(in_layout) == 1, "in_layout must have only 1 axis"]()
    alias length = in_layout.shape[0].value()
    constrained[
        length.is_power_of_two(), "input sequence length must be a power of two"
    ]()
    constrained[
        out_layout.shape == IntTuple(length, 2),
        "out_layout shape must be (in_layout.shape[0], 2)",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()
    # TODO: maybe constraint layout on row major for read write perf.
    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    alias ordered_items = _get_ordered_items[length]()
    alias offsets: InlineArray[UInt, stages] = _get_offsets[stages]()
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    # so we just calculate the twiddle factors on the cpu at runtime
    var twiddle_factors = _get_twiddle_factors[stages, length, out_dtype]()
    alias stages_limit = UInt(
        log2(Float64(threads_per_block)).cast[DType.index]()
    )

    # run the _intra_block_fft_kernel over the first stages
    # handles initial reordering as well

    @parameter
    fn _intra_block_fft_kernel_wrapper[
        in_dtype: DType,
        out_dtype: DType,
        in_layout: Layout,
        out_layout: Layout,
    ](
        output: LayoutTensor[mut=True, out_dtype, out_layout],
        x: LayoutTensor[mut=False, in_dtype, in_layout],
        # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
        _twiddle_factors: InlineArray[
            ComplexSIMD[out_dtype, 1], threads_per_block - 1
        ],
    ):
        alias offsets = _get_offsets[stages_limit]()

        _intra_block_fft_kernel_core[
            threads_per_block=threads_per_block,
            stages=stages_limit,
            length=threads_per_block,
            ordered_items_length=length,
            ordered_items=ordered_items,
            offsets=offsets,
        ](output, x, _twiddle_factors)

    var limited_twiddle_factors = _get_twiddle_factors[
        stages_limit, threads_per_block, out_dtype
    ]()
    ctx.enqueue_function[
        _intra_block_fft_kernel_wrapper[
            in_dtype, out_dtype, in_layout, out_layout
        ]
    ](
        output,
        limited_twiddle_factors,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )
    # run the _inter_block_fft_kernel
    ctx.enqueue_function[
        _inter_block_fft_kernel[
            out_dtype,
            out_layout,
            threads_per_block=threads_per_block,
            skipped_stages=stages_limit,
            stages=stages,
            length=length,
            ordered_items=ordered_items,
            offsets=offsets,
        ]
    ](
        output,
        twiddle_factors,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )
