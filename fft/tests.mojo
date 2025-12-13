from complex import ComplexScalar
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of
from utils.numerics import nan

from testing import assert_almost_equal

from fft.fft import fft
from fft._fft import (
    _intra_block_fft_kernel_radix_n,
    _launch_inter_or_intra_multiprocessor_fft,
)
from fft._utils import _get_ordered_bases_processed_list
from fft._test_values import (
    _TestValues,
    _get_test_values_2,
    _get_test_values_3,
    _get_test_values_4,
    _get_test_values_5,
    _get_test_values_6,
    _get_test_values_7,
    _get_test_values_8,
    _get_test_values_10,
    _get_test_values_16,
    _get_test_values_20,
    _get_test_values_21,
    _get_test_values_32,
    _get_test_values_35,
    _get_test_values_48,
    _get_test_values_60,
    _get_test_values_64,
    _get_test_values_100,
    _get_test_values_128,
)


fn test_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    test_num: UInt,
    bases: List[UInt],
    inverse: Bool = False,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    constrained[in_layout.rank() == 3, "in_layout must have rank 3"]()
    comptime batches = UInt(in_layout.shape[0].value())
    comptime sequence_length = UInt(in_layout.shape[1].value())
    comptime in_complex = in_layout.shape[2].value()
    comptime do_rfft = in_complex == 1

    constrained[
        do_rfft or in_complex == 2,
        "The layout should match one of: {(batches, sequence_length, 1), ",
        "(batches, sequence_length, 2)}",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    comptime bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((sequence_length // last_base) * (last_base - 1))
        var offsets = List[UInt](capacity=c * len(bases))
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]

    @parameter
    if is_cpu[target]():
        fft[bases= [bases], inverse=inverse](output, x)
        return
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_m = gpu_info.threads_per_multiprocessor
    constrained[
        threads_per_m > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    comptime max_threads_available = UInt(threads_per_m * gpu_info.sm_count)
    comptime num_threads = sequence_length // ordered_bases[
        len(ordered_bases) - 1
    ]

    # TODO?: Implement for sequences > max_threads_available in the same GPU
    constrained[
        num_threads <= max_threads_available,
        "fft for sequences longer than max_threads_available",
        "is not implemented yet. max_threads_available: ",
        String(max_threads_available),
    ]()

    comptime num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_per_m = UInt(gpu_info.shared_memory_per_multiprocessor)
    comptime shared_mem_per_block = max_threads_per_block * (
        shared_mem_per_m // UInt(threads_per_m)
    )
    comptime output_size = UInt(size_of[out_dtype]()) * sequence_length * 2
    comptime run_in_block = num_threads <= max_threads_per_block and (
        output_size
    ) <= shared_mem_per_block
    comptime batch_size = max_threads_available // num_threads
    comptime block_dim_inter_multiprocessor = UInt(
        ceil(num_threads / num_blocks).cast[DType.uint]()
    )

    @parameter
    fn _launch_fn[batch_size: Int, offset: Int]() raises:
        comptime out_batch_layout = Layout.row_major(
            batch_size, Int(sequence_length), 2
        )
        var out_batch = LayoutTensor[
            out_dtype, out_batch_layout, output.origin
        ](output.ptr + output.stride[0]() * offset)
        comptime x_batch_layout = Layout.row_major(
            batch_size, Int(sequence_length), in_complex
        )
        var x_batch = LayoutTensor[in_dtype, x_batch_layout, x.origin](
            x.ptr + x.stride[0]() * offset
        )
        comptime block_func_batch = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            x_batch_layout,
            out_batch_layout,
            x.origin,
            output.origin,
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            warp_exec = UInt(gpu_info.warp_size) >= num_threads
            and test_num == 1,
        ]

        @parameter
        if run_in_block and (test_num == 0 or test_num == 1):
            ctx.enqueue_function_checked[block_func_batch, block_func_batch](
                out_batch,
                x_batch,
                grid_dim=(1, batch_size),
                block_dim=Int(num_threads),
            )
        else:
            _launch_inter_or_intra_multiprocessor_fft[
                length=sequence_length,
                processed_list=processed_list,
                ordered_bases=ordered_bases,
                do_rfft=do_rfft,
                inverse=inverse,
                block_dim=block_dim_inter_multiprocessor,
                num_blocks=num_blocks,
                batches = UInt(batch_size),
                total_twfs=total_twfs,
                twf_offsets=twf_offsets,
                max_cluster_size = UInt(0 if test_num == 2 else 8),
            ](out_batch, x_batch, ctx)

    @parameter
    for i in range(batches // batch_size):
        _launch_fn[Int(batch_size), Int(i * batch_size)]()

    comptime remainder = batches % batch_size

    @parameter
    if remainder > 0:
        _launch_fn[Int(remainder), Int((batches - remainder) * batch_size)]()


def test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
]():
    comptime BATCHES = len(test_values)
    comptime SIZE = len(test_values[0][0])
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(
        BATCHES, SIZE, 2
    ) if inverse else Layout.row_major(BATCHES, SIZE, 1)
    comptime out_layout = Layout.row_major(BATCHES, SIZE, 2)

    @parameter
    if debug:
        print("----------------------------")
        print("SIZE:", SIZE)
        print("Buffers for Bases: ", end="")
        var b = materialize[bases]()
        print(b.__str__().replace("UInt(", "").replace(")", ""))
        print("----------------------------")

    @parameter
    fn _eval[
        res_layout: Layout, res_origin: MutOrigin
    ](
        result: LayoutTensor[out_dtype, res_layout, res_origin],
        scalar_in: List[Int],
        complex_out: List[ComplexScalar[out_dtype]],
    ) raises:
        @parameter
        if debug:
            print("out: ", end="")
            for i in range(SIZE):
                if i == 0:
                    print("[", result[i, 0], ", ", result[i, 1], sep="", end="")
                else:
                    print(
                        ", ", result[i, 0], ", ", result[i, 1], sep="", end=""
                    )
            print("]")
            print("expected: ", end="")

        comptime ATOL = 1e-3 if dtype is DType.float64 else (
            1e-2 if dtype is DType.float32 else 1e-1
        )
        comptime RTOL = 1e-5

        # gather all real parts and then the imaginary parts
        @parameter
        if inverse:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", scalar_in[i], ".0, 0.0", sep="", end="")
                    else:
                        print(", ", scalar_in[i], ".0, 0.0", sep="", end="")
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    Scalar[out_dtype](scalar_in[i]),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(result[i, 1], 0, atol=ATOL, rtol=RTOL)
        else:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", complex_out[i].re, ", ", sep="", end="")
                    else:
                        print(", ", complex_out[i].re, ", ", sep="", end="")
                    print(complex_out[i].im, end="")
                print("]")
            for i in range(SIZE):
                # break
                assert_almost_equal(
                    result[i, 0],
                    complex_out[i].re.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )

    with DeviceContext() as ctx:

        @parameter
        if target == "cpu":
            var out_data = List[Scalar[in_dtype]](
                length=out_layout.size(), fill=nan[in_dtype]()
            )
            var x_data = List[Scalar[out_dtype]](
                length=in_layout.size(), fill=nan[out_dtype]()
            )
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                Span(out_data)
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                Span(x_data)
            )

            for idx, test in enumerate(materialize[test_values]()):
                comptime x_layout = Layout.row_major(
                    in_layout.shape[1].value(), in_layout.shape[2].value()
                )
                var x = LayoutTensor[mut=True, in_dtype, x_layout](
                    batch_x.ptr + batch_x.stride[0]() * idx
                )
                for i in range(SIZE):

                    @parameter
                    if inverse:
                        x[i, 0] = test[1][i].re.cast[in_dtype]()
                        x[i, 1] = test[1][i].im.cast[in_dtype]()
                    else:
                        x[i, 0] = Scalar[in_dtype](test[0][i])

            test_fft[bases=bases, inverse=inverse, target=target, test_num=0](
                batch_output, batch_x, ctx
            )

            for idx, test in enumerate(materialize[test_values]()):
                comptime output_layout = Layout.row_major(
                    out_layout.shape[1].value(), 2
                )
                var output = LayoutTensor[
                    mut=True, out_dtype, output_layout, batch_output.origin
                ](batch_output.ptr + batch_output.stride[0]() * idx)
                _eval(output, test[0], test[1])
        else:
            var x_data = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
            x_data.enqueue_fill(Scalar[in_dtype].MAX)
            var out_data = ctx.enqueue_create_buffer[out_dtype](
                out_layout.size()
            )
            out_data.enqueue_fill(nan[out_dtype]())
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                out_data.unsafe_ptr()
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                x_data.unsafe_ptr()
            )
            with x_data.map_to_host() as x_host:
                for idx, test in enumerate(materialize[test_values]()):
                    comptime x_layout = Layout.row_major(
                        in_layout.shape[1].value(), in_layout.shape[2].value()
                    )
                    var x = LayoutTensor[mut=True, in_dtype, x_layout](
                        x_host.unsafe_ptr() + batch_x.stride[0]() * idx
                    )

                    for i in range(SIZE):

                        @parameter
                        if inverse:
                            x[i, 0] = test[1][i].re.cast[in_dtype]()
                            x[i, 1] = test[1][i].im.cast[in_dtype]()
                        else:
                            x[i, 0] = Scalar[in_dtype](test[0][i])

            ctx.synchronize()
            test_fft[
                bases=bases,
                inverse=inverse,
                target="gpu",
                test_num=test_num,
            ](batch_output, batch_x, ctx)
            ctx.synchronize()
            with out_data.map_to_host() as out_host:
                for idx, test in enumerate(materialize[test_values]()):
                    comptime output_layout = Layout.row_major(
                        out_layout.shape[1].value(), 2
                    )
                    var output = LayoutTensor[
                        mut=True, out_dtype, output_layout, batch_output.origin
                    ](out_host.unsafe_ptr() + batch_output.stride[0]() * idx)
                    _eval(output, test[0], test[1])

        @parameter
        if debug:
            print("----------------------------")
            print("Tests passed")
            print("----------------------------")


def _test_fft[
    dtype: DType,
    func: fn[bases: List[UInt], test_values: _TestValues[dtype]] () raises,
]():
    comptime L = List[UInt]

    comptime values_2 = _get_test_values_2[dtype]()
    func[[2], values_2]()

    comptime values_3 = _get_test_values_3[dtype]()
    func[[3], values_3]()

    comptime values_4 = _get_test_values_4[dtype]()
    func[[4], values_4]()
    func[[2], values_4]()

    comptime values_5 = _get_test_values_5[dtype]()
    func[[5], values_5]()

    comptime values_6 = _get_test_values_6[dtype]()
    func[[6], values_6]()
    func[[3, 2], values_6]()
    func[[2, 3], values_6]()

    comptime values_7 = _get_test_values_7[dtype]()
    func[[7], values_7]()

    comptime values_8 = _get_test_values_8[dtype]()
    func[[8], values_8]()
    func[[2], values_8]()
    func[[4, 2], values_8]()
    func[[2, 4], values_8]()

    comptime values_10 = _get_test_values_10[dtype]()
    func[[10], values_10]()
    func[[5, 2], values_10]()

    comptime values_16 = _get_test_values_16[dtype]()
    func[[16], values_16]()
    func[[2], values_16]()
    func[[4], values_16]()
    func[[2, 4], values_16]()
    func[[8, 2], values_16]()
    func[[2, 8], values_16]()

    comptime values_20 = _get_test_values_20[dtype]()
    func[[20], values_20]()
    func[[10, 2], values_20]()
    func[[5, 4], values_20]()
    func[[5, 2], values_20]()

    comptime values_21 = _get_test_values_21[dtype]()
    func[[7, 3], values_21]()

    comptime values_32 = _get_test_values_32[dtype]()
    func[[2], values_32]()
    func[[16, 2], values_32]()
    func[[8, 4], values_32]()
    func[[4, 2], values_32]()
    func[[8, 2], values_32]()

    comptime values_35 = _get_test_values_35[dtype]()
    func[[7, 5], values_35]()

    comptime values_48 = _get_test_values_48[dtype]()
    func[[8, 6], values_48]()
    func[[3, 2], values_48]()

    comptime values_60 = _get_test_values_60[dtype]()
    func[[10, 6], values_60]()
    func[[6, 5, 2], values_60]()
    func[[5, 4, 3], values_60]()
    func[[3, 4, 5], values_60]()
    func[[5, 3, 2], values_60]()

    comptime values_64 = _get_test_values_64[dtype]()
    func[[2], values_64]()
    func[[8], values_64]()
    func[[4], values_64]()
    func[[16, 4], values_64]()

    comptime values_100 = _get_test_values_100[dtype]()
    func[[20, 5], values_100]()
    func[[10], values_100]()
    func[[5, 4], values_100]()

    comptime values_128 = _get_test_values_128[dtype]()
    # func[[32, 4], values_128]()  # long compile times, but important to test
    func[[16, 8], values_128]()
    func[[16, 4, 2], values_128]()
    func[[8, 8, 2], values_128]()
    func[[8, 4, 4], values_128]()
    func[[8, 4, 2, 2], values_128]()
    func[[8, 2, 2, 2, 2], values_128]()
    func[[4, 4, 4, 2], values_128]()
    func[[4, 4, 2, 2, 2], values_128]()
    func[[4, 2, 2, 2, 2, 2], values_128]()
    func[[2], values_128]()


comptime _test[
    dtype: DType,
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
] = _test_fft[
    dtype,
    test_fft_radix_n[
        dtype, inverse=inverse, target=target, test_num=test_num, debug=debug
    ],
]


def test_fft():
    comptime dtype = DType.float64
    _test[dtype, False, "cpu", 0, debug=False]()
    # _test[dtype, False, "gpu", 0, debug=False]()
    # _test[dtype, False, "gpu", 1, debug=False]()
    # _test[dtype, False, "gpu", 2, debug=False]()
    # _test[dtype, False, "gpu", 3, debug=False]()


def test_ifft():
    comptime dtype = DType.float64
    _test[dtype, True, "cpu", 0, debug=False]()
    _test[dtype, True, "gpu", 0, debug=False]()
    _test[dtype, True, "gpu", 1, debug=False]()
    _test[dtype, True, "gpu", 2, debug=False]()
    _test[dtype, True, "gpu", 3, debug=False]()


comptime Co = ComplexScalar[DType.float64]

comptime input_2d: InlineArray[InlineArray[UInt8, 4], 6] = [
    [1, 0, 7, 4],
    [1, 7, 2, 1],
    [8, 1, 0, 9],
    [6, 4, 8, 0],
    [2, 1, 1, 4],
    [7, 5, 3, 7],
]

comptime expected_2d: InlineArray[InlineArray[Co, 4], 6] = [
    [Co(89.0, 0.0), Co(4.0, 7.0), Co(3.0, 0.0), Co(4.0, -7.0)],
    [
        Co(-2.5, 0.866025404),
        Co(-9.59807621, -1.23205081),
        Co(-7.5, 2.59807621),
        Co(-4.40192379, -2.23205081),
    ],
    [
        Co(0.5, 18.1865335),
        Co(-25.25833025, 6.89230485),
        Co(19.5, 2.59807621),
        Co(-2.74166975, 13.89230485),
    ],
    [Co(-13.0, 0.0), Co(2.0, 23.0), Co(-3.0, 0.0), Co(2.0, -23.0)],
    [
        Co(0.5, -18.1865335),
        Co(-2.74166975, -13.89230485),
        Co(19.5, -2.59807621),
        Co(-25.25833025, -6.89230485),
    ],
    [
        Co(-2.5, -0.866025404),
        Co(-4.40192379, 2.23205081),
        Co(-7.5, -2.59807621),
        Co(-9.59807621, 1.23205081),
    ],
]


def test_2d_cpu[debug: Bool]():
    comptime ROWS = 6
    comptime COLS = 4

    comptime x_layout = Layout.row_major(1, ROWS, COLS, 2)
    var x_buf = materialize[input_2d]()
    var x_buf2 = InlineArray[InlineArray[SIMD[DType.uint8, 2], 4], 6](
        uninitialized=True
    )
    for i in range(ROWS):
        for j in range(COLS):
            x_buf2[i][j] = {x_buf[i][j], 0}
    var x = LayoutTensor[mut=False, DType.uint8, x_layout](
        x_buf2.unsafe_ptr().bitcast[UInt8]()
    )

    comptime out_layout = Layout.row_major(1, ROWS, COLS, 2)
    var out_buf = InlineArray[Co, ROWS * COLS](
        fill=Co(nan[DType.float64](), nan[DType.float64]())
    )
    var out = LayoutTensor[mut=True, DType.float64, out_layout](
        out_buf.unsafe_ptr().bitcast[Float64]()
    )

    fft(out, x)

    @parameter
    if debug:
        print("Output values:")
        for i in range(ROWS):
            for j in range(COLS):
                print(
                    "out[0, ",
                    i,
                    ", ",
                    j,
                    "]: [",
                    out[0, i, j, 0],
                    ", ",
                    out[0, i, j, 1],
                    "] expected: [",
                    expected_2d[i][j].re,
                    ", ",
                    expected_2d[i][j].im,
                    "]",
                    sep="",
                )

    for i in range(ROWS):
        for j in range(COLS):
            assert_almost_equal(
                out[0, i, j, 0],
                expected_2d[i][j].re,
                String("i: ", i, " j: ", j, " re"),
            )
            assert_almost_equal(
                out[0, i, j, 1],
                expected_2d[i][j].im,
                String("i: ", i, " j: ", j, " im"),
            )


def test_2d_gpu():
    comptime ROWS = 6
    comptime COLS = 4
    comptime in_dtype = DType.uint8
    comptime out_dtype = DType.float64
    comptime in_layout = Layout.row_major(1, ROWS, COLS, 1)
    comptime out_layout = Layout.row_major(1, ROWS, COLS, 2)

    with DeviceContext() as ctx:
        var x_data = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
        x_data.enqueue_fill(Scalar[in_dtype].MAX)
        var out_data = ctx.enqueue_create_buffer[out_dtype](out_layout.size())
        out_data.enqueue_fill(nan[out_dtype]())

        var out = LayoutTensor[mut=True, out_dtype, out_layout](
            out_data.unsafe_ptr()
        )
        var x = LayoutTensor[mut=True, in_dtype, in_layout](x_data.unsafe_ptr())

        with x_data.map_to_host() as x_host:
            comptime test_data = materialize[input_2d]()
            var x_view = type_of(x)(x_host.unsafe_ptr())

            for i in range(ROWS):
                for j in range(COLS):
                    x_view[0, i, j, 0] = Scalar[in_dtype](test_data[i][j])

        ctx.synchronize()

        fft[target="gpu"](out, x.get_immutable(), ctx)

        ctx.synchronize()

        with out_data.map_to_host() as out_host:
            comptime expected_data = materialize[expected_2d]()
            var out_view = type_of(out)(out_host.unsafe_ptr())

            for i in range(ROWS):
                for j in range(COLS):
                    assert_almost_equal(
                        out_view[0, i, j, 0],
                        expected_data[i][j].re,
                        String("i: ", i, " j: ", j, " re"),
                    )
                    assert_almost_equal(
                        out_view[0, i, j, 1],
                        expected_data[i][j].im,
                        String("i: ", i, " j: ", j, " im"),
                    )


comptime input_3d: InlineArray[InlineArray[InlineArray[UInt8, 8], 4], 6] = [
    [
        [187, 94, 48, 255, 45, 95, 163, 8],
        [199, 162, 40, 224, 156, 114, 206, 188],
        [118, 216, 11, 84, 216, 30, 139, 187],
        [105, 76, 185, 28, 66, 210, 187, 202],
    ],
    [
        [247, 77, 120, 173, 33, 167, 123, 207],
        [18, 190, 243, 163, 119, 145, 185, 88],
        [178, 100, 125, 145, 25, 41, 53, 183],
        [198, 167, 226, 69, 250, 95, 32, 94],
    ],
    [
        [198, 194, 10, 122, 90, 78, 197, 22],
        [187, 228, 225, 111, 27, 138, 214, 93],
        [230, 52, 231, 116, 192, 222, 223, 82],
        [15, 8, 77, 54, 229, 4, 130, 91],
    ],
    [
        [141, 220, 93, 39, 245, 179, 113, 47],
        [161, 94, 4, 170, 50, 250, 64, 170],
        [63, 134, 128, 74, 119, 69, 99, 195],
        [142, 157, 59, 6, 83, 100, 163, 188],
    ],
    [
        [29, 86, 68, 118, 147, 213, 14, 235],
        [116, 221, 27, 29, 242, 222, 181, 29],
        [50, 155, 120, 157, 222, 254, 144, 75],
        [62, 76, 90, 239, 196, 221, 251, 142],
    ],
    [
        [95, 83, 220, 234, 255, 75, 255, 24],
        [22, 105, 225, 32, 11, 49, 131, 134],
        [169, 20, 183, 87, 84, 214, 118, 84],
        [36, 213, 189, 165, 53, 66, 15, 117],
    ],
]

comptime expected_3d: InlineArray[InlineArray[InlineArray[Co, 8], 4], 6] = [
    [
        [
            Co(24626.0, 0.0),
            Co(-282.338095, 533.610173),
            Co(-226.0, -600.0),
            Co(-95.6619049, -372.389827),
            Co(310.0, 0.0),
            Co(-95.6619049, 372.389827),
            Co(-226.0, 600.0),
            Co(-282.338095, -533.610173),
        ],
        [
            Co(-35.0, -575.0),
            Co(-184.241774, -387.399062),
            Co(-293.0, 229.0),
            Co(-90.5735931, -1353.11183),
            Co(-173.0, 547.0),
            Co(978.241774, 209.399062),
            Co(685.0, 307.0),
            Co(-175.426407, -136.888168),
        ],
        [
            Co(168.0, 0.0),
            Co(-305.614357, -34.9066376),
            Co(986.0, 370.0),
            Co(811.614357, -264.906638),
            Co(400.0, 0.0),
            Co(811.614357, 264.906638),
            Co(986.0, -370.0),
            Co(-305.614357, 34.9066376),
        ],
        [
            Co(-35.0, 575.0),
            Co(-175.426407, 136.888168),
            Co(685.0, -307.0),
            Co(978.241774, -209.399062),
            Co(-173.0, -547.0),
            Co(-90.5735931, 1353.11183),
            Co(-293.0, -229.0),
            Co(-184.241774, 387.399062),
        ],
    ],
    [
        [
            Co(185.5, -151.554446),
            Co(128.827155, -1919.76088),
            Co(-266.378912, 688.869293),
            Co(261.837906, -560.891253),
            Co(304.5, -939.637563),
            Co(543.484425, -316.170727),
            Co(-749.621088, -1146.13071),
            Co(711.850515, -449.040354),
        ],
        [
            Co(-517.86515, 234.233938),
            Co(26.9984468, 8.20881129),
            Co(241.692884, 298.15648),
            Co(-636.662075, 229.82709),
            Co(1145.32925, -70.3981502),
            Co(872.98747, -131.197121),
            Co(48.5221358, -179.074374),
            Co(-728.449209, -224.279501),
        ],
        [
            Co(-458.5, 205.248021),
            Co(491.692126, -459.591094),
            Co(499.449328, 374.89746),
            Co(288.800478, -1245.33676),
            Co(-91.5, 532.605623),
            Co(1112.89357, 166.041482),
            Co(272.550672, -548.10254),
            Co(-163.386173, -1298.21285),
        ],
        [
            Co(1196.86515, 34.2339376),
            Co(-313.955516, -713.619075),
            Co(-2142.52214, 289.925626),
            Co(1398.3819, -1045.73684),
            Co(-576.329251, 605.60185),
            Co(809.0668, -410.833336),
            Co(-227.692884, -810.84352),
            Co(-356.367817, 56.7485293),
        ],
    ],
    [
        [
            Co(-218.5, -742.183771),
            Co(1169.94632, 1160.5257),
            Co(614.799383, -664.602355),
            Co(133.731476, -217.971493),
            Co(-855.5, 1438.4682),
            Co(-501.932232, 685.495932),
            Co(793.200617, -859.602355),
            Co(392.254441, -426.006877),
        ],
        [
            Co(724.076766, -315.2147),
            Co(267.032324, 251.423488),
            Co(483.498113, -1192.70131),
            Co(1835.51316, -316.59163),
            Co(623.451215, 704.248021),
            Co(523.115043, -920.439964),
            Co(-878.013393, -532.685843),
            Co(-203.063514, 479.018638),
        ],
        [
            Co(-616.5, 1391.70282),
            Co(-491.17944, -1902.6073),
            Co(660.863263, -163.228277),
            Co(-418.788682, -289.683096),
            Co(506.5, 1556.24765),
            Co(-1795.93184, 871.801065),
            Co(-788.863263, 689.771723),
            Co(-8.10003441, 76.8768563),
        ],
        [
            Co(-419.076766, -5.2146997),
            Co(-69.4833464, -813.760763),
            Co(-373.986607, 134.314157),
            Co(1630.20974, -288.295123),
            Co(131.548785, -293.751979),
            Co(-864.966302, 38.1877696),
            Co(748.501887, 1376.29869),
            Co(-1342.35711, -438.721352),
        ],
    ],
    [
        [
            Co(904.0, 0.0),
            Co(-969.974747, 1371.10051),
            Co(512.0, -356.0),
            Co(19.9747468, -1390.89949),
            Co(180.0, 0.0),
            Co(19.9747468, 1390.89949),
            Co(512.0, 356.0),
            Co(-969.974747, -1371.10051),
        ],
        [
            Co(-1585.0, -695.0),
            Co(71.5227279, -928.962554),
            Co(-779.0, -335.0),
            Co(18.8049071, 310.886435),
            Co(-1007.0, -185.0),
            Co(962.477272, -21.0374465),
            Co(531.0, -1325.0),
            Co(1331.19509, -780.886435),
        ],
        [
            Co(-730.0, 0.0),
            Co(201.918831, -720.411255),
            Co(152.0, -326.0),
            Co(-179.918831, 41.588745),
            Co(-1122.0, 0.0),
            Co(-179.918831, -41.588745),
            Co(152.0, 326.0),
            Co(201.918831, 720.411255),
        ],
        [
            Co(-1585.0, 695.0),
            Co(1331.19509, 780.886435),
            Co(531.0, 1325.0),
            Co(962.477272, 21.0374465),
            Co(-1007.0, 185.0),
            Co(18.8049071, -310.886435),
            Co(-779.0, 335.0),
            Co(71.5227279, 928.962554),
        ],
    ],
    [
        [
            Co(-218.5, 742.183771),
            Co(392.254441, 426.006877),
            Co(793.200617, 859.602355),
            Co(-501.932232, -685.495932),
            Co(-855.5, -1438.4682),
            Co(133.731476, 217.971493),
            Co(614.799383, 664.602355),
            Co(1169.94632, -1160.5257),
        ],
        [
            Co(-419.076766, 5.2146997),
            Co(-1342.35711, 438.721352),
            Co(748.501887, -1376.29869),
            Co(-864.966302, -38.1877696),
            Co(131.548785, 293.751979),
            Co(1630.20974, 288.295123),
            Co(-373.986607, -134.314157),
            Co(-69.4833464, 813.760763),
        ],
        [
            Co(-616.5, -1391.70282),
            Co(-8.10003441, -76.8768563),
            Co(-788.863263, -689.771723),
            Co(-1795.93184, -871.801065),
            Co(506.5, -1556.24765),
            Co(-418.788682, 289.683096),
            Co(660.863263, 163.228277),
            Co(-491.17944, 1902.6073),
        ],
        [
            Co(724.076766, 315.2147),
            Co(-203.063514, -479.018638),
            Co(-878.013393, 532.685843),
            Co(523.115043, 920.439964),
            Co(623.451215, -704.248021),
            Co(1835.51316, 316.59163),
            Co(483.498113, 1192.70131),
            Co(267.032324, -251.423488),
        ],
    ],
    [
        [
            Co(185.5, 151.554446),
            Co(711.850515, 449.040354),
            Co(-749.621088, 1146.13071),
            Co(543.484425, 316.170727),
            Co(304.5, 939.637563),
            Co(261.837906, 560.891253),
            Co(-266.378912, -688.869293),
            Co(128.827155, 1919.76088),
        ],
        [
            Co(1196.86515, -34.2339376),
            Co(-356.367817, -56.7485293),
            Co(-227.692884, 810.84352),
            Co(809.0668, 410.833336),
            Co(-576.329251, -605.60185),
            Co(1398.3819, 1045.73684),
            Co(-2142.52214, -289.925626),
            Co(-313.955516, 713.619075),
        ],
        [
            Co(-458.5, -205.248021),
            Co(-163.386173, 1298.21285),
            Co(272.550672, 548.10254),
            Co(1112.89357, -166.041482),
            Co(-91.5, -532.605623),
            Co(288.800478, 1245.33676),
            Co(499.449328, -374.89746),
            Co(491.692126, 459.591094),
        ],
        [
            Co(-517.86515, -234.233938),
            Co(-728.449209, 224.279501),
            Co(48.5221358, 179.074374),
            Co(872.98747, 131.197121),
            Co(1145.32925, 70.3981502),
            Co(-636.662075, -229.82709),
            Co(241.692884, -298.15648),
            Co(26.9984468, -8.20881129),
        ],
    ],
]


def test_3d_cpu[debug: Bool]():
    comptime D1 = 6
    comptime D2 = 4
    comptime D3 = 8

    comptime x_layout = Layout.row_major(1, D1, D2, D3, 2)
    var x_buf = materialize[input_3d]()

    var x_buf2 = InlineArray[
        InlineArray[InlineArray[SIMD[DType.uint8, 2], D3], D2], D1
    ](uninitialized=True)
    for k in range(D1):
        for i in range(D2):
            for j in range(D3):
                x_buf2[k][i][j] = {x_buf[k][i][j], 0}

    var x = LayoutTensor[mut=False, DType.uint8, x_layout](
        x_buf2.unsafe_ptr().bitcast[UInt8]()
    )

    comptime out_layout = Layout.row_major(1, D1, D2, D3, 2)
    var out_buf = InlineArray[Co, D1 * D2 * D3](
        fill=Co(nan[DType.float64](), nan[DType.float64]())
    )
    var out = LayoutTensor[mut=True, DType.float64, out_layout](
        out_buf.unsafe_ptr().bitcast[Float64]()
    )

    fft(out, x)

    @parameter
    if debug:
        print("Output values:")
        for k in range(D1):
            for i in range(D2):
                for j in range(D3):
                    print(
                        "out[0, ",
                        k,
                        ", ",
                        i,
                        ", ",
                        j,
                        "]: [",
                        out[0, k, i, j, 0],
                        ", ",
                        out[0, k, i, j, 1],
                        "] expected: [",
                        expected_3d[k][i][j].re,
                        ", ",
                        expected_3d[k][i][j].im,
                        "]",
                        sep="",
                    )

    for k in range(D1):
        for i in range(D2):
            for j in range(D3):
                assert_almost_equal(
                    out[0, k, i, j, 0],
                    expected_3d[k][i][j].re,
                    String("k: ", k, " i: ", i, " j: ", j, " re"),
                )
                assert_almost_equal(
                    out[0, k, i, j, 1],
                    expected_3d[k][i][j].im,
                    String("k: ", k, " i: ", i, " j: ", j, " im"),
                )


def main():
    # test_fft()
    # test_ifft()
    # test_2d_cpu[debug=False]()
    # test_2d_gpu()
    test_3d_cpu[debug=False]()
