from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu
from layout import TileTensor, row_major
from std.math import ceil
from std.sys.info import has_accelerator, size_of
from std.utils.numerics import nan

from std.testing import assert_almost_equal

from fft.fft.fft import fft, plan_fft, _estimate_best_bases_nd
from fft.fft._ndim_fft_gpu import _run_gpu_nd_fft, _GPUTest
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

# FIXME: once we have better support for cosine in
# comptime ATOL[dtype: DType] = 1e-3 if dtype == DType.float64 else (
#     1e-2 if dtype == DType.float32 else 1e-1
# )
comptime ATOL[dtype: DType] = 1e-2
comptime RTOL = 1e-5


def _round(val: SIMD) -> type_of(val):
    return round(val, 5)


def _test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    gpu_test: Optional[_GPUTest] = None,
](debug: Bool) raises:
    comptime BATCHES = len(test_values)
    comptime SIZE = len(test_values[0][0])
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime x_dim = 2 if inverse else 1
    comptime in_layout = row_major[BATCHES, SIZE, x_dim]()
    comptime out_layout = row_major[BATCHES, SIZE, 2]()
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize
    if debug:
        print("----------------------------")
        print("SIZE:", SIZE)
        print("Buffers for Bases: ", end="")
        var b = materialize[bases]()
        print(String(b).replace("SIMD[DType.uint, 1](", "").replace(")", ""))
        print("----------------------------")

    @parameter
    def _eval(
        result: TileTensor[out_dtype, ...],
        scalar_in: List[Int],
        complex_out: List[ComplexScalar[out_dtype]],
    ) raises:
        comptime assert result.flat_rank == 2
        if debug:
            print("out: ", end="")
            for i in range(SIZE):
                if i == 0:
                    print(
                        "[",
                        _round(result[i, 0]),
                        ", ",
                        _round(result[i, 1]),
                        sep="",
                        end="",
                    )
                else:
                    print(
                        ", ",
                        _round(result[i, 0]),
                        ", ",
                        _round(result[i, 1]),
                        sep="",
                        end="",
                    )
            print("]")
            print("expected: ", end="")

        # gather all real parts and then the imaginary parts
        comptime if inverse:
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print(
                            "[",
                            _round(Scalar[DType.int](scalar_in[i])),
                            ".0, 0.0",
                            sep="",
                            end="",
                        )
                    else:
                        print(
                            ", ",
                            _round(Scalar[DType.int](scalar_in[i])),
                            ".0, 0.0",
                            sep="",
                            end="",
                        )
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    Scalar[out_dtype](scalar_in[i]),
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1], 0, atol=ATOL[out_dtype], rtol=RTOL
                )
        else:
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print(
                            "[", _round(complex_out[i].re), ", ", sep="", end=""
                        )
                    else:
                        print(
                            ", ",
                            _round(complex_out[i].re),
                            ", ",
                            sep="",
                            end="",
                        )
                    print(_round(complex_out[i].im), end="")
                print("]")
            for i in range(SIZE):
                # break
                assert_almost_equal(
                    result[i, 0],
                    complex_out[i].re.cast[out_dtype](),
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )

    comptime if target == "cpu":
        var out_data = List[Scalar[in_dtype]](
            length=out_size, fill=nan[in_dtype]()
        )
        var x_data = List[Scalar[out_dtype]](
            length=in_size, fill=nan[out_dtype]()
        )
        var batch_output = TileTensor(Span(out_data), layout=out_layout)
        var batch_x = TileTensor(Span(x_data), layout=in_layout)

        comptime batch_x_stride = in_layout.static_stride[0]
        for idx, test in enumerate(materialize[test_values]()):
            comptime x_layout = row_major[SIZE, x_dim]()
            var x = TileTensor(
                ptr=batch_x.ptr + batch_x_stride * idx, layout=x_layout
            )
            for i in range(SIZE):
                comptime if inverse:
                    x[i, 0] = test[1][i].re.cast[in_dtype]()
                    x[i, 1] = test[1][i].im.cast[in_dtype]()
                else:
                    x[i, 0] = Scalar[in_dtype](test[0][i])

        var plan = plan_fft[
            in_dtype,
            out_dtype,
            type_of(in_layout),
            type_of(out_layout),
            bases=[bases],
            inverse=inverse,
        ]()
        fft(batch_output, batch_x.as_immut(), plan=plan)

        comptime batch_out_stride = out_layout.static_stride[0]
        for idx, test in enumerate(materialize[test_values]()):
            comptime output_layout = row_major[SIZE, 2]()
            var output = TileTensor(
                ptr=batch_output.ptr + batch_out_stride * idx,
                layout=output_layout,
            )
            _eval(output, test[0], test[1])
    else:
        with DeviceContext() as ctx:
            var x_data = ctx.enqueue_create_buffer[in_dtype](in_size)
            x_data.enqueue_fill(Scalar[in_dtype].MAX)
            var out_data = ctx.enqueue_create_buffer[out_dtype](out_size)
            out_data.enqueue_fill(nan[out_dtype]())
            var batch_output = TileTensor(out_data, layout=out_layout)
            var batch_x = TileTensor(x_data, layout=in_layout)
            comptime batch_x_stride = in_layout.static_stride[0]
            with x_data.map_to_host() as x_host:
                for idx, test in enumerate(materialize[test_values]()):
                    comptime x_layout = row_major[SIZE, x_dim]()
                    var x = TileTensor(
                        ptr=x_host.unsafe_ptr() + batch_x_stride * idx,
                        layout=x_layout,
                    )

                    for i in range(SIZE):
                        comptime if inverse:
                            x[i, 0] = test[1][i].re.cast[in_dtype]()
                            x[i, 1] = test[1][i].im.cast[in_dtype]()
                        else:
                            x[i, 0] = Scalar[in_dtype](test[0][i])

            ctx.synchronize()
            var plan = plan_fft[
                in_dtype,
                out_dtype,
                type_of(in_layout),
                type_of(out_layout),
                bases=[bases],
                inverse=inverse,
                runtime_twfs=True,
                _test=gpu_test,
            ](ctx=ctx)
            _run_gpu_nd_fft(batch_output, batch_x, ctx, plan=plan)
            ctx.synchronize()
            with out_data.map_to_host() as out_host:
                comptime batch_out_stride = out_layout.static_stride[0]
                for idx, test in enumerate(materialize[test_values]()):
                    comptime output_layout = row_major[SIZE, 2]()
                    var output = TileTensor(
                        ptr=out_host.unsafe_ptr() + batch_out_stride * idx,
                        layout=output_layout,
                    )
                    _eval(output, test[0], test[1])

    if debug:
        print("----------------------------")
        print("Tests passed")
        print("----------------------------")


def _test_fft[
    dtype: DType,
    func: def[bases: List[UInt], test_values: _TestValues[dtype]](
        debug: Bool
    ) thin raises,
](debug: Bool) raises:
    comptime L = List[UInt]

    comptime values_2 = _get_test_values_2[dtype]()
    func[[2], values_2](debug)

    comptime values_3 = _get_test_values_3[dtype]()
    func[[3], values_3](debug)

    comptime values_4 = _get_test_values_4[dtype]()
    func[[4], values_4](debug)
    func[[2], values_4](debug)

    comptime values_5 = _get_test_values_5[dtype]()
    func[[5], values_5](debug)

    comptime values_6 = _get_test_values_6[dtype]()
    func[[6], values_6](debug)
    func[[3, 2], values_6](debug)
    func[[2, 3], values_6](debug)

    comptime values_7 = _get_test_values_7[dtype]()
    func[[7], values_7](debug)

    comptime values_8 = _get_test_values_8[dtype]()
    func[[8], values_8](debug)
    func[[2], values_8](debug)
    func[[4, 2], values_8](debug)
    func[[2, 4], values_8](debug)

    comptime values_10 = _get_test_values_10[dtype]()
    func[[10], values_10](debug)
    func[[5, 2], values_10](debug)

    comptime values_16 = _get_test_values_16[dtype]()
    func[[16], values_16](debug)
    func[[2], values_16](debug)
    func[[4], values_16](debug)
    func[[2, 4], values_16](debug)
    func[[8, 2], values_16](debug)
    func[[2, 8], values_16](debug)

    comptime values_20 = _get_test_values_20[dtype]()
    func[[20], values_20](debug)
    func[[10, 2], values_20](debug)
    func[[5, 4], values_20](debug)
    func[[5, 2], values_20](debug)

    comptime values_21 = _get_test_values_21[dtype]()
    func[[7, 3], values_21](debug)

    comptime values_32 = _get_test_values_32[dtype]()
    func[[2], values_32](debug)
    func[[16, 2], values_32](debug)
    func[[8, 4], values_32](debug)
    func[[4, 4, 2], values_32](debug)
    func[[8, 2, 2], values_32](debug)

    comptime values_35 = _get_test_values_35[dtype]()
    func[[7, 5], values_35](debug)

    comptime values_48 = _get_test_values_48[dtype]()
    func[[8, 6], values_48](debug)
    func[[3, 2], values_48](debug)

    comptime values_60 = _get_test_values_60[dtype]()
    func[[10, 6], values_60](debug)
    func[[6, 5, 2], values_60](debug)
    func[[5, 4, 3], values_60](debug)
    func[[3, 4, 5], values_60](debug)
    func[[5, 3, 2], values_60](debug)

    comptime values_64 = _get_test_values_64[dtype]()
    func[[2], values_64](debug)
    func[[8], values_64](debug)
    func[[4], values_64](debug)
    func[[16, 4], values_64](debug)

    comptime values_100 = _get_test_values_100[dtype]()
    func[[20, 5], values_100](debug)
    func[[10], values_100](debug)
    func[[5, 4], values_100](debug)

    comptime values_128 = _get_test_values_128[dtype]()
    # func[[32, 4], values_128](debug)  # long compile times, but important to test
    func[[16, 8], values_128](debug)
    func[[16, 4, 2], values_128](debug)
    func[[8, 8, 2], values_128](debug)
    func[[8, 4, 4], values_128](debug)
    func[[8, 4, 2, 2], values_128](debug)
    func[[8, 2, 2, 2, 2], values_128](debug)
    func[[4, 4, 4, 2], values_128](debug)
    func[[4, 4, 2, 2, 2], values_128](debug)
    func[[4, 2, 2, 2, 2, 2], values_128](debug)
    func[[2], values_128](debug)


comptime _test[
    dtype: DType,
    inverse: Bool,
    target: StaticString,
    gpu_test: Optional[_GPUTest] = None,
] = _test_fft[
    dtype,
    _test_fft_radix_n[
        dtype,
        inverse=inverse,
        target=target,
        gpu_test=gpu_test,
        ...,
    ],
]


def test_fft_1d_cpu(debug: Bool = False) raises:
    comptime dtype = DType.float64
    _test[dtype, False, "cpu"](debug)


def test_fft_1d_gpu(debug: Bool = False) raises:
    comptime dtype = DType.float64
    _test[dtype, False, "gpu", gpu_test=_GPUTest.BLOCK](debug)
    # _test[dtype, False, "gpu", gpu_test = _GPUTest.WARP](debug)
    # _test[dtype, False, "gpu", gpu_test = _GPUTest.DEVICE_WIDE](debug)
    # _test[dtype, False, "gpu", gpu_test = _GPUTest.CLUSTER](debug)


def test_ifft_1d_cpu(debug: Bool = False) raises:
    comptime dtype = DType.float64
    _test[dtype, True, "cpu"](debug)


def test_ifft_1d_gpu(debug: Bool = False) raises:
    comptime dtype = DType.float64
    _test[dtype, True, "cpu"](debug)
    _test[dtype, True, "gpu", gpu_test=_GPUTest.BLOCK](debug)
    # _test[dtype, True, "gpu", gpu_test=_GPUTest.WARP](debug)
    # _test[dtype, True, "gpu", gpu_test=_GPUTest.DEVICE_WIDE](debug)
    # _test[dtype, True, "gpu", gpu_test=_GPUTest.CLUSTER](debug)


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


def test_2d_cpu[debug: Bool = False]() raises:
    comptime ROWS = 6
    comptime COLS = 4

    comptime x_layout = row_major[1, ROWS, COLS, 1]()
    ref x_buf = global_constant[input_2d]()
    var x = TileTensor(ptr=x_buf.unsafe_ptr().unsafe_bitcast[UInt8](), layout=x_layout)

    comptime out_layout = row_major[1, ROWS, COLS, 2]()
    comptime out_dtype = DType.float64
    var out_buf = InlineArray[Co, ROWS * COLS](
        fill=Co(nan[out_dtype](), nan[out_dtype]())
    )
    var out = TileTensor(
        ptr=UnsafePointer(to=out_buf[0]).unsafe_bitcast[Float64](),
        layout=out_layout,
    )
    var plan = plan_fft[
        DType.uint8, out_dtype, type_of(x_layout), type_of(out_layout)
    ]()
    fft(out, x, plan=plan)

    ref expected = global_constant[expected_2d]()

    if debug:
        print("Values:")
        for i in range(ROWS):
            for j in range(COLS):
                print(
                    "out[0, ",
                    i,
                    ", ",
                    j,
                    "]: [",
                    _round(out[0, i, j, 0]),
                    ", ",
                    _round(out[0, i, j, 1]),
                    "] expected: [",
                    expected[i][j].re,
                    ", ",
                    expected[i][j].im,
                    "]",
                    sep="",
                )

    for i in range(ROWS):
        for j in range(COLS):
            assert_almost_equal(
                out[0, i, j, 0],
                expected[i][j].re,
                atol=ATOL[out_dtype],
                rtol=RTOL,
            )
            assert_almost_equal(
                out[0, i, j, 1],
                expected[i][j].im,
                atol=ATOL[out_dtype],
                rtol=RTOL,
            )


def _test_2d_gpu[inverse: Bool, gpu_test: _GPUTest](debug: Bool) raises:
    comptime ROWS = 6
    comptime COLS = 4
    comptime in_dtype = DType.uint8
    comptime out_dtype = DType.float64
    comptime in_layout = row_major[1, ROWS, COLS, 1]()
    comptime out_layout = row_major[1, ROWS, COLS, 2]()
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize

    with DeviceContext() as ctx:
        var x_data = ctx.enqueue_create_buffer[in_dtype](in_size)
        x_data.enqueue_fill(Scalar[in_dtype].MAX)
        var out_data = ctx.enqueue_create_buffer[out_dtype](out_size)
        out_data.enqueue_fill(nan[out_dtype]())
        var out = TileTensor(out_data, layout=out_layout)
        var x = TileTensor(x_data, layout=in_layout)

        ref input_2d_v = global_constant[input_2d]()

        with x_data.map_to_host() as x_host:
            var x_view = TileTensor(x_host, layout=in_layout)

            for i in range(ROWS):
                for j in range(COLS):
                    x_view[0, i, j, 0] = Scalar[in_dtype](input_2d_v[i][j])

        ctx.synchronize()
        var plan = plan_fft[
            in_dtype,
            out_dtype,
            type_of(in_layout),
            type_of(out_layout),
            inverse=inverse,
            _test=gpu_test,
            runtime_twfs=True,
        ](ctx=ctx)
        _run_gpu_nd_fft(out, x.as_immut(), ctx, plan=plan)
        ctx.synchronize()

        ref expected = global_constant[expected_2d]()

        with out_data.map_to_host() as out_host:
            var out_view = TileTensor(out_host, layout=out_layout)

            if debug:
                print("Values:")
                for i in range(ROWS):
                    for j in range(COLS):
                        print(
                            "out[0, ",
                            i,
                            ", ",
                            j,
                            "]: [",
                            _round(out_view[0, i, j, 0]),
                            ", ",
                            _round(out_view[0, i, j, 1]),
                            "] expected: [",
                            expected[i][j].re,
                            ", ",
                            expected[i][j].im,
                            "]",
                            sep="",
                        )

            for i in range(ROWS):
                for j in range(COLS):
                    assert_almost_equal(
                        out_view[0, i, j, 0],
                        expected[i][j].re,
                        atol=ATOL[out_dtype],
                        rtol=RTOL,
                    )
                    assert_almost_equal(
                        out_view[0, i, j, 1],
                        expected[i][j].im,
                        atol=ATOL[out_dtype],
                        rtol=RTOL,
                    )


def test_2d_gpu(debug: Bool = False) raises:
    _test_2d_gpu[False, _GPUTest.BLOCK](debug)
    # _test_2d_gpu[False, _GPUTest.WARP](debug)
    # _test_2d_gpu[False, _GPUTest.DEVICE_WIDE](debug)
    # _test_2d_gpu[False, _GPUTest.CLUSTER](debug)


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


def test_3d_cpu[debug: Bool = False]() raises:
    comptime D1 = 6
    comptime D2 = 4
    comptime D3 = 8

    comptime x_layout = row_major[1, D1, D2, D3, 1]()
    ref x_buf = global_constant[input_3d]()
    var x = TileTensor(ptr=x_buf.unsafe_ptr().unsafe_bitcast[UInt8](), layout=x_layout)

    comptime out_layout = row_major[1, D1, D2, D3, 2]()
    comptime out_dtype = DType.float64
    comptime n = nan[out_dtype]()
    var out_buf = InlineArray[Co, D1 * D2 * D3](fill=Co(n, n))
    var out = TileTensor(
        ptr=UnsafePointer(to=out_buf[0]).unsafe_bitcast[Float64](),
        layout=out_layout,
    )

    var plan = plan_fft[
        DType.uint8, out_dtype, type_of(x_layout), type_of(out_layout)
    ]()
    fft(out, x, plan=plan)

    ref expected = global_constant[expected_3d]()

    if debug:
        print("Values:")
        for i in range(D1):
            for j in range(D2):
                for k in range(D3):
                    print(
                        "out[0, ",
                        i,
                        ", ",
                        j,
                        ", ",
                        k,
                        "]: [",
                        _round(out[0, i, j, k, 0]),
                        ", ",
                        _round(out[0, i, j, k, 1]),
                        "] expected: [",
                        expected[i][j][k].re,
                        ", ",
                        expected[i][j][k].im,
                        "]",
                        sep="",
                    )

    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                assert_almost_equal(
                    out[0, i, j, k, 0],
                    expected[i][j][k].re,
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )
                assert_almost_equal(
                    out[0, i, j, k, 1],
                    expected[i][j][k].im,
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )


def _test_3d_gpu[inverse: Bool, gpu_test: _GPUTest](debug: Bool) raises:
    comptime D1 = 6
    comptime D2 = 4
    comptime D3 = 8
    comptime in_dtype = DType.uint8
    comptime out_dtype = DType.float64
    comptime in_layout = row_major[1, D1, D2, D3, 1]()
    comptime out_layout = row_major[1, D1, D2, D3, 2]()
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize

    with DeviceContext() as ctx:
        var x_data = ctx.enqueue_create_buffer[in_dtype](in_size)
        x_data.enqueue_fill(Scalar[in_dtype].MAX)
        var out_data = ctx.enqueue_create_buffer[out_dtype](out_size)
        out_data.enqueue_fill(nan[out_dtype]())
        var out = TileTensor(out_data, layout=out_layout)
        var x = TileTensor(x_data, layout=in_layout)

        ref input_3d_v = global_constant[input_3d]()

        with x_data.map_to_host() as x_host:
            var x_view = TileTensor(x_host, layout=in_layout)

            for i in range(D1):
                for j in range(D2):
                    for k in range(D3):
                        x_view[0, i, j, k, 0] = Scalar[in_dtype](
                            input_3d_v[i][j][k]
                        )

        ctx.synchronize()
        var plan = plan_fft[
            in_dtype,
            out_dtype,
            type_of(in_layout),
            type_of(out_layout),
            inverse=inverse,
            _test=gpu_test,
            runtime_twfs=True,
        ](ctx=ctx)
        _run_gpu_nd_fft(out, x.as_immut(), ctx, plan=plan)
        ctx.synchronize()

        ref expected = global_constant[expected_3d]()

        with out_data.map_to_host() as out_host:
            var out_view = TileTensor(out_host, layout=out_layout)

            if debug:
                print("Values:")

                for i in range(D1):
                    for j in range(D2):
                        for k in range(D3):
                            print(
                                "out[0, ",
                                i,
                                ", ",
                                j,
                                ", ",
                                k,
                                "]: [",
                                _round(out_view[0, i, j, k, 0]),
                                ", ",
                                _round(out_view[0, i, j, k, 1]),
                                "] expected: [",
                                expected[i][j][k].re,
                                ", ",
                                expected[i][j][k].im,
                                "]",
                                sep="",
                            )

            for i in range(D1):
                for j in range(D2):
                    for k in range(D3):
                        assert_almost_equal(
                            out_view[0, i, j, k, 0],
                            expected[i][j][k].re,
                            atol=ATOL[out_dtype],
                            rtol=RTOL,
                        )
                        assert_almost_equal(
                            out_view[0, i, j, k, 1],
                            expected[i][j][k].im,
                            atol=ATOL[out_dtype],
                            rtol=RTOL,
                        )


def test_3d_gpu(debug: Bool = False) raises:
    _test_3d_gpu[False, _GPUTest.BLOCK](debug)
    # _test_3d_gpu[False, _GPUTest.WARP](debug)
    # _test_3d_gpu[False, _GPUTest.DEVICE_WIDE](debug)
    # _test_3d_gpu[False, _GPUTest.CLUSTER](debug)


def main() raises:
    test_fft_1d_cpu()
    # test_ifft_1d_cpu()
    test_2d_cpu()
    test_3d_cpu()

    # GPU tests require a known accelerator target; skip when none is present.
    comptime _run_gpu = has_accelerator()
    comptime if _run_gpu:
        test_fft_1d_gpu()
        # test_ifft_1d_gpu()
        test_2d_gpu()
        test_3d_gpu()
