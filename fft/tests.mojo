from builtin.globals import global_constant
from complex import ComplexScalar
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of
from utils.numerics import nan

from testing import assert_almost_equal

from fft.fft.fft import fft, _estimate_best_bases_nd
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


def _test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    debug: Bool,
    gpu_test: Optional[_GPUTest] = None,
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
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1], 0, atol=ATOL[out_dtype], rtol=RTOL
                )
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
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=ATOL[out_dtype],
                    rtol=RTOL,
                )

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
        var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](Span(x_data))

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

        fft[bases= [bases], inverse=inverse](batch_output, batch_x)

        for idx, test in enumerate(materialize[test_values]()):
            comptime output_layout = Layout.row_major(
                out_layout.shape[1].value(), 2
            )
            var output = LayoutTensor[
                mut=True, out_dtype, output_layout, batch_output.origin
            ](batch_output.ptr + batch_output.stride[0]() * idx)
            _eval(output, test[0], test[1])
    else:
        with DeviceContext() as ctx:
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
            _run_gpu_nd_fft[inverse=inverse, bases= [bases], test=gpu_test](
                batch_output, batch_x, ctx
            )
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
    func[[4, 4, 2], values_32]()
    func[[8, 2, 2], values_32]()

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
    debug: Bool,
    gpu_test: Optional[_GPUTest] = None,
] = _test_fft[
    dtype,
    _test_fft_radix_n[
        dtype, inverse=inverse, target=target, gpu_test=gpu_test, debug=debug
    ],
]


def test_fft[debug: Bool = False]():
    comptime dtype = DType.float64
    _test[dtype, False, "cpu", debug=debug]()
    _test[dtype, False, "gpu", debug=debug, gpu_test = _GPUTest.BLOCK]()
    _test[dtype, False, "gpu", debug=debug, gpu_test = _GPUTest.WARP]()
    _test[dtype, False, "gpu", debug=debug, gpu_test = _GPUTest.DEVICE_WIDE]()
    _test[dtype, False, "gpu", debug=debug, gpu_test = _GPUTest.CLUSTER]()


def test_ifft[debug: Bool = False]():
    comptime dtype = DType.float64
    _test[dtype, True, "cpu", debug=debug]()
    _test[dtype, True, "gpu", debug=debug, gpu_test = _GPUTest.BLOCK]()
    _test[dtype, True, "gpu", debug=debug, gpu_test = _GPUTest.WARP]()
    _test[dtype, True, "gpu", debug=debug, gpu_test = _GPUTest.DEVICE_WIDE]()
    _test[dtype, True, "gpu", debug=debug, gpu_test = _GPUTest.CLUSTER]()


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


def test_2d_cpu[debug: Bool = False]():
    comptime ROWS = 6
    comptime COLS = 4

    comptime x_layout = Layout.row_major(1, ROWS, COLS, 1)
    ref x_buf = global_constant[input_2d]()
    var x = LayoutTensor[mut=False, DType.uint8, x_layout](
        x_buf.unsafe_ptr().bitcast[UInt8]()
    )

    comptime out_layout = Layout.row_major(1, ROWS, COLS, 2)
    comptime out_dtype = DType.float64
    var out_buf = InlineArray[Co, ROWS * COLS](
        fill=Co(nan[out_dtype](), nan[out_dtype]())
    )
    var out = LayoutTensor[mut=True, out_dtype, out_layout](
        out_buf.unsafe_ptr().bitcast[Float64]()
    )

    fft(out, x)

    ref expected = global_constant[expected_2d]()

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


def _test_2d_gpu[debug: Bool, inverse: Bool, gpu_test: _GPUTest]():
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

        ref input_2d_v = global_constant[input_2d]()

        with x_data.map_to_host() as x_host:
            var x_view = type_of(x)(x_host.unsafe_ptr())

            for i in range(ROWS):
                for j in range(COLS):
                    x_view[0, i, j, 0] = Scalar[in_dtype](input_2d_v[i][j])

        ctx.synchronize()
        comptime bases = _estimate_best_bases_nd[in_layout, out_layout, "gpu"]()
        _run_gpu_nd_fft[inverse=inverse, bases=bases, test=gpu_test](
            out, x.get_immutable(), ctx
        )
        ctx.synchronize()

        ref expected = global_constant[expected_2d]()

        with out_data.map_to_host() as out_host:
            var out_view = type_of(out)(out_host.unsafe_ptr())

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
                            out_view[0, i, j, 0],
                            ", ",
                            out_view[0, i, j, 1],
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


def test_2d_gpu[debug: Bool = False]():
    _test_2d_gpu[debug, False, _GPUTest.BLOCK]()
    _test_2d_gpu[debug, False, _GPUTest.WARP]()
    _test_2d_gpu[debug, False, _GPUTest.DEVICE_WIDE]()
    _test_2d_gpu[debug, False, _GPUTest.CLUSTER]()


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


def test_3d_cpu[debug: Bool = False]():
    comptime D1 = 6
    comptime D2 = 4
    comptime D3 = 8

    comptime x_layout = Layout.row_major(1, D1, D2, D3, 1)
    ref x_buf = global_constant[input_3d]()
    var x = LayoutTensor[mut=False, DType.uint8, x_layout](
        x_buf.unsafe_ptr().bitcast[UInt8]()
    )

    comptime out_layout = Layout.row_major(1, D1, D2, D3, 2)
    comptime out_dtype = DType.float64
    comptime n = nan[out_dtype]()
    var out_buf = InlineArray[Co, D1 * D2 * D3](fill=Co(n, n))
    var out = LayoutTensor[mut=True, out_dtype, out_layout](
        out_buf.unsafe_ptr().bitcast[Float64]()
    )

    fft(out, x)

    ref expected = global_constant[expected_3d]()

    @parameter
    if debug:
        print("Output values:")
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
                        out[0, i, j, k, 0],
                        ", ",
                        out[0, i, j, k, 1],
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


def _test_3d_gpu[debug: Bool, inverse: Bool, gpu_test: _GPUTest]():
    comptime D1 = 6
    comptime D2 = 4
    comptime D3 = 8
    comptime in_dtype = DType.uint8
    comptime out_dtype = DType.float64
    comptime in_layout = Layout.row_major(1, D1, D2, D3, 1)
    comptime out_layout = Layout.row_major(1, D1, D2, D3, 2)

    with DeviceContext() as ctx:
        var x_data = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
        x_data.enqueue_fill(Scalar[in_dtype].MAX)
        var out_data = ctx.enqueue_create_buffer[out_dtype](out_layout.size())
        out_data.enqueue_fill(nan[out_dtype]())

        var out = LayoutTensor[mut=True, out_dtype, out_layout](
            out_data.unsafe_ptr()
        )
        var x = LayoutTensor[mut=True, in_dtype, in_layout](x_data.unsafe_ptr())

        ref input_3d_v = global_constant[input_3d]()

        with x_data.map_to_host() as x_host:
            var x_view = type_of(x)(x_host.unsafe_ptr())

            for i in range(D1):
                for j in range(D2):
                    for k in range(D3):
                        x_view[0, i, j, k, 0] = Scalar[in_dtype](
                            input_3d_v[i][j][k]
                        )

        ctx.synchronize()
        comptime bases = _estimate_best_bases_nd[in_layout, out_layout, "gpu"]()
        _run_gpu_nd_fft[inverse=inverse, bases=bases, test=gpu_test](
            out, x.get_immutable(), ctx
        )
        ctx.synchronize()

        ref expected = global_constant[expected_3d]()

        with out_data.map_to_host() as out_host:
            var out_view = type_of(out)(out_host.unsafe_ptr())

            @parameter
            if debug:
                print("Output values:")

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
                                out_view[0, i, j, k, 0],
                                ", ",
                                out_view[0, i, j, k, 1],
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


def test_3d_gpu[debug: Bool = False]():
    _test_3d_gpu[debug, False, _GPUTest.BLOCK]()
    # _test_3d_gpu[debug, False, _GPUTest.WARP]()
    # _test_3d_gpu[debug, False, _GPUTest.DEVICE_WIDE]()
    # _test_3d_gpu[debug, False, _GPUTest.CLUSTER]()


def main():
    # test_fft()
    # test_ifft()
    # test_2d_cpu()
    # test_2d_gpu()
    # test_3d_cpu()
    test_3d_gpu()
