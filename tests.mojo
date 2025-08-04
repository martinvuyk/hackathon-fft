from complex import ComplexSIMD
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext

from testing import assert_almost_equal

from fft import fft
from _test_values import (
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


def test_fft_radix_n[bases: List[UInt], test_values: _TestValues]():
    alias SIZE = len(test_values[0][0])
    alias TPB = SIZE
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = DType.float64
    alias out_dtype = DType.float64
    alias in_layout = Layout.row_major(SIZE)
    alias out_layout = Layout.row_major(SIZE, 2)
    alias calc_dtype = DType.float64
    alias Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[out_dtype](SIZE * 2).enqueue_fill(0)
        x = ctx.enqueue_create_buffer[in_dtype](SIZE).enqueue_fill(0)
        print("----------------------------")
        print("Buffers")
        print("----------------------------")
        for test in test_values:
            ref series = test[0]
            ref expected = test[1]
            with x.map_to_host() as x_host:
                for i in range(SIZE):
                    x_host[i] = series[i]

            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                out.unsafe_ptr()
            )
            var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
                x.unsafe_ptr()
            )
            fft[bases=bases](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected:", end=" ")
                # gather all real parts and then the imaginary parts
                for i in range(SIZE):
                    if i > 0:
                        print(", ", expected[i].re, ",", sep="", end=" ")
                    else:
                        print("[", end="")
                        print(expected[i].re, ",", sep="", end=" ")
                    print(expected[i].im, end="")
                print("]")
                for i in range(SIZE):
                    assert_almost_equal(
                        out_host[2 * i],
                        expected[i].re.cast[out_dtype](),
                        atol=1e-3,
                        rtol=1e-5,
                    )
                    assert_almost_equal(
                        out_host[2 * i + 1],
                        expected[i].im.cast[out_dtype](),
                        atol=1e-3,
                        rtol=1e-5,
                    )
        print("----------------------------")
        print("Tests passed")
        print("----------------------------")


def test_ifft_radix_n[bases: List[UInt], test_values: _TestValues]():
    alias SIZE = len(test_values[0][0])
    alias TPB = SIZE
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = DType.float64
    alias out_dtype = DType.float64
    alias in_layout = Layout.row_major(SIZE, 2)
    alias out_layout = Layout.row_major(SIZE, 2)
    alias calc_dtype = DType.float64
    alias Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        x = ctx.enqueue_create_buffer[in_dtype](SIZE * 2).enqueue_fill(0)
        out = ctx.enqueue_create_buffer[out_dtype](SIZE * 2).enqueue_fill(0)
        print("----------------------------")
        print("Buffers")
        print("----------------------------")
        for test in test_values:
            ref expected = test[0]
            ref series = test[1]
            with x.map_to_host() as x_host:
                for i in range(SIZE):
                    x_host[i * 2] = series[i].re.cast[in_dtype]()
                    x_host[i * 2 + 1] = series[i].im.cast[in_dtype]()

            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                out.unsafe_ptr()
            )
            var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
                x.unsafe_ptr()
            )
            fft[bases=bases, inverse=True](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected: ", end="")
                # gather all real parts and then the imaginary parts
                for i in range(SIZE):
                    if i > 0:
                        print(", ", expected[i], ",", sep="", end=" ")
                    else:
                        print("[", end="")
                        print(expected[i], ",", sep="", end=" ")
                    print(0, end="")
                print("]")
                for i in range(SIZE):
                    assert_almost_equal(
                        out_host[2 * i],
                        Scalar[out_dtype](expected[i]),
                        atol=1e-3,
                        rtol=1e-5,
                    )
                    assert_almost_equal(
                        out_host[2 * i + 1], 0, atol=1e-3, rtol=1e-5
                    )
        print("----------------------------")
        print("Tests passed")
        print("----------------------------")


def _test_fft[
    func: fn[bases: List[UInt], test_values: _TestValues] () raises
]():
    alias L = List[UInt]

    alias values_2 = _get_test_values_2[DType.float64]()
    func[L(2), values_2]()

    alias values_3 = _get_test_values_3[DType.float64]()
    func[L(3), values_3]()

    alias values_4 = _get_test_values_4[DType.float64]()
    func[L(4), values_4]()
    func[L(2), values_4]()

    alias values_5 = _get_test_values_5[DType.float64]()
    func[L(5), values_5]()

    alias values_6 = _get_test_values_6[DType.float64]()
    func[L(6), values_6]()
    func[L(3, 2), values_6]()
    func[L(2, 3), values_6]()

    alias values_7 = _get_test_values_7[DType.float64]()
    func[L(7), values_7]()

    alias values_8 = _get_test_values_8[DType.float64]()
    func[L(8), values_8]()
    func[L(2), values_8]()
    func[L(4, 2), values_8]()
    func[L(2, 4), values_8]()

    alias values_10 = _get_test_values_10[DType.float64]()
    func[L(10), values_10]()
    func[L(5, 2), values_10]()

    alias values_16 = _get_test_values_16[DType.float64]()
    func[L(16), values_16]()
    func[L(2), values_16]()
    func[L(4), values_16]()
    func[L(2, 4), values_16]()
    func[L(8, 2), values_16]()
    func[L(2, 8), values_16]()

    alias values_20 = _get_test_values_20[DType.float64]()
    func[L(10, 2), values_20]()
    func[L(5, 4), values_20]()
    func[L(5, 2), values_20]()

    alias values_21 = _get_test_values_21[DType.float64]()
    func[L(7, 3), values_21]()

    alias values_32 = _get_test_values_32[DType.float64]()
    func[L(2), values_32]()
    func[L(16, 2), values_32]()
    func[L(8, 4), values_32]()
    func[L(4, 2), values_32]()
    func[L(8, 2), values_32]()

    alias values_35 = _get_test_values_35[DType.float64]()
    func[L(7, 5), values_35]()

    alias values_48 = _get_test_values_48[DType.float64]()
    func[L(8, 6), values_48]()
    func[L(3, 2), values_48]()

    alias values_60 = _get_test_values_60[DType.float64]()
    func[L(10, 6), values_60]()
    func[L(6, 5, 2), values_60]()
    func[L(5, 4, 3), values_60]()
    func[L(3, 4, 5), values_60]()
    func[L(5, 3, 2), values_60]()

    alias values_64 = _get_test_values_64[DType.float64]()
    func[L(2), values_64]()
    func[L(8), values_64]()
    func[L(4), values_64]()
    func[L(16, 4), values_64]()

    alias values_100 = _get_test_values_100[DType.float64]()
    func[L(20, 5), values_100]()
    func[L(10), values_100]()
    func[L(5, 4), values_100]()

    alias values_128 = _get_test_values_128[DType.float64]()
    func[L(2), values_128]()
    func[L(16, 8), values_128]()


def test_fft():
    _test_fft[test_fft_radix_n]()


def test_ifft():
    _test_fft[test_ifft_radix_n]()


def main():
    test_fft()
    test_ifft()
