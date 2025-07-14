from complex import ComplexSIMD
from layout import Layout, LayoutTensor
from math import log2
from gpu.host import DeviceContext

from testing import assert_almost_equal


from _test_values import _get_test_values_8, _get_test_values_4
from main import (
    _intra_block_fft_launch,
    _intra_block_fft_launch_radix_n,
)


def test_intra_block_radix_2_with_8_samples():
    alias SIZE = 8
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
        for test in _get_test_values_8[calc_dtype]():
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
            _intra_block_fft_launch[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                threads_per_block=SIZE,
                blocks_per_grid=1,
            ](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected:", end=" ")
                first = True
                # gather all real parts and then the imaginary parts
                for item in expected:
                    if not first:
                        print(",", item.re, end="")
                    else:
                        first = False
                        print("[", end="")
                        print(item.re, end="")
                for item in expected:
                    if not first:
                        print(",", item.im, end="")
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


def test_intra_block_radix_n_with_8_samples[base: UInt]():
    alias SIZE = 8
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
        for test in _get_test_values_8[calc_dtype]():
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
            _intra_block_fft_launch_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                threads_per_block=SIZE,
                blocks_per_grid=1,
                base=base,
            ](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected:", end=" ")
                first = True
                # gather all real parts and then the imaginary parts
                for item in expected:
                    if not first:
                        print(",", item.re, end="")
                    else:
                        first = False
                        print("[", end="")
                        print(item.re, end="")
                for item in expected:
                    if not first:
                        print(",", item.im, end="")
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


def test_intra_block_radix_2_with_4_samples():
    alias SIZE = 4
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
        for test in _get_test_values_4[calc_dtype]():
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
            _intra_block_fft_launch[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                threads_per_block=SIZE,
                blocks_per_grid=1,
            ](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected:", end=" ")
                first = True
                # gather all real parts and then the imaginary parts
                for item in expected:
                    if not first:
                        print(",", item.re, end="")
                    else:
                        first = False
                        print("[", end="")
                        print(item.re, end="")
                for item in expected:
                    if not first:
                        print(",", item.im, end="")
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


def test_intra_block_radix_n_with_4_samples[base: UInt]():
    alias SIZE = 4
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
        for test in _get_test_values_4[calc_dtype]():
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
            _intra_block_fft_launch_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                threads_per_block=SIZE,
                blocks_per_grid=1,
                base=base,
            ](out_tensor, x_tensor, ctx)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                print("out:", out_host)
                print("expected:", end=" ")
                first = True
                # gather all real parts and then the imaginary parts
                for item in expected:
                    if not first:
                        print(",", item.re, end="")
                    else:
                        first = False
                        print("[", end="")
                        print(item.re, end="")
                for item in expected:
                    if not first:
                        print(",", item.im, end="")
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
