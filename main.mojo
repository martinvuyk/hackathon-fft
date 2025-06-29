from bit import bit_reverse
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import log2, exp, pi, cos, sin, iota
from sys import sizeof, argv

from testing import assert_almost_equal
from _test_values import _get_test_values_8


fn _get_ordered_items[stages: UInt](out res: InlineArray[UInt, 2**stages]):
    """The Butterfly diagram orders indexes by bit-reversed size."""
    res = InlineArray[UInt, 2**stages](uninitialized=True)
    values = List[UInt](capacity=2**stages)
    for i in range(2**stages):
        values.append(bit_reverse(i))
    sort(values)
    for i in range(2**stages):
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


fn fast_fourier_transform[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    threads_per_block: Int,
    calc_dtype: DType = DType.float64,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    twiddle_factors: InlineArray[
        ComplexSIMD[calc_dtype, 1], in_layout.shape[0].value() - 1
    ],
):
    constrained[len(in_layout) == 1, "in_layout must have only 1 axis"]()
    alias length = in_layout.shape[0].value()
    constrained[
        length.is_power_of_two(), "input sequence length must be a power of two"
    ]()
    constrained[
        out_layout.shape == IntTuple(2, length),
        "out_layout shape must be (2, in_layout.shape[0])",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()
    constrained[
        calc_dtype.is_floating_point(), "calc_dtype must be floating point"
    ]()
    # TODO: maybe constraint layout on column major for output write perf.
    # but it might also be faster the other way around. so find out which

    alias stages = UInt(log2(Float64(length)).cast[DType.index]())
    alias offsets: InlineArray[UInt, stages] = _get_offsets[stages]()
    alias ordered_items = _get_ordered_items[stages]()
    alias Complex = ComplexSIMD[calc_dtype, _]
    # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
    # alias twiddle_factors: InlineArray[
    #     Complex[1], length - 1
    # ] = _get_twiddle_factors[stages, length, calc_dtype]()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # complex vectors for the frequencies
    shared_f = (
        tb[calc_dtype]().row_major[2, threads_per_block]().shared().alloc()
    )

    # reorder input x(local_i) items to match F(current_item) layout
    current_item = ordered_items[local_i]
    shared_f[0, current_item] = x[global_i].cast[calc_dtype]()
    shared_f[1, current_item] = 0  # imaginary part

    barrier()

    @parameter
    for stage in range(stages):
        next_idx = local_i + offsets[stage]
        # Run the Danielson-Lanczos Lemma if the current local_i is an "x_0"
        # line in the butterfly diagram. "x_0" would mean that the given
        # item is the lhs term that always gets added without any multiplication
        is_execution_thread = False
        if next_idx < length:
            alias value = 2**stage
            delta = ordered_items[local_i] + value - ordered_items[next_idx]
            delta_0_reference = (
                ordered_items[0] + value - ordered_items[offsets[stage]]
            )
            # only "x_0" paths get the same delta as the 0th line
            is_execution_thread = delta == delta_0_reference

        if is_execution_thread:
            # get the twiddle factor W
            twiddle_idx = UInt(local_i % 2**stage)
            # base_idx is the offset to the end of the previous stage
            alias base_idx: UInt = (2 ** (stage + 1)) // 2 - 1
            twiddle_factor = twiddle_factors[base_idx + twiddle_idx]
            twiddle_factor_vec = Complex[shared_f.element_size](
                twiddle_factor.re, twiddle_factor.im
            )

            # get the upper and lower paths in the butterfly diagram
            x_0 = Complex(shared_f[0, local_i], shared_f[1, local_i])
            x_1 = Complex(shared_f[0, next_idx], shared_f[1, next_idx])

            # f_0 = x_0 + W * x_1
            res_0 = twiddle_factor_vec.fma(x_1, x_0)
            shared_f[0, local_i] = res_0.re
            shared_f[1, local_i] = res_0.im
            # f_1 = x_0 - W * x_1
            res_1 = (-twiddle_factor_vec).fma(x_1, x_0)
            shared_f[0, next_idx] = res_1.re
            shared_f[1, next_idx] = res_1.im
        barrier()

    output[0, global_i] = shared_f[0, local_i].cast[out_dtype]()
    output[1, global_i] = shared_f[1, local_i].cast[out_dtype]()


def main():
    alias TPB = 8
    alias SIZE = 8
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = DType.float64
    alias out_dtype = DType.float64
    alias in_layout = Layout.row_major(SIZE)
    alias out_layout = Layout.row_major(2, SIZE)
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

            var out_tensor = LayoutTensor[mut=False, out_dtype, out_layout](
                out.unsafe_ptr()
            )
            var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
                x.unsafe_ptr()
            )
            # FIXME: mojo limitation. cos and sin don't seem to behave at comptime
            # so we just calculate the twiddle factors on the cpu at runtime
            alias length = in_layout.shape[0].value()
            alias stages = UInt(log2(Float64(length)).cast[DType.index]())
            var twiddle_factors = _get_twiddle_factors[
                stages, length, calc_dtype
            ]()
            ctx.enqueue_function[
                fast_fourier_transform[
                    in_dtype,
                    out_dtype,
                    in_layout,
                    out_layout,
                    threads_per_block=TPB,
                    calc_dtype=calc_dtype,
                ]
            ](
                out_tensor,
                x_tensor,
                twiddle_factors,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

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
                        out_host[i],
                        expected[i].re.cast[out_dtype](),
                        atol=1e-3,
                        rtol=1e-5,
                    )
                    assert_almost_equal(
                        out_host[SIZE + i],
                        expected[i].im.cast[out_dtype](),
                        atol=1e-3,
                        rtol=1e-5,
                    )
        print("----------------------------")
        print("Tests passed")
        print("----------------------------")
