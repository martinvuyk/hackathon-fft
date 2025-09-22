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

from testing import assert_almost_equal

from fft.fft import (
    _DEFAULT_BASES,
    _cpu_fft_kernel_radix_n,
    _intra_block_fft_kernel_radix_n,
    _launch_inter_multiprocessor_fft,
)
from fft.utils import _get_ordered_bases_processed_list
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
    bases: List[UInt] = _DEFAULT_BASES,
    inverse: Bool = False,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
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
    if test_num == 0:
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
    elif test_num == 1:
        alias block_dim = UInt(ceil(num_threads / num_blocks))
        _launch_inter_multiprocessor_fft[
            length=length,
            processed_list=processed_list,
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
            inverse=inverse,
            block_dim=1,
            num_blocks=num_threads,
        ](output, x, ctx)
    else:
        # TODO: Implement for sequences > max_threads_available in the same GPU
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


def test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
]():
    alias SIZE = len(test_values[0][0])
    alias TPB = SIZE
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(
        SIZE, 2
    ) if inverse else Layout.row_major(SIZE)
    alias out_layout = Layout.row_major(SIZE, 2)
    alias calc_dtype = dtype
    alias Complex = ComplexSIMD[calc_dtype, 1]
    print("----------------------------")
    print("Buffers")
    print("----------------------------")

    @parameter
    fn _eval(
        result: LayoutTensor[mut=True, out_dtype, out_layout],
        scalar_in: List[Int],
        complex_out: List[ComplexSIMD[out_dtype, 1]],
    ) raises:
        print("out: ", end="")
        for i in range(SIZE):
            if i == 0:
                print("[", result[i, 0], ", ", result[i, 1], sep="", end="")
            else:
                print(",", result[i, 0], ",", result[i, 1], end="")
        print("]")
        print("expected: ", end="")

        # gather all real parts and then the imaginary parts
        @parameter
        if inverse:
            for i in range(SIZE):
                if i == 0:
                    print("[", scalar_in[i], ",", sep="", end="")
                else:
                    print(", ", scalar_in[i], sep="", end="")
            print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    Scalar[out_dtype](scalar_in[i]),
                    atol=1e-3,
                    rtol=1e-5,
                )
                assert_almost_equal(result[i, 1], 0, atol=1e-3, rtol=1e-5)
        else:
            for i in range(SIZE):
                if i == 0:
                    print("[", complex_out[i].re, ", ", sep="", end="")
                else:
                    print(", ", complex_out[i].re, ", ", sep="", end="")
                print(complex_out[i].im, end="")
            print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    complex_out[i].re.cast[out_dtype](),
                    atol=1e-3,
                    rtol=1e-5,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=1e-3,
                    rtol=1e-5,
                )

    with DeviceContext() as ctx:

        @parameter
        if target == "cpu":
            var out = List[Scalar[in_dtype]](length=SIZE * 2, fill=0)
            var x = List[Scalar[out_dtype]](length=in_layout.size(), fill=0)
            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                Span(out)
            )
            var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](Span(x))

            for test in materialize[test_values]():
                for i in range(SIZE):

                    @parameter
                    if inverse:
                        x_tensor[i, 0] = test[1][i].re.cast[in_dtype]()
                        x_tensor[i, 1] = test[1][i].im.cast[in_dtype]()
                    else:
                        x_tensor[i] = Scalar[in_dtype](test[0][i])
                test_fft[
                    bases=bases, inverse=inverse, target=target, test_num=0
                ](out_tensor, x_tensor, ctx)

                _eval(out_tensor, test[0], test[1])
        else:
            var x = ctx.enqueue_create_buffer[in_dtype](
                in_layout.size()
            ).enqueue_fill(0)
            var out = ctx.enqueue_create_buffer[out_dtype](
                SIZE * 2
            ).enqueue_fill(0)
            for test in materialize[test_values]():
                with x.map_to_host() as x_host:
                    for i in range(SIZE):

                        @parameter
                        if inverse:
                            x_host[i * 2] = test[1][i].re.cast[in_dtype]()
                            x_host[i * 2 + 1] = test[1][i].im.cast[in_dtype]()
                        else:
                            x_host[i] = Scalar[in_dtype](test[0][i])

                var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                    out.unsafe_ptr()
                )
                var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
                    x.unsafe_ptr()
                )
                test_fft[
                    bases=bases,
                    inverse=inverse,
                    target="gpu",
                    test_num=test_num,
                ](out_tensor, x_tensor, ctx)

                ctx.synchronize()

                with out.map_to_host() as out_host:
                    var tmp = LayoutTensor[mut=True, out_dtype, out_layout](
                        out_host.unsafe_ptr()
                    )
                    _eval(tmp, test[0], test[1])
        print("----------------------------")
        print("Tests passed")
        print("----------------------------")


def _test_fft[
    dtype: DType,
    func: fn[bases: List[UInt], test_values: _TestValues[dtype]] () raises,
]():
    alias L = List[UInt]

    alias values_2 = _get_test_values_2[dtype]()
    func[L(2), values_2]()

    alias values_3 = _get_test_values_3[dtype]()
    func[L(3), values_3]()

    alias values_4 = _get_test_values_4[dtype]()
    func[L(4), values_4]()
    func[L(2), values_4]()

    alias values_5 = _get_test_values_5[dtype]()
    func[L(5), values_5]()

    alias values_6 = _get_test_values_6[dtype]()
    func[L(6), values_6]()
    func[L(3, 2), values_6]()
    func[L(2, 3), values_6]()

    alias values_7 = _get_test_values_7[dtype]()
    func[L(7), values_7]()

    alias values_8 = _get_test_values_8[dtype]()
    func[L(8), values_8]()
    func[L(2), values_8]()
    func[L(4, 2), values_8]()
    func[L(2, 4), values_8]()

    alias values_10 = _get_test_values_10[dtype]()
    func[L(10), values_10]()
    func[L(5, 2), values_10]()

    alias values_16 = _get_test_values_16[dtype]()
    func[L(16), values_16]()
    func[L(2), values_16]()
    func[L(4), values_16]()
    func[L(2, 4), values_16]()
    func[L(8, 2), values_16]()
    func[L(2, 8), values_16]()

    alias values_20 = _get_test_values_20[dtype]()
    func[L(10, 2), values_20]()
    func[L(5, 4), values_20]()
    func[L(5, 2), values_20]()

    alias values_21 = _get_test_values_21[dtype]()
    func[L(7, 3), values_21]()

    alias values_32 = _get_test_values_32[dtype]()
    func[L(2), values_32]()
    func[L(16, 2), values_32]()
    func[L(8, 4), values_32]()
    func[L(4, 2), values_32]()
    func[L(8, 2), values_32]()

    alias values_35 = _get_test_values_35[dtype]()
    func[L(7, 5), values_35]()

    alias values_48 = _get_test_values_48[dtype]()
    func[L(8, 6), values_48]()
    func[L(3, 2), values_48]()

    alias values_60 = _get_test_values_60[dtype]()
    func[L(10, 6), values_60]()
    func[L(6, 5, 2), values_60]()
    func[L(5, 4, 3), values_60]()
    func[L(3, 4, 5), values_60]()
    func[L(5, 3, 2), values_60]()

    alias values_64 = _get_test_values_64[dtype]()
    func[L(2), values_64]()
    func[L(8), values_64]()
    func[L(4), values_64]()
    func[L(16, 4), values_64]()

    alias values_100 = _get_test_values_100[dtype]()
    func[L(20, 5), values_100]()
    func[L(10), values_100]()
    func[L(5, 4), values_100]()

    alias values_128 = _get_test_values_128[dtype]()
    func[L(2), values_128]()
    func[L(16, 8), values_128]()


def test_fft():
    alias dtype = DType.float64
    # _test_fft[dtype, test_fft_radix_n[dtype, inverse=False, target="cpu"]]()
    _test_fft[
        dtype, test_fft_radix_n[dtype, inverse=False, target="gpu", test_num=0]
    ]()
    _test_fft[
        dtype, test_fft_radix_n[dtype, inverse=False, target="gpu", test_num=1]
    ]()


def test_ifft():
    alias dtype = DType.float64
    # _test_fft[dtype, test_fft_radix_n[dtype, inverse=True, target="cpu"]]()
    _test_fft[
        dtype, test_fft_radix_n[dtype, inverse=True, target="gpu", test_num=0]
    ]()
    _test_fft[
        dtype, test_fft_radix_n[dtype, inverse=True, target="gpu", test_num=1]
    ]()


def main():
    test_fft()
    test_ifft()
