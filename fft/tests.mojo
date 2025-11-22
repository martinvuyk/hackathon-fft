from complex import ComplexScalar
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of
from utils.numerics import nan

from testing import assert_almost_equal

from fft.fft import (
    _cpu_fft_kernel_radix_n,
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
        _cpu_fft_kernel_radix_n[
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
        ](output, x)
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
            x_data.enqueue_fill(nan[in_dtype]())
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
    # _test[dtype, False, "cpu", 0, debug=False]()
    # _test[dtype, False, "gpu", 0, debug=False]()
    _test[dtype, False, "gpu", 1, debug=False]()
    # _test[dtype, False, "gpu", 2, debug=False]()
    # _test[dtype, False, "gpu", 3, debug=False]()


def test_ifft():
    comptime dtype = DType.float64
    # _test[dtype, True, "cpu", 0, debug=False]()
    # _test[dtype, True, "gpu", 0, debug=False]()
    # _test[dtype, True, "gpu", 1, debug=False]()
    # _test[dtype, True, "gpu", 2, debug=False]()
    # _test[dtype, True, "gpu", 3, debug=False]()


def main():
    test_fft()
    test_ifft()
