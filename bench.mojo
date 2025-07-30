from complex import ComplexSIMD
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from os import abort
from random import seed

# from nn import irfft


from _test_values import _get_test_values_128
from tests import _TestValues
from fft import _intra_block_fft_launch_radix_n, _intra_block_fft_launch


@parameter
fn bench_intra_block_radix_n[
    dtype: DType, bases: List[UInt], test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    alias SIZE = len(test_values[0][0])
    alias TPB = SIZE
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(SIZE)
    alias out_layout = Layout.row_major(SIZE, 2)
    alias calc_dtype = dtype
    alias Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[out_dtype](SIZE * 2).enqueue_fill(0)
        x = ctx.enqueue_create_buffer[in_dtype](SIZE).enqueue_fill(0)
        for test in test_values:
            ref series = test[0]
            with x.map_to_host() as x_host:
                for i in range(SIZE):
                    x_host[i] = series[i]

            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                out.unsafe_ptr()
            )
            var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
                x.unsafe_ptr()
            )

            @always_inline
            @parameter
            fn call_fn():
                try:
                    # _intra_block_fft_launch[
                    #     threads_per_block=SIZE, blocks_per_grid=1
                    # ](out_tensor, x_tensor, ctx)
                    _intra_block_fft_launch_radix_n[bases=bases](
                        out_tensor, x_tensor, ctx
                    )
                    ctx.synchronize()
                except e:
                    abort(String("benchmark code failed: ", e))

            b.iter[call_fn]()
            _ = out_tensor
            _ = x_tensor


# @parameter
# fn bench_cufft[test_values: _TestValues[DType.float32]](mut b: Bencher) raises:
#     alias SIZE = len(test_values[0][0])
#     alias TPB = SIZE
#     alias BLOCKS_PER_GRID = (1, 1)
#     alias THREADS_PER_BLOCK = (TPB, 1)
#     alias in_dtype = DType.float32
#     alias out_dtype = DType.float32
#     alias in_layout = Layout.row_major(SIZE)
#     alias out_layout = Layout.row_major(SIZE, 2)
#     alias calc_dtype = DType.float32
#     alias Complex = ComplexSIMD[calc_dtype, 1]

#     with DeviceContext() as ctx:
#         out = ctx.enqueue_create_buffer[out_dtype](SIZE * 2).enqueue_fill(0)
#         x = ctx.enqueue_create_buffer[in_dtype](SIZE).enqueue_fill(0)
#         for test in test_values:
#             ref series = test[0]
#             with x.map_to_host() as x_host:
#                 for i in range(SIZE):
#                     x_host[i] = series[i]

#             var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
#                 out.unsafe_ptr()
#             )
#             var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
#                 x.unsafe_ptr()
#             )

#             @always_inline
#             @parameter
#             fn call_fn():
#                 try:
#                     irfft(
#                         x_tensor,
#                         out_tensor,
#                         SIZE * 2,
#                         SIZE * 2 * 32 // 8,
#                         ctx,
#                     )
#                     ctx.synchronize()
#                 except e:
#                     abort(String("benchmark code failed: ", e))

#             b.iter[call_fn]()
#             _ = out_tensor
#             _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=10))
    alias bases_list: List[List[UInt]] = [[2]]  # [[16, 8], [8, 2], [4, 2], [2]]
    alias test_values = _get_test_values_128[DType.float32]()

    @parameter
    for bases in bases_list:
        alias suffix = String(bases.__str__(), ", ", 128, "]")
        m.bench_function[
            bench_intra_block_radix_n[DType.float32, bases, test_values]
        ](BenchId(String("bench_intra_block_radix_n[", suffix)))

    # m.bench_function[bench_cufft[test_values]](BenchId("bench_cufft[128]"))

    results = Dict[String, (Float64, Int)]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=", ")
