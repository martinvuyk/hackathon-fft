from complex import ComplexSIMD
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep,
    ThroughputMeasure,
    BenchMetric,
)
from layout import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext
from random import seed, randn, random_ui64
from sys.info import size_of

from fft.fft.fft import fft
from fft.fft._utils import _product_of_dims

comptime GPU_ITERS = 1


@parameter
fn bench_gpu_radix_n_rfft[dtype: DType, shape: IntTuple](mut b: Bencher) raises:
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(IntTuple(shape, 1).flatten())
    comptime out_layout = Layout.row_major(IntTuple(shape, 2).flatten())

    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[out_dtype](out_layout.size())
        out.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
        with x.map_to_host() as x_host:
            randn(x_host.unsafe_ptr(), in_layout.size())

        var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
            out.unsafe_ptr()
        )
        var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
            x.unsafe_ptr()
        )

        @always_inline
        @parameter
        fn call_fn(ctx: DeviceContext) raises:
            fft(out_tensor, x_tensor, ctx)
            ctx.synchronize()

        ctx.synchronize()
        b.iter_custom[call_fn](ctx)


@parameter
fn bench_cpu_radix_n_rfft[
    dtype: DType,
    shape: IntTuple,
    *,
    cpu_workers: Optional[UInt] = None,
](mut b: Bencher) raises:
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(IntTuple(shape, 1).flatten())
    comptime out_layout = Layout.row_major(IntTuple(shape, 2).flatten())

    var out = List[Scalar[out_dtype]](capacity=out_layout.size())
    var x = List[Scalar[in_dtype]](capacity=in_layout.size())
    randn(x.unsafe_ptr(), in_layout.size())

    var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
        out.unsafe_ptr()
    )
    var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](x.unsafe_ptr())

    @always_inline
    @parameter
    fn call_fn() raises:
        fft(out_tensor, x_tensor, cpu_workers=cpu_workers)

    b.iter[call_fn]()

    _ = out_tensor
    _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=5))
    comptime shapes: List[IntTuple] = [
        # {100_000, 128},
        {100_000, 2**10},
        # {100, 2**14},
        # {100, 640, 480},
        # {10, 1920, 1080},
        # {1, 3840, 2160},
        # {1, 7680, 4320},
        # {100, 64, 64, 64},
        # {10, 128, 128, 128},
        # {1, 256, 256, 256},
        # {1, 512, 512, 512},
    ]

    @parameter
    for shape in shapes:
        comptime dtype = DType.float32
        comptime name = String("bench_gpu_radix_n_rfft[", shape, "]")
        comptime num_elems = _product_of_dims(shape)
        comptime total_bytes = num_elems * size_of[dtype]() * 3 * GPU_ITERS
        m.bench_function[bench_gpu_radix_n_rfft[dtype, shape]](
            BenchId(name), [ThroughputMeasure(BenchMetric.bytes, total_bytes)]
        )
        comptime cpu_bench = "bench_cpu_radix_n_rfft["
        # m.bench_function[
        #     bench_cpu_radix_n_rfft[dtype, shape, cpu_workers= {1}]
        # ](BenchId(String(cpu_bench, shape, ", workers=1]")))
        # m.bench_function[bench_cpu_radix_n_rfft[dtype, shape]](
        #     BenchId(String(cpu_bench, shape, ", workers=n]"))
        # )

    # results = Dict[String, Tuple[Float64, Int]]()
    # for info in m.info_vec:
    #     n = info.name
    #     time = info.result.mean("ms")
    #     avg, amnt = results.get(n, (Float64(0), 0))
    #     results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    # print("")
    # for k_v in results.items():
    #     print(k_v.key, k_v.value[0].__round__(3), sep=", ")
    print(m)
