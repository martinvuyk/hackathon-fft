from complex import ComplexSIMD
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from random import seed

from fft._test_values import _get_test_values_128
from fft.tests import _TestValues
from fft.fft.fft import fft


@parameter
fn profile_intra_block_radix_n[
    dtype: DType, test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    comptime values = test_values[len(test_values) - 1]
    comptime smallest_base = 2
    comptime SIZE = len(values[0])
    comptime max_threads_available = 48 * 32 * 170
    comptime BATCHES = max_threads_available // (SIZE // Int(smallest_base))
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(BATCHES, SIZE, 1)
    comptime out_layout = Layout.row_major(BATCHES, SIZE, 2)
    comptime calc_dtype = dtype
    comptime Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[out_dtype](out_layout.size())
        x = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
        ref series = values[0]
        var idx = 0
        with x.map_to_host() as x_host:
            for _ in range(BATCHES):
                for i in range(SIZE):
                    x_host[idx] = series[i]
                    idx += 1

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

        b.iter_custom[call_fn](ctx)

        _ = out_tensor
        _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime dtype = DType.float32
    comptime test_values = _get_test_values_128[dtype]()
    m.bench_function[profile_intra_block_radix_n[dtype, test_values]](
        BenchId(String("profile_intra_block_radix_n"))
    )
