from std.complex import ComplexSIMD
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.random import seed

from fft._test_values import _get_test_values_128
from fft.tests import _TestValues
from fft.fft.fft import fft, plan_fft


def profile_intra_block_radix_n[
    dtype: DType, test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    comptime values = test_values[len(test_values) - 1]
    comptime smallest_base = 2
    comptime SIZE = len(values[0])
    comptime max_threads_available = 48 * 32 * 170
    comptime BATCHES = max_threads_available // (SIZE // Int(smallest_base))
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = row_major[BATCHES, SIZE, 1]()
    comptime out_layout = row_major[BATCHES, SIZE, 2]()
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize
    comptime calc_dtype = dtype
    comptime Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[out_dtype](out_size)
        x = ctx.enqueue_create_buffer[in_dtype](in_size)
        ref series = materialize[values]()[0]
        var idx = 0
        with x.map_to_host() as x_host:
            for _ in range(BATCHES):
                for i in range(SIZE):
                    x_host[idx] = {series[i]}
                    idx += 1

        var out_tensor = TileTensor(out, layout=out_layout)
        var x_tensor = TileTensor(x, layout=in_layout)
        comptime bases: List[List[UInt]] = [[UInt(2)]]
        var plan = plan_fft[
            in_dtype,
            out_dtype,
            type_of(in_layout),
            type_of(out_layout),
            bases=bases,
            runtime_twfs=True,
        ](ctx=ctx)
        ctx.synchronize()

        @always_inline
        def call_fn(
            ctx: DeviceContext,
        ) raises {mut out_tensor, imm x_tensor, imm plan,}:
            fft(out_tensor, x_tensor, ctx, plan=plan)
            ctx.synchronize()

        b.iter_custom(call_fn, ctx)

        _ = out_tensor
        _ = x_tensor


def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime dtype = DType.float32
    comptime test_values = _get_test_values_128[dtype]()
    m.bench_function[profile_intra_block_radix_n[dtype, test_values]](
        BenchId(String("profile_intra_block_radix_n"))
    )
