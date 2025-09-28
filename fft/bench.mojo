from complex import ComplexSIMD
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from os import abort
from random import seed
from sys.info import is_64bit
from bit import count_trailing_zeros

from fft._test_values import _get_test_values_128
from fft.tests import _TestValues
from fft.fft import (
    fft,
    _intra_block_fft_kernel_radix_n,
    _get_ordered_bases_processed_list,
)


def _bench_intra_block_fft_launch_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt],
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
    mut b: Bencher,
):
    alias length = in_layout.shape[1].value()
    alias bases_processed = _get_ordered_bases_processed_list[
        length, bases, "gpu"
    ]()
    alias ordered_bases = bases_processed[0]
    alias processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        alias last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = (length // last_base) * (last_base - 1) * len(bases)
        var offsets = List[UInt](capacity=c)
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (length // base) * (base - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]

    @parameter
    fn call_fn[
        in_dtype: DType,
        out_dtype: DType,
        in_layout: Layout,
        out_layout: Layout,
        *,
        ordered_bases: List[UInt],
        processed_list: List[UInt],
    ](
        output: LayoutTensor[mut=True, out_dtype, out_layout],
        x: LayoutTensor[mut=False, in_dtype, in_layout],
    ):
        for _ in range(10_000):
            _intra_block_fft_kernel_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                length=length,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
                do_rfft=True,
                inverse=False,
                total_twfs=total_twfs,
                twf_offsets=twf_offsets,
            ](output, x)

    alias num_threads = length // ordered_bases[len(ordered_bases) - 1]
    ctx.enqueue_function[
        call_fn[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
        ]
    ](output, x, grid_dim=1, block_dim=num_threads)
    ctx.synchronize()
    _ = processed_list  # origin bug


@parameter
fn bench_intra_block_radix_n_rfft[
    dtype: DType, bases: List[UInt], test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    alias values = test_values[len(test_values) - 1]
    alias SIZE = len(values[0])
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(1, SIZE, 1)
    alias out_layout = Layout.row_major(1, SIZE, 2)
    alias calc_dtype = dtype
    alias Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[out_dtype](
            out_layout.size()
        ).enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[in_dtype](
            in_layout.size()
        ).enqueue_fill(0)
        ref series = values[0]
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
        fn call_fn(ctx: DeviceContext) raises:
            _bench_intra_block_fft_launch_radix_n[bases=bases](
                out_tensor, x_tensor, ctx, b
            )

        b.iter_custom[call_fn](ctx)

        _ = out_tensor
        _ = x_tensor


@parameter
fn bench_cpu_radix_n_rfft[
    dtype: DType,
    bases: List[UInt],
    batches: UInt,
    test_values: _TestValues[dtype],
    *,
    cpu_workers: Optional[UInt] = None,
](mut b: Bencher) raises:
    alias values = test_values[len(test_values) - 1]
    alias SIZE = len(values[0])
    alias BATCHES = batches
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(BATCHES, SIZE, 1)
    alias out_layout = Layout.row_major(BATCHES, SIZE, 2)
    alias calc_dtype = dtype
    alias Complex = ComplexSIMD[calc_dtype, 1]

    var out = List[Scalar[out_dtype]](capacity=out_layout.size())
    var x = List[Scalar[in_dtype]](capacity=in_layout.size())
    ref series = values[0]
    var idx = 0
    for _ in range(BATCHES):
        for i in range(SIZE):
            x[idx] = series[i]
            idx += 1

    var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
        out.unsafe_ptr()
    )
    var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](x.unsafe_ptr())

    @always_inline
    @parameter
    fn call_fn() raises:
        fft[bases=bases, target="cpu"](
            out_tensor, x_tensor, DeviceContext(), cpu_workers=cpu_workers
        )

    b.iter[call_fn]()

    _ = out_tensor
    _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    alias bases_list: List[List[UInt]] = [
        [64, 2],
        [32, 4],
        [16, 8],
        [16, 4, 2],
        [8, 8, 2],
        [8, 4, 4],
        [8, 4, 2, 2],
        [8, 2, 2, 2, 2],
        [4, 4, 4, 2],
        [4, 4, 2, 2, 2],
        [4, 2, 2, 2, 2, 2],
        [2],
    ]
    alias test_values_fp32 = _get_test_values_128[DType.float32]()
    alias test_values_fp64 = _get_test_values_128[DType.float64]()

    @parameter
    for bases in bases_list:
        alias b = bases.__str__().replace("UInt(", "").replace(")", "")
        m.bench_function[
            bench_intra_block_radix_n_rfft[
                DType.float32, bases, test_values_fp32
            ]
        ](BenchId(String("bench_intra_block_radix_n_rfft[", b, ", 128]")))
        alias cpu_bench = "bench_cpu_radix_n_rfft["
        m.bench_function[
            bench_cpu_radix_n_rfft[
                DType.float64,
                bases,
                100_000,
                test_values_fp64,
                cpu_workers = UInt(1),
            ]
        ](BenchId(String(cpu_bench, b, ", 100_000, 128, workers=1]")))
        m.bench_function[
            bench_cpu_radix_n_rfft[
                DType.float64,
                bases,
                1_000_000,
                test_values_fp64,
                cpu_workers = UInt(1),
            ]
        ](BenchId(String(cpu_bench, b, ", 1_000_000, 128, workers=1]")))
        m.bench_function[
            bench_cpu_radix_n_rfft[
                DType.float64, bases, 100_000, test_values_fp64
            ]
        ](BenchId(String(cpu_bench, b, ", 100_000, 128, workers=n]")))
        m.bench_function[
            bench_cpu_radix_n_rfft[
                DType.float64, bases, 1_000_000, test_values_fp64
            ]
        ](BenchId(String(cpu_bench, b, ", 1_000_000, 128, workers=n]")))

    results = Dict[String, (Float64, Int)]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0].__round__(3), sep=", ")
