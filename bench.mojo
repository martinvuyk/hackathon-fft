from complex import ComplexSIMD
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from os import abort
from random import seed

# from nn import irfft


from _test_values import _get_test_values_128
from tests import _TestValues
from fft import (
    _intra_block_fft_launch_radix_n,
    _intra_block_fft_launch,
    _intra_block_fft_kernel_radix_n,
    _get_twiddle_factors,
    _log_mod,
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
    alias length = in_layout.shape[0].value()
    alias twiddle_factors = _get_twiddle_factors[length, out_dtype]()

    @parameter
    fn _build_ordered_bases() -> List[UInt]:
        var processed = UInt(1)
        var new_bases = List[UInt](capacity=len(bases))

        @parameter
        for base in bases:
            var amnt_divisible = _log_mod[base](length // processed)[0]
            for _ in range(amnt_divisible):
                new_bases.append(base)
                processed *= base
        sort(new_bases)  # FIXME: this should just be ascending=False
        new_bases.reverse()
        return new_bases

    alias ordered_bases = _build_ordered_bases()

    @parameter
    fn _build_processed_list() -> List[UInt]:
        var processed_list = List[UInt](capacity=len(ordered_bases))
        var processed = 1
        for base in ordered_bases:
            processed_list.append(processed)
            processed *= base
        return processed_list

    alias processed_list = _build_processed_list()
    constrained[
        processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length,
        "powers of the bases must multiply together",
        " to equal the sequence length",
    ]()

    # @parameter
    # fn call_fn[
    #     in_dtype: DType,
    #     out_dtype: DType,
    #     in_layout: Layout,
    #     out_layout: Layout,
    #     *,
    #     twiddle_factors: InlineArray[
    #         ComplexSIMD[out_dtype, 1], in_layout.shape[0].value() - 1
    #     ],
    #     ordered_bases: List[UInt],
    #     processed_list: List[UInt],
    # ](
    #     output: LayoutTensor[mut=True, out_dtype, out_layout],
    #     x: LayoutTensor[mut=False, in_dtype, in_layout],
    #     ctx: DeviceContext,
    # ):
    #     @always_inline
    #     @parameter
    #     fn inner_fn(ctx: DeviceContext) raises:
    #         _intra_block_fft_kernel_radix_n[
    #             in_dtype,
    #             out_dtype,
    #             in_layout,
    #             out_layout,
    #             twiddle_factors=twiddle_factors,
    #             ordered_bases=ordered_bases,
    #             processed_list=processed_list,
    #         ](output, x)
    #         ctx.synchronize()

    #     try:
    #         b.iter_custom[inner_fn](ctx)
    #     except e:
    #         abort(String("something: ", e))

    # ctx.enqueue_function[
    #     call_fn[
    #         in_dtype,
    #         out_dtype,
    #         in_layout,
    #         out_layout,
    #         twiddle_factors=twiddle_factors,
    #         ordered_bases=ordered_bases,
    #         processed_list=processed_list,
    #     ]
    # ](
    #     output,
    #     x,
    #     ctx,
    #     grid_dim=1,
    #     block_dim=length // ordered_bases[len(ordered_bases) - 1],
    # )
    @parameter
    fn call_fn[
        in_dtype: DType,
        out_dtype: DType,
        in_layout: Layout,
        out_layout: Layout,
        *,
        twiddle_factors: InlineArray[
            ComplexSIMD[out_dtype, 1], in_layout.shape[0].value() - 1
        ],
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
                twiddle_factors=twiddle_factors,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
            ](output, x)

    ctx.enqueue_function[
        call_fn[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors=twiddle_factors,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
        ]
    ](
        output,
        x,
        grid_dim=1,
        block_dim=length // ordered_bases[len(ordered_bases) - 1],
    )
    ctx.synchronize()
    _ = processed_list  # origin bug


@parameter
fn bench_intra_block_radix_n[
    dtype: DType, bases: List[UInt], test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    alias values = test_values[len(test_values) - 1]
    alias SIZE = len(values[0])
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
            # _intra_block_fft_launch_radix_n[bases=bases](
            #     out_tensor, x_tensor, ctx
            # )
            # ctx.synchronize()
            _bench_intra_block_fft_launch_radix_n[bases=bases](
                out_tensor, x_tensor, ctx, b
            )

        b.iter_custom[call_fn](ctx)

        # b.iter[call_fn]()
        _ = out_tensor
        _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=10))
    # pure radix-2 implementation takes ~23 ms. the generic one ~90 ms
    # in my machine. Cufft takes ~40 ms.
    # NOTE: radix-2 implementation needs fixing to actually pass the tests.
    alias bases_list: List[List[UInt]] = [[16, 8], [8, 2], [4, 2], [2]]
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
