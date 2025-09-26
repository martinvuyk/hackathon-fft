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
from fft.fft import fft

from fft.fft import (
    _intra_block_fft_kernel_radix_n,
    _get_ordered_bases_processed_list,
)


def _profile_intra_block_fft_launch_radix_n[
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
    alias bases_processed = _get_ordered_bases_processed_list[length, bases]()
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
    alias batches = in_layout.shape[0].value()
    alias max_threads_available = 48 * 32 * 170
    alias batch_size = max_threads_available // num_threads
    alias func = call_fn[
        in_dtype,
        out_dtype,
        in_layout,
        out_layout,
        ordered_bases=ordered_bases,
        processed_list=processed_list,
    ]

    @parameter
    for _ in range(batches // batch_size):
        ctx.enqueue_function[func](
            output, x, grid_dim=(1, batch_size), block_dim=num_threads
        )
    alias remainder = batches % batch_size

    @parameter
    if remainder > 0:
        ctx.enqueue_function[func](
            output, x, grid_dim=(1, remainder), block_dim=num_threads
        )


@parameter
fn profile_intra_block_radix_n[
    dtype: DType, ordered_bases: List[UInt], test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    alias values = test_values[len(test_values) - 1]
    alias smallest_base = ordered_bases[len(ordered_bases) - 1]
    alias SIZE = len(values[0])
    alias max_threads_available = 48 * 32 * 170
    alias BATCHES = max_threads_available // (SIZE // smallest_base)
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(BATCHES, SIZE, 1)
    alias out_layout = Layout.row_major(BATCHES, SIZE, 2)
    alias calc_dtype = dtype
    alias Complex = ComplexSIMD[calc_dtype, 1]

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
            _profile_intra_block_fft_launch_radix_n[bases=ordered_bases](
                out_tensor, x_tensor, ctx, b
            )

        b.iter_custom[call_fn](ctx)
        ctx.synchronize()

        _ = out_tensor
        _ = x_tensor


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    alias ordered_bases: List[UInt] = [2]
    alias dtype = DType.float32
    alias test_values = _get_test_values_128[dtype]()
    m.bench_function[
        profile_intra_block_radix_n[dtype, ordered_bases, test_values]
    ](BenchId(String("profile_intra_block_radix_n")))
