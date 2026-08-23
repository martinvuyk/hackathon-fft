"""One-shot ND FFT for nvprof / phase inspection. Not part of the bench suite."""

from layout import RowMajorLayout, TileTensor, IntTuple
from layout.int_tuple import _IntTupleToCoordLike
from max.gpu.host import DeviceContext
from std.random import seed, randn

from fft.fft.fft import fft, plan_fft


def _run_shape[shape: IntTuple]() raises:
    comptime dtype = DType.float32
    comptime shape_flat = IntTuple(shape, 2).flatten()
    comptime in_layout = RowMajorLayout[
        *_IntTupleToCoordLike[DType.int64, shape_flat]
    ]()
    comptime out_layout = in_layout
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize

    print("shape", shape)
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](out_size)
        out.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](in_size)
        with x.map_to_host() as x_host:
            randn(x_host.unsafe_ptr(), in_size)

        var out_tensor = TileTensor(out, layout=out_layout)
        var x_tensor = TileTensor(x, layout=in_layout)
        var plan = plan_fft[
            dtype,
            dtype,
            type_of(in_layout),
            type_of(out_layout),
            runtime_twfs=True,
        ](ctx=ctx)
        ctx.synchronize()

        # Warmup
        for _ in range(3):
            fft(out_tensor, x_tensor, ctx, plan=plan)
            ctx.synchronize()

        # Timed iters (nvprof attributes GPU time; host loop is fine)
        for _ in range(20):
            fft(out_tensor, x_tensor, ctx, plan=plan)
            ctx.synchronize()


def main() raises:
    seed()
    _run_shape[{100, 64, 64, 64}]()
    _run_shape[{10, 128, 128, 128}]()
