from layout import RowMajorLayout, TileTensor, IntTuple
from layout.int_tuple import _IntTupleToCoordLike
from max.gpu.host import DeviceContext
from std.random import seed, randn
from fft.fft.fft import fft, plan_fft

def main() raises:
    seed()
    comptime dtype = DType.float32
    comptime shape = IntTuple(20, 640, 480)
    comptime shape_flat = IntTuple(shape, 2).flatten()
    comptime in_layout = RowMajorLayout[
        *_IntTupleToCoordLike[DType.int64, shape_flat]
    ]()
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](in_layout.static_cosize)
        out.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](in_layout.static_cosize)
        with x.map_to_host() as xh:
            randn(xh.unsafe_ptr(), in_layout.static_cosize)
        var ot = TileTensor(out, layout=in_layout)
        var xt = TileTensor(x, layout=in_layout)
        var plan = plan_fft[dtype, dtype, type_of(in_layout), type_of(in_layout)](ctx=ctx)
        ctx.synchronize()
        fft(ot, xt, ctx, plan=plan)
        ctx.synchronize()
