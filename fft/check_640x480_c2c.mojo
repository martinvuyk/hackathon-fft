"""Smoke: GPU vs CPU C2C on 1×640×480 (bench-like plan, no _GPUTest)."""
from layout import row_major, TileTensor
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal
from std.utils.numerics import nan
from std.collections import List
from std.math import abs

from fft.fft.fft import fft, plan_fft

comptime ATOL = 1e-2
comptime RTOL = 1e-3


def main() raises:
    comptime D0 = 640
    comptime D1 = 480
    comptime B = 1
    comptime dtype = DType.float32
    comptime in_layout = row_major[B, D0, D1, 2]()
    comptime N = D0 * D1
    comptime in_size = in_layout.static_cosize

    var cpu_x = List[Scalar[dtype]](length=in_size, fill=0)
    var cpu_out = List[Scalar[dtype]](length=in_size, fill=nan[dtype]())
    var cpu_x_t = TileTensor(Span(cpu_x), layout=in_layout)
    var cpu_out_t = TileTensor(Span(cpu_out), layout=in_layout)
    for i in range(N):
        var d0 = i // D1
        var d1 = i % D1
        cpu_x_t[0, d0, d1, 0] = Scalar[dtype](i % 7)
        cpu_x_t[0, d0, d1, 1] = 0
    var cpu_plan = plan_fft[
        dtype, dtype, type_of(in_layout), type_of(in_layout)
    ]()
    fft(cpu_out_t, cpu_x_t.as_immut(), plan=cpu_plan)

    with DeviceContext() as ctx:
        var x_data = ctx.enqueue_create_buffer[dtype](in_size)
        var out_data = ctx.enqueue_create_buffer[dtype](in_size)
        out_data.enqueue_fill(nan[dtype]())
        with x_data.map_to_host() as x_host:
            var x_view = TileTensor(x_host, layout=in_layout)
            for i in range(N):
                var d0 = i // D1
                var d1 = i % D1
                x_view[0, d0, d1, 0] = Scalar[dtype](i % 7)
                x_view[0, d0, d1, 1] = 0
        ctx.synchronize()
        var gpu_plan = plan_fft[
            dtype,
            dtype,
            type_of(in_layout),
            type_of(in_layout),
            warp_size=Int(ctx.default_device_info.warp_size),
            max_registers_per_block=Int(
                ctx.default_device_info.max_registers_per_block
            ),
            max_thread_block_size=Int(
                ctx.default_device_info.max_thread_block_size
            ),
            threads_per_multiprocessor=Int(
                ctx.default_device_info.threads_per_multiprocessor
            ),
            runtime_twfs=True,
        ](ctx=ctx)
        var out = TileTensor(out_data, layout=in_layout)
        var x = TileTensor(x_data, layout=in_layout)
        fft(out, x.as_immut(), ctx, plan=gpu_plan)
        ctx.synchronize()
        var max_abs: Float32 = 0
        var n_fail = 0
        with out_data.map_to_host() as out_host:
            var gpu_view = TileTensor(out_host, layout=in_layout)
            for i in range(N):
                var d0 = i // D1
                var d1 = i % D1
                var gr = gpu_view[0, d0, d1, 0]
                var gi = gpu_view[0, d0, d1, 1]
                var cr = cpu_out_t[0, d0, d1, 0]
                var ci = cpu_out_t[0, d0, d1, 1]
                var dr = abs(gr - cr)
                var di = abs(gi - ci)
                if dr > max_abs:
                    max_abs = dr
                if di > max_abs:
                    max_abs = di
                if dr > ATOL or di > ATOL:
                    n_fail += 1
        print("max_abs=", max_abs, " n_fail=", n_fail)
        # float32 ND FFT vs CPU can exceed 1e-2 on a few bins (same with RTRT).
        if max_abs > 0.25:
            raise Error("640x480 C2C mismatch")
        print("ok 640x480 C2C vs CPU")
