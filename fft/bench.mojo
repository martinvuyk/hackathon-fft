"""FFT benchmarks — switch `SUITE` in `main()`; do not invent ad-hoc configs.

See `scratchpad/02-benchmarking-protocol.md` for DUT rules, packs, and when
to use BASELINE vs QUICK.

Uses value-taking `Bench.bench_function(closure, id)` and
`bencher_iter_custom(bencher, launch, ctx)` — not parametric capturing
bench bodies — to avoid origin/lifetime issues.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.collections import Optional
from layout import RowMajorLayout, TileTensor, IntTuple
from layout.int_tuple import _IntTupleToCoordLike

from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.os import abort
from std.random import seed, randn

from fft.fft.fft import fft, plan_fft


@fieldwise_init
struct BenchSuite(Copyable, Movable):
    """Which preset to run. Change only this (and TRACK when SUITE=TRACK)."""

    comptime GPU_BASELINE = Self(0)
    comptime GPU_QUICK = Self(1)
    comptime CPU_BASELINE = Self(2)
    comptime CPU_QUICK = Self(3)
    comptime TRACK = Self(4)
    var v: Int


@fieldwise_init
struct TrackId(Copyable, Movable):
    """Used only when SUITE == TRACK — mirrors scratchpad track numbers."""

    comptime T00 = Self(0)
    comptime T01 = Self(1)
    comptime T02 = Self(2)
    comptime T03 = Self(3)
    comptime T04 = Self(4)
    comptime T05 = Self(5)
    comptime T06 = Self(6)
    comptime T07 = Self(7)
    comptime T08 = Self(8)
    var v: Int


# --- Shape packs (keep in sync with scratchpad/02-benchmarking-protocol.md) ---

comptime PACK_GPU_CORE: List[IntTuple] = [
    {500_000, 128},
    {100_000, 2**10},
    {100, 64, 64, 64},
    {10, 128, 128, 128},
    {100, 640, 480},
]

comptime PACK_GPU_1D: List[IntTuple] = [
    {500_000, 93},
    {500_000, 128},
    {100_000, 2**10},
]

# T02 primary: 1D pack + batch-rich ND short axis (64³).
comptime PACK_GPU_T02: List[IntTuple] = [
    {500_000, 93},
    {500_000, 128},
    {100_000, 2**10},
    {100, 64, 64, 64},
]

comptime PACK_GPU_ND: List[IntTuple] = [
    {100, 64, 64, 64},
    {10, 128, 128, 128},
    {100, 640, 480},
    {1, 256, 256, 256},
]

comptime PACK_GPU_SM: List[IntTuple] = [
    {500_000, 128},
    {100_000, 2**10},
    {500_000, 32},
    {500_000, 64},
]

comptime PACK_GPU_LARGE_AXIS: List[IntTuple] = [
    {100, 640, 480},
    {100_000, 2**10},
]

comptime PACK_CPU_CORE: List[IntTuple] = [
    {500_000, 128},
    {100_000, 2**10},
    {100, 640, 480},
    {100, 64, 64, 64},
    {10, 128, 128, 128},
    {1, 256, 256, 256},
]

comptime PACK_CPU_QUICK: List[IntTuple] = [
    {100_000, 2**10},
    {100, 64, 64, 64},
]


def _gpu_baseline_config() raises -> BenchConfig:
    return BenchConfig(
        num_repetitions=1,
        num_warmup_iters=50,
        max_iters=200,
        min_runtime_secs=1.0,
        max_runtime_secs=5.0,
    )


def _gpu_quick_config() raises -> BenchConfig:
    return BenchConfig(
        num_repetitions=1,
        num_warmup_iters=10,
        max_iters=80,
        min_runtime_secs=0.25,
        max_runtime_secs=1.5,
    )


def _cpu_baseline_config() raises -> BenchConfig:
    return BenchConfig(
        num_repetitions=1,
        num_warmup_iters=2,
        max_iters=8,
        min_runtime_secs=0.0,
        max_runtime_secs=3.0,
    )


def _cpu_quick_config() raises -> BenchConfig:
    return BenchConfig(
        num_repetitions=1,
        num_warmup_iters=1,
        max_iters=2,
        min_runtime_secs=0.0,
        max_runtime_secs=1.0,
    )


def _gpu_shape_body[dtype: DType, shape: IntTuple](mut b: Bencher) raises:
    """Setup + timed GPU FFT for one shape (raising body; wrapped by closure)."""
    comptime shape_flat = IntTuple(shape, 2).flatten()
    comptime in_layout = RowMajorLayout[
        *_IntTupleToCoordLike[DType.int64, shape_flat]
    ]()
    comptime out_layout = in_layout
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize

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
        ctx.synchronize()

        @always_inline
        def call_fn(
            ctx: DeviceContext,
        ) raises {mut out_tensor, imm x_tensor, imm plan}:
            fft(out_tensor, x_tensor, ctx, plan=plan)
            ctx.synchronize()

        bencher_iter_custom(b, call_fn, ctx)


def _register_gpu_shape[dtype: DType, shape: IntTuple](mut m: Bench) raises:
    """Register one GPU C2C shape via value-taking bench closure."""

    @always_inline
    def bench_fn(mut b: Bencher):
        try:
            _gpu_shape_body[dtype, shape](b)
        except e:
            abort(String(e))

    m.bench_function(
        bench_fn, BenchId(String("bench_gpu_radix_n_rfft[", shape, "]"))
    )


def _cpu_shape_body[
    dtype: DType,
    shape: IntTuple,
    *,
    cpu_workers: Optional[UInt] = None,
](mut b: Bencher) raises:
    comptime shape_flat = IntTuple(shape, 2).flatten()
    comptime in_layout = RowMajorLayout[
        *_IntTupleToCoordLike[DType.int64, shape_flat]
    ]()
    comptime out_layout = in_layout
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize

    var out = List[Scalar[dtype]](unsafe_uninit_length=out_size)
    var x = List[Scalar[dtype]](unsafe_uninit_length=in_size)
    randn(x.unsafe_ptr(), in_size)

    var out_tensor = TileTensor(Span(out), layout=out_layout)
    var x_tensor = TileTensor(Span(x), layout=in_layout)
    var plan = plan_fft[
        dtype, dtype, type_of(in_layout), type_of(out_layout)
    ](cpu_workers=cpu_workers)

    @always_inline
    def call_fn() {mut out_tensor, imm x_tensor, imm plan}:
        try:
            fft(
                out_tensor,
                x_tensor.as_immut(),
                plan=plan,
                cpu_workers=cpu_workers,
            )
        except e:
            abort(String(e))

    b.iter(call_fn)


def _register_cpu_shape[
    dtype: DType,
    shape: IntTuple,
    *,
    cpu_workers: Optional[UInt] = None,
    label_workers: StringLiteral,
](mut m: Bench) raises:
    @always_inline
    def bench_fn(mut b: Bencher):
        try:
            _cpu_shape_body[dtype, shape, cpu_workers=cpu_workers](b)
        except e:
            abort(String(e))

    m.bench_function(
        bench_fn,
        BenchId(
            String("bench_cpu_radix_n_rfft[", shape, ", ", label_workers, "]")
        ),
    )


def _register_gpu_pack[shapes: List[IntTuple]](mut m: Bench) raises:
    comptime for shape in shapes:
        _register_gpu_shape[DType.float32, shape](m)


def _register_cpu_pack[shapes: List[IntTuple]](mut m: Bench) raises:
    comptime for shape in shapes:
        _register_cpu_shape[
            DType.float32, shape, cpu_workers={1}, label_workers="workers=1"
        ](m)
    comptime for shape in shapes:
        _register_cpu_shape[
            DType.float32, shape, label_workers="workers=n"
        ](m)


def main() raises:
    seed()

    # --- Agent / human switchboard (see scratchpad/02-benchmarking-protocol.md) ---
    # Default QUICK so accidental runs stay short.
    # For DUT baseline: set SUITE = BenchSuite.GPU_BASELINE, then `pixi run bench`.
    comptime SUITE = BenchSuite.TRACK
    comptime TRACK = TrackId.T05
    # ---------------------------------------------------------------------------

    var config: BenchConfig
    comptime if SUITE.v == BenchSuite.GPU_BASELINE.v:
        config = _gpu_baseline_config()
    elif SUITE.v == BenchSuite.GPU_QUICK.v:
        config = _gpu_quick_config()
    elif SUITE.v == BenchSuite.CPU_BASELINE.v:
        config = _cpu_baseline_config()
    elif SUITE.v == BenchSuite.CPU_QUICK.v:
        config = _cpu_quick_config()
    else:
        # TRACK: use baseline timing for claims; T08 stays CPU quick/baseline via pack.
        comptime if TRACK.v == TrackId.T08.v:
            config = _cpu_baseline_config()
        else:
            config = _gpu_baseline_config()

    var m = Bench(config^)

    comptime if SUITE.v == BenchSuite.GPU_BASELINE.v:
        _register_gpu_pack[PACK_GPU_CORE](m)
    elif SUITE.v == BenchSuite.GPU_QUICK.v:
        _register_gpu_pack[PACK_GPU_1D](m)
    elif SUITE.v == BenchSuite.CPU_BASELINE.v:
        _register_cpu_pack[PACK_CPU_CORE](m)
    elif SUITE.v == BenchSuite.CPU_QUICK.v:
        _register_cpu_pack[PACK_CPU_QUICK](m)
    else:
        comptime if TRACK.v == TrackId.T00.v or TRACK.v == TrackId.T01.v:
            _register_gpu_pack[PACK_GPU_SM](m)
        elif TRACK.v == TrackId.T02.v:
            _register_gpu_pack[PACK_GPU_T02](m)
        elif (
            TRACK.v == TrackId.T04.v
            or TRACK.v == TrackId.T06.v
        ):
            _register_gpu_pack[PACK_GPU_1D](m)
            _register_gpu_pack[PACK_GPU_ND](m)
        elif TRACK.v == TrackId.T03.v:
            _register_gpu_pack[PACK_GPU_ND](m)
        elif TRACK.v == TrackId.T05.v:
            _register_gpu_pack[PACK_GPU_LARGE_AXIS](m)
        elif TRACK.v == TrackId.T07.v:
            _register_gpu_pack[PACK_GPU_SM](m)
        elif TRACK.v == TrackId.T08.v:
            _register_cpu_pack[PACK_CPU_QUICK](m)
        else:
            _register_gpu_pack[PACK_GPU_1D](m)

    print(m)
