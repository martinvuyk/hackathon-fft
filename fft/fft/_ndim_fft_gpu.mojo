from std.algorithm import parallelize, vectorize
from std.builtin.globals import global_constant
from std.complex import ComplexScalar
from std.collections import OptionalReg
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    barrier,
    cluster_arrive_relaxed,
    cluster_wait,
)
from std.gpu.host import DeviceContext, DeviceBuffer, Dim, DeviceFunction
from std.gpu.host.info import Vendor, is_cpu, GPUInfo
from layout import Layout, LayoutTensor, IntTuple
from std.utils.index import IndexList
from layout.int_tuple import IntArray
from std.runtime.asyncrt import parallelism_level
from std.sys.info import has_accelerator, size_of, simd_width_of
from std.math import ceildiv
from std.utils import Variant

from ._utils import (
    _get_dtype,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _product_of_dims,
    _get_cascade_idxes,
    _get_twiddle_factors,
    _num_stages_end_of,
    _calc_batches_M_N,
)
from ._fft import _radix_n_fft_kernel_cooley_tukey, _radix_n_fft_kernel_stockham


@fieldwise_init
struct _GPUExecConfig[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
    dim_idx: Int,
]:
    comptime rank = Self.out_layout.rank()
    comptime dims = Self.out_layout.shape[1 : Self.rank - 1]
    comptime prod = _product_of_dims(Self.dims)
    """The product of the dimensions in the tensor."""
    comptime start_dim_idx = len(Self.dims) - 1
    """We are running the ffts from right to left in the layout."""
    comptime dim = Self.dims[Self.dim_idx].value()
    """The selected dimension to run the contiguous fft for."""

    comptime num_threads = UInt(Self.dim)
    """The total number of threads per worload."""
    comptime batches = UInt(
        Self.out_layout.shape[0].value() * Self.prod // Self.dim
    )
    """The total amount of batches in the workload."""

    comptime max_threads_per_block = UInt(Self.gpu_info.max_thread_block_size)
    comptime threads_per_m = Self.gpu_info.threads_per_multiprocessor
    comptime max_threads_available = UInt(
        Self.threads_per_m * Self.gpu_info.sm_count
    )

    comptime num_blocks = ceildiv(Self.num_threads, Self.max_threads_per_block)
    """The total number of blocks that need to be scheduled for the worload."""
    comptime shared_mem_per_m = UInt(
        Self.gpu_info.shared_memory_per_multiprocessor
    )
    comptime shared_mem_per_t = Self.shared_mem_per_m // UInt(
        Self.threads_per_m
    )
    comptime shared_mem_per_block = (
        Self.shared_mem_per_t * Self.max_threads_per_block
    )
    comptime shared_mem_per_warp = (Self.shared_mem_per_t * Self.warp_size)
    comptime shared_mem_per_cluster = (
        Self.shared_mem_per_block * Self.max_cluster_size
    )
    comptime warp_size = UInt(Self.gpu_info.warp_size)

    comptime can_run_in_warp = Self.num_threads <= Self.warp_size and (
        Self.test.or_else(_GPUTest.WARP).v == _GPUTest.WARP.v
    )

    comptime can_run_in_block = Self.num_threads <= (
        Self.max_threads_per_block
    ) and (
        Self.test.or_else(_GPUTest.BLOCK).v
        in (_GPUTest.BLOCK.v, _GPUTest.WARP.v)
    )

    comptime is_sm_90_or_newer = (
        Self.gpu_info.vendor == Vendor.NVIDIA_GPU
        and Self.gpu_info.compute >= 9.0
    )
    comptime can_run_in_block_cluster = Self.num_blocks <= (
        Self.max_cluster_size
    ) and Self.is_sm_90_or_newer and (
        Self.test.or_else(_GPUTest.CLUSTER).v == _GPUTest.CLUSTER.v
    )

    comptime dim_layout = Layout.row_major(Self.dim, 2)
    comptime dim_size = UInt(Self.dim_layout.size())
    comptime dim_byte_size = UInt(size_of[Self.out_dtype]()) * Self.dim_size

    comptime use_shared_mem = (
        2 * Self.dim_byte_size <= Self.shared_mem_per_warp
    ) or (2 * Self.dim_byte_size <= Self.shared_mem_per_block) or (
        2 * Self.dim_byte_size <= Self.shared_mem_per_cluster
    )

    comptime block_threads = ceildiv(Self.num_threads, Self.num_blocks)
    comptime thread_batch_size = Self.max_threads_available // Self.num_threads
    comptime batch_size = min(Self.batches, Self.thread_batch_size)

    comptime inline_twfs = Self.dim_size <= UInt(
        Self.gpu_info.max_registers_per_block // 2
    )

    comptime max_shared_mem_size = (
        Self.shared_mem_per_warp if (
            Self.can_run_in_warp
        ) else Self.shared_mem_per_block if (
            Self.can_run_in_block
        ) else Self.shared_mem_per_cluster
    )

    comptime use_shared_memory = 2 * Self.dim_size <= Self.max_shared_mem_size


def _use_shared_memory_fn[config: _GPUExecConfig, idx: Int]() -> Bool:
    return _GPUExecConfig[
        config.out_dtype,
        config.out_layout,
        config.inverse,
        config.bases,
        config.test,
        config.gpu_info,
        config.max_cluster_size,
        config.runtime_twfs,
        idx,
    ].use_shared_memory


@fieldwise_init
struct _GPUPlan[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    test: Optional[_GPUTest],
    gpu_info: GPUInfo,
    max_cluster_size: UInt,
    runtime_twfs: Bool,
](Copyable):
    comptime config[dim_idx: Int] = _GPUExecConfig[
        Self.out_dtype,
        Self.out_layout,
        Self.inverse,
        Self.bases,
        Self.test,
        Self.gpu_info,
        Self.max_cluster_size,
        Self.runtime_twfs,
        dim_idx,
    ]()

    var twfs_buffer: List[Optional[DeviceBuffer[Self.out_dtype]]]
    var calc_buf: DeviceBuffer[Self.out_dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime assert Self.config[0].threads_per_m > 0, (
            "Unknown number of threads per sm for the given device. "
            "It is needed in order to run the gpu implementation."
        )
        comptime out_size = Self.out_layout.size()
        self.calc_buf = ctx.enqueue_create_buffer[Self.out_dtype](out_size)

        comptime amnt_dims = len(Self.config[0].dims)
        self.twfs_buffer = {capacity = amnt_dims}
        comptime for dim_idx, dim in enumerate(Self.config[0].dims):
            comptime length = UInt(dim.value())
            comptime config = Self.config[dim_idx]
            comptime if config.inline_twfs or config.runtime_twfs:
                self.twfs_buffer.append(None)
                continue

            var twfs = ctx.enqueue_create_buffer[Self.out_dtype](
                Int(config.dim_size)
            )
            comptime twfs_array = _get_twiddle_factors[
                length, Self.out_dtype, Self.inverse
            ]()
            # FIXME(#5686): replace with this once it's solved
            # ref twfs_array_runtime = global_constant[twfs_array]()
            var twfs_array_runtime = materialize[twfs_array]()
            var ptr = twfs_array_runtime.unsafe_ptr()
            ctx.enqueue_copy(twfs, ptr.bitcast[Scalar[Self.out_dtype]]())
            self.twfs_buffer.append(twfs^)


def _transpose[
    dst_dtype: DType,
    dst_layout: Layout,
    dst_origin: MutOrigin,
    src_origin: ImmutOrigin,
    *,
    into_: Int,
    from_: Int,
    scheduled_batches: UInt,
](
    dst: LayoutTensor[dst_dtype, dst_layout, dst_origin],
    src: LayoutTensor[dst_dtype, dst_layout, src_origin],
):
    comptime dims = src.layout.shape[1 : src.rank - 1]

    comptime sizes = _calc_batches_M_N[dims, into_, from_]()
    comptime intra_fft_batches = sizes[0]
    comptime M = sizes[1]
    comptime N = sizes[2]

    # assuming a block size of 1024 threads
    comptime TILE = 32

    # x maps to N (columns), y maps to M (rows), z maps to Batches
    var tx = thread_idx.x
    var ty = thread_idx.y
    var bx = block_idx.x * TILE
    var by = block_idx.y * TILE
    var b = block_idx.z

    comptime tile_layout = Layout.row_major(TILE, TILE, 2)
    var shared_tile = LayoutTensor[
        src.dtype,
        tile_layout,
        MutExternalOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var tile_ptr = shared_tile.ptr

    comptime extra_fft_batches = UInt(src.layout.shape[0].value())
    comptime total_batches = intra_fft_batches * extra_fft_batches
    comptime scheduled_runs = ceildiv(total_batches, scheduled_batches)
    for batch in range(scheduled_runs):
        var current_batch = b + batch * scheduled_batches
        if current_batch >= total_batches:
            return

        var src_base = src.ptr + current_batch * M * N * 2
        var dst_base = dst.ptr + current_batch * M * N * 2

        var x_in = bx + tx
        var y_in = by + ty

        if x_in < N and y_in < M:
            var val = src_base.load[2]((y_in * N + x_in) * 2)
            tile_ptr.store((ty * TILE + (tx ^ ty)) * 2, val)

        barrier()

        var x_out = by + tx
        var y_out = bx + ty

        if x_out < M and y_out < N:
            var val = tile_ptr.load[2]((tx * TILE + (tx ^ ty)) * 2)
            dst_base.store((y_out * M + x_out) * 2, val)

        barrier()


@always_inline
def _intra_something_gpu_fft_kernel_radix_n_multi_dim[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    calc_buf_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
    config: _GPUExecConfig,
    shared_address_space: AddressSpace,
    stage_sync_fn: def(),
    runtime_twfs: Bool,
    dim_idx: Int,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[out_dtype, twf_layout, twf_origin],
    calc_buf: LayoutTensor[out_dtype, out_layout, calc_buf_origin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y

    comptime total_threads = config.block_threads * config.num_blocks
    comptime x_complex_in = in_layout.shape[config.rank - 1].value()

    comptime base_out_layout = Layout.row_major(config.dim, 2)
    comptime out_t = LayoutTensor[out_dtype, base_out_layout, ...]
    comptime base_out_t = out_t[output.origin]
    comptime base_calc_t = out_t[calc_buf.origin]

    comptime base_x_layout = Layout.row_major(config.dim, x_complex_in)
    comptime base_x_t = LayoutTensor[
        in_dtype, base_x_layout, x.origin, address_space=x.address_space
    ]

    comptime shared_f_t = type_of(
        LayoutTensor[
            out_dtype,
            config.dim_layout if (
                config.use_shared_memory
            ) else Layout.row_major(0),
            MutExternalOrigin,
            address_space=shared_address_space,
        ].stack_allocation()
    )

    var shared_f_lhs = shared_f_t.stack_allocation()
    var shared_f_rhs = shared_f_t.stack_allocation()

    comptime total_stages = _num_stages_end_of[
        bases, config.dims, 0, _use_shared_memory_fn[config, _]
    ]() + 2 * config.start_dim_idx

    @always_inline
    @parameter
    def _run_1d_fft(
        shared_f_lhs: LayoutTensor[mut=True, out_dtype, ...],
        shared_f_rhs: LayoutTensor[mut=True, out_dtype, ...],
        x: LayoutTensor[mut=False, ...],
    ):
        comptime assert shared_f_lhs.layout == shared_f_rhs.layout
        comptime length = UInt(x.layout.shape[0].value())
        comptime do_rfft = dim_idx == config.start_dim_idx and x_complex_in == 1
        comptime bases_processed = materialize[
            _get_ordered_bases_processed_list[length, bases[dim_idx]]()
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]

        comptime fft_stages = _num_stages_end_of[
            bases, config.dims, dim_idx + 1, _use_shared_memory_fn[config, _]
        ]()
        comptime prev_stages = fft_stages + (config.start_dim_idx - dim_idx)

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime func = _radix_n_fft_kernel_stockham[
                ...,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                ordered_bases=ordered_bases,
                runtime_twfs=runtime_twfs,
                inline_twfs=config.inline_twfs,
            ]

            comptime s = prev_stages + b
            comptime write_lhs = (total_stages - (s + 1)) % 2 == 0

            comptime if b == 0 and write_lhs:
                func(shared_f_lhs, x, global_i, twiddle_factors)
            elif b == 0:
                func(shared_f_rhs, x, global_i, twiddle_factors)
            elif write_lhs:
                func(shared_f_lhs, shared_f_rhs, global_i, twiddle_factors)
            else:
                func(shared_f_rhs, shared_f_lhs, global_i, twiddle_factors)

            stage_sync_fn()

    @always_inline
    @parameter
    def _run_ndim_fft(
        base_out: base_out_t, base_calc: base_calc_t, base_x: base_x_t
    ):
        comptime if not config.use_shared_memory:
            return _run_1d_fft(base_out, base_calc, base_x)

        comptime prev_stages = _num_stages_end_of[
            bases, config.dims, dim_idx + 1, _use_shared_memory_fn[config, _]
        ]() + (config.start_dim_idx - dim_idx)
        comptime write_global_lhs = (total_stages - (prev_stages + 1)) % 2 == 0

        comptime if dim_idx == config.start_dim_idx:
            _run_1d_fft(shared_f_lhs, shared_f_rhs, base_x)
        elif write_global_lhs:
            _run_1d_fft(shared_f_lhs, shared_f_rhs, base_calc)
        else:
            _run_1d_fft(shared_f_lhs, shared_f_rhs, base_out)

        comptime length_idx = UInt(config.dims[dim_idx].value())
        comptime bases_processed_idx = materialize[
            _get_ordered_bases_processed_list[length_idx, bases[dim_idx]]()
        ]()
        comptime num_stages = len(bases_processed_idx[0])

        comptime s = prev_stages + num_stages - 1
        comptime last_write_lhs = (total_stages - (s + 1)) % 2 == 0

        var c_num: SIMD[out_dtype, 2]
        comptime if last_write_lhs:
            c_num = shared_f_lhs.load[2](Int(global_i), 0)
        else:
            c_num = shared_f_rhs.load[2](Int(global_i), 0)

        comptime if write_global_lhs:
            base_out.store(Int(global_i), 0, c_num)
        else:
            base_calc.store(Int(global_i), 0, c_num)

    comptime batched_iters = max(config.batches // config.batch_size, 1)
    comptime x_stride = Int(config.dim) * x_complex_in
    comptime out_stride = Int(config.dim) * 2

    for i in range(batched_iters):
        var offset = Int(block_num + i * config.batch_size)
        var base_x = base_x_t(x.ptr + x_stride * offset)
        var base_out = out_t(output.ptr + out_stride * offset)
        var base_calc = out_t(calc_buf.ptr + out_stride * offset)
        _run_ndim_fft(base_out, base_calc, base_x)
        stage_sync_fn()

    comptime full_iters = batched_iters * config.batch_size
    comptime remainder = config.batches - full_iters

    comptime if remainder > 0:
        if block_num < remainder:
            var offset = Int(full_iters + block_num)
            var base_x = base_x_t(x.ptr + x_stride * offset)
            var base_out = out_t(output.ptr + out_stride * offset)
            var base_calc = out_t(calc_buf.ptr + out_stride * offset)
            _run_ndim_fft(base_out, base_calc, base_x)
        stage_sync_fn()


@fieldwise_init
struct _GPUTest(Movable):
    comptime BLOCK = Self(0)
    comptime WARP = Self(1)
    comptime DEVICE_WIDE = Self(2)
    comptime CLUSTER = Self(3)
    var v: UInt


def _run_gpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
    runtime_twfs: Bool,
    max_cluster_size: UInt,
    //,
    *,
    test: Optional[_GPUTest] = None,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout, _],
    x: LayoutTensor[mut=False, in_dtype, in_layout, _],
    ctx: DeviceContext,
    plan: _GPUPlan[
        out_dtype,
        out_layout,
        inverse,
        bases,
        test,
        ctx.default_device_info,
        max_cluster_size=max_cluster_size,
        runtime_twfs=runtime_twfs,
    ],
) raises:
    comptime assert (
        out_dtype.is_floating_point()
    ), "out_dtype must be floating point"
    comptime assert (
        has_accelerator()
    ), "The non-cpu implementation is for GPU only"

    var calc_buf = LayoutTensor[out_dtype, output.layout](
        plan.calc_buf.unsafe_ptr()
    )

    @always_inline
    @parameter
    def _schedule_run[dim_idx: Int]() raises:
        comptime config = plan.config[dim_idx]

        @always_inline
        def cluster_stage_sync_fn():
            cluster_arrive_relaxed()
            cluster_wait()

        @always_inline
        def block_or_warp_stage_sync_fn():
            barrier()

        comptime stage_sync_fn = block_or_warp_stage_sync_fn if (
            config.can_run_in_block or config.can_run_in_warp
        ) else cluster_stage_sync_fn
        comptime address_space = AddressSpace.SHARED if (
            config.can_run_in_block or config.can_run_in_warp
        ) else AddressSpace.SHARED_CLUSTER

        var twiddle_factors: LayoutTensor[
            mut=False, out_dtype, config.dim_layout, MutAnyOrigin
        ]
        comptime if not (config.inline_twfs or config.runtime_twfs):
            twiddle_factors = {
                plan.twfs_buffer.unsafe_get(dim_idx).value().unsafe_ptr()
            }
        else:
            twiddle_factors = {unsafe_ptr = {}}

        comptime block_func_batch = _intra_something_gpu_fft_kernel_radix_n_multi_dim[
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            in_layout=x.layout,
            out_layout=output.layout,
            in_origin=x.origin,
            out_origin=output.origin,
            twf_layout=twiddle_factors.layout,
            twf_origin=twiddle_factors.origin,
            calc_buf_origin=calc_buf.origin,
            inverse=inverse,
            bases=bases,
            config=config,
            stage_sync_fn=stage_sync_fn,
            shared_address_space=address_space,
            runtime_twfs=runtime_twfs,
            dim_idx=dim_idx,
        ]
        comptime grid_dim = (Int(config.num_blocks), config.batch_size)
        comptime run_cluster = config.can_run_in_block_cluster and (
            config.num_blocks > 1
        )

        ctx.enqueue_function[block_func_batch, block_func_batch](
            output,
            x,
            twiddle_factors,
            calc_buf,
            grid_dim=grid_dim,
            cluster_dim=OptionalReg[Dim](
                config.num_blocks
            ) if run_cluster else None,
            block_dim=config.block_threads,
        )

    @always_inline
    @parameter
    def _schedule_transpose[dim_idx: Int, *, forward: Bool]() raises:
        comptime config = plan.config[dim_idx]
        comptime into_ = dim_idx + Int(forward)
        comptime from_ = dim_idx + Int(not forward)
        comptime sizes = _calc_batches_M_N[
            config.dims, into_=into_, from_=from_
        ]()
        comptime intra_fft_batches = sizes[0]
        comptime M = sizes[1]
        comptime N = sizes[2]

        comptime TILE = 32
        comptime M_ = ceildiv(M, TILE)
        comptime N_ = ceildiv(N, TILE)
        comptime num_threads = N_ * M_ * TILE * TILE
        comptime extra_fft_batches = UInt(output.layout.shape[0].value())
        comptime total_batches = intra_fft_batches * extra_fft_batches
        comptime thread_batch_size = config.max_threads_available // num_threads
        comptime scheduled_batches = min(
            total_batches, max(thread_batch_size, 1)
        )

        comptime grid_dim = (N_, M_, scheduled_batches)
        comptime block_dim = (TILE, TILE, 1)

        comptime total_stages = _num_stages_end_of[
            bases, config.dims, 0, _use_shared_memory_fn[config, _]
        ]() + 2 * config.start_dim_idx
        comptime fft_stages = _num_stages_end_of[
            bases,
            config.dims,
            (0 if forward else dim_idx + 1),
            _use_shared_memory_fn[config, _],
        ]()
        comptime s = fft_stages + config.start_dim_idx + (
            dim_idx if forward else -(dim_idx + 1)
        )
        comptime write_lhs = (total_stages - (s + 1)) % 2 == 0

        comptime if write_lhs:
            comptime func = _transpose[
                dst_dtype=out_dtype,
                dst_layout=output.layout,
                dst_origin=output.origin,
                src_origin=calc_buf.origin,
                into_=into_,
                from_=from_,
                scheduled_batches=scheduled_batches,
            ]
            ctx.enqueue_function[func, func](
                output, calc_buf, grid_dim=grid_dim, block_dim=block_dim
            )
        else:
            comptime func = _transpose[
                dst_dtype=out_dtype,
                dst_layout=calc_buf.layout,
                dst_origin=calc_buf.origin,
                src_origin=output.origin,
                into_=into_,
                from_=from_,
                scheduled_batches=scheduled_batches,
            ]
            ctx.enqueue_function[func, func](
                calc_buf, output, grid_dim=grid_dim, block_dim=block_dim
            )

    comptime start_dim_idx = plan.config[0].start_dim_idx
    _schedule_run[start_dim_idx]()

    comptime for dim_idx in reversed(range(start_dim_idx)):
        _schedule_transpose[dim_idx, forward=False]()
        _schedule_run[dim_idx]()

    comptime for dim_idx in range(start_dim_idx):
        _schedule_transpose[dim_idx, forward=True]()
