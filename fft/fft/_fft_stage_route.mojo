"""Stage route types and shared kernel runners over route buffers."""

from layout import (
    TileTensor,
    TensorLayout,
    TensorStorage,
    row_major,
    stack_allocation,
)
from std.algorithm import vectorize
from std.sys.info import simd_width_of

from ._fft import (
    _radix_n_fft_kernel_butterfly,
    _radix_n_fft_kernel_butterfly_comptime,
    _radix_n_fft_kernel_elem_per_thread,
    _radix_n_fft_kernel_elem_per_thread_comptime,
)
from ._fft_payload import _FftStockhamPayload
from ._fft_pipeline import _Fft1dStageExec


trait _FftStageRoute(ImplicitlyDeletable):
    """Read/write routing view over a `_FftStockhamPayload`.

    # FIXME(#6810): WriteTile/ReadTile use UnsafeAnyOrigin because an abstract
    # `Self.origin: MutOrigin` cannot specialize `TileTensor`. Buffer methods
    # cast with `as_unsafe_any_origin`.
    """

    comptime out_dtype: DType
    comptime ConcretePayload: TrivialRegisterPassable

    # --- Payload.in_tile ---
    comptime in_layout: TensorLayout
    comptime in_storage: TensorStorage
    comptime in_address_space: AddressSpace
    comptime in_linear_idx_type: DType
    comptime in_tile = TileTensor[
        mut=False,
        Self.out_dtype,
        Self.in_layout,
        # FIXME(#6810)
        ImmutAnyOrigin,
        Storage=Self.in_storage,
        address_space=Self.in_address_space,
        linear_idx_type=Self.in_linear_idx_type,
    ]

    # --- Payload.lhs_tile ---
    comptime lhs_layout: TensorLayout
    comptime lhs_storage: TensorStorage
    comptime lhs_address_space: AddressSpace
    comptime lhs_linear_idx_type: DType
    comptime lhs_tile = TileTensor[
        mut=True,
        Self.out_dtype,
        Self.lhs_layout,
        # FIXME(#6810)
        MutAnyOrigin,
        Storage=Self.lhs_storage,
        address_space=Self.lhs_address_space,
        linear_idx_type=Self.lhs_linear_idx_type,
    ]

    # --- Payload.rhs_tile ---
    comptime rhs_layout: TensorLayout
    comptime rhs_storage: TensorStorage
    comptime rhs_address_space: AddressSpace
    comptime rhs_linear_idx_type: DType
    comptime rhs_tile = TileTensor[
        mut=True,
        Self.out_dtype,
        Self.rhs_layout,
        # FIXME(#6810)
        MutAnyOrigin,
        Storage=Self.rhs_storage,
        address_space=Self.rhs_address_space,
        linear_idx_type=Self.rhs_linear_idx_type,
    ]

    comptime payload_type = _FftStockhamPayload[
        Self.out_dtype, Self.in_tile, Self.lhs_tile, Self.rhs_tile
    ]

    # --- WriteTile (always mut) ---
    comptime write_layout: TensorLayout
    comptime write_storage: TensorStorage
    comptime write_address_space: AddressSpace
    comptime write_linear_idx_type: DType
    comptime WriteTile = TileTensor[
        mut=True,
        Self.out_dtype,
        Self.write_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=True],
        Storage=Self.write_storage,
        address_space=Self.write_address_space,
        linear_idx_type=Self.write_linear_idx_type,
    ]

    # --- ReadTile (always immutable) ---
    comptime read_dtype: DType
    comptime read_layout: TensorLayout
    comptime read_storage: TensorStorage
    comptime read_address_space: AddressSpace
    comptime read_linear_idx_type: DType
    comptime ReadTile = TileTensor[
        mut=False,
        Self.read_dtype,
        Self.read_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=False],
        Storage=Self.read_storage,
        address_space=Self.read_address_space,
        linear_idx_type=Self.read_linear_idx_type,
    ]

    def __init__(out self, payload: Self.ConcretePayload):
        ...

    @always_inline
    def write_buffer(self) -> Self.WriteTile:
        ...

    @always_inline
    def read_buffer(self) -> Self.ReadTile:
        ...


struct _FftRouteFirstWriteLhs[
    od: DType,
    concrete_in: type_of(TileTensor[mut=False, ...]),
    concrete_lhs: type_of(TileTensor[mut=True, od, ...]),
    concrete_rhs: type_of(TileTensor[mut=True, od, ...]),
](TrivialRegisterPassable, _FftStageRoute):
    comptime out_dtype = Self.od
    comptime ConcretePayload = _FftStockhamPayload[
        Self.od, Self.concrete_in, Self.concrete_lhs, Self.concrete_rhs
    ]

    comptime in_layout = Self.concrete_in.LayoutType
    comptime in_storage = Self.concrete_in.Storage
    comptime in_address_space = Self.concrete_in.address_space
    comptime in_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime lhs_layout = Self.concrete_lhs.LayoutType
    comptime lhs_storage = Self.concrete_lhs.Storage
    comptime lhs_address_space = Self.concrete_lhs.address_space
    comptime lhs_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime rhs_layout = Self.concrete_rhs.LayoutType
    comptime rhs_storage = Self.concrete_rhs.Storage
    comptime rhs_address_space = Self.concrete_rhs.address_space
    comptime rhs_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime write_layout = Self.concrete_lhs.LayoutType
    comptime write_storage = Self.concrete_lhs.Storage
    comptime write_address_space = Self.concrete_lhs.address_space
    comptime write_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime read_dtype = Self.concrete_in.dtype
    comptime read_layout = Self.concrete_in.LayoutType
    comptime read_storage = Self.concrete_in.Storage
    comptime read_address_space = Self.concrete_in.address_space
    comptime read_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime WriteTile = TileTensor[
        mut=True,
        Self.od,
        Self.write_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=True],
        Storage=Self.write_storage,
        address_space=Self.write_address_space,
        linear_idx_type=Self.write_linear_idx_type,
    ]
    comptime ReadTile = TileTensor[
        mut=False,
        Self.read_dtype,
        Self.read_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=False],
        Storage=Self.read_storage,
        address_space=Self.read_address_space,
        linear_idx_type=Self.read_linear_idx_type,
    ]

    var payload: Self.ConcretePayload

    def __init__(out self, payload: Self.ConcretePayload):
        self.payload = payload

    @always_inline
    def write_buffer(self) -> Self.WriteTile:
        return self.payload.lhs.as_unsafe_any_origin()

    @always_inline
    def read_buffer(self) -> Self.ReadTile:
        return self.payload.x_in.as_unsafe_any_origin()


struct _FftRouteFirstWriteRhs[
    od: DType,
    concrete_in: type_of(TileTensor[mut=False, ...]),
    concrete_lhs: type_of(TileTensor[mut=True, od, ...]),
    concrete_rhs: type_of(TileTensor[mut=True, od, ...]),
](TrivialRegisterPassable, _FftStageRoute):
    comptime out_dtype = Self.od
    comptime ConcretePayload = _FftStockhamPayload[
        Self.od, Self.concrete_in, Self.concrete_lhs, Self.concrete_rhs
    ]

    comptime in_layout = Self.concrete_in.LayoutType
    comptime in_storage = Self.concrete_in.Storage
    comptime in_address_space = Self.concrete_in.address_space
    comptime in_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime lhs_layout = Self.concrete_lhs.LayoutType
    comptime lhs_storage = Self.concrete_lhs.Storage
    comptime lhs_address_space = Self.concrete_lhs.address_space
    comptime lhs_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime rhs_layout = Self.concrete_rhs.LayoutType
    comptime rhs_storage = Self.concrete_rhs.Storage
    comptime rhs_address_space = Self.concrete_rhs.address_space
    comptime rhs_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime write_layout = Self.concrete_rhs.LayoutType
    comptime write_storage = Self.concrete_rhs.Storage
    comptime write_address_space = Self.concrete_rhs.address_space
    comptime write_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime read_dtype = Self.concrete_in.dtype
    comptime read_layout = Self.concrete_in.LayoutType
    comptime read_storage = Self.concrete_in.Storage
    comptime read_address_space = Self.concrete_in.address_space
    comptime read_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime WriteTile = TileTensor[
        mut=True,
        Self.od,
        Self.write_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=True],
        Storage=Self.write_storage,
        address_space=Self.write_address_space,
        linear_idx_type=Self.write_linear_idx_type,
    ]
    comptime ReadTile = TileTensor[
        mut=False,
        Self.read_dtype,
        Self.read_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=False],
        Storage=Self.read_storage,
        address_space=Self.read_address_space,
        linear_idx_type=Self.read_linear_idx_type,
    ]

    var payload: Self.ConcretePayload

    def __init__(out self, payload: Self.ConcretePayload):
        self.payload = payload

    @always_inline
    def write_buffer(self) -> Self.WriteTile:
        return self.payload.rhs.as_unsafe_any_origin()

    @always_inline
    def read_buffer(self) -> Self.ReadTile:
        return self.payload.x_in.as_unsafe_any_origin()


struct _FftRoutePingWriteLhs[
    od: DType,
    concrete_in: type_of(TileTensor[mut=False, ...]),
    concrete_lhs: type_of(TileTensor[mut=True, od, ...]),
    concrete_rhs: type_of(TileTensor[mut=True, od, ...]),
](TrivialRegisterPassable, _FftStageRoute):
    comptime out_dtype = Self.od
    comptime ConcretePayload = _FftStockhamPayload[
        Self.od, Self.concrete_in, Self.concrete_lhs, Self.concrete_rhs
    ]

    comptime in_layout = Self.concrete_in.LayoutType
    comptime in_storage = Self.concrete_in.Storage
    comptime in_address_space = Self.concrete_in.address_space
    comptime in_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime lhs_layout = Self.concrete_lhs.LayoutType
    comptime lhs_storage = Self.concrete_lhs.Storage
    comptime lhs_address_space = Self.concrete_lhs.address_space
    comptime lhs_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime rhs_layout = Self.concrete_rhs.LayoutType
    comptime rhs_storage = Self.concrete_rhs.Storage
    comptime rhs_address_space = Self.concrete_rhs.address_space
    comptime rhs_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime write_layout = Self.concrete_lhs.LayoutType
    comptime write_storage = Self.concrete_lhs.Storage
    comptime write_address_space = Self.concrete_lhs.address_space
    comptime write_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime read_dtype = Self.concrete_rhs.dtype
    comptime read_layout = Self.concrete_rhs.LayoutType
    comptime read_storage = Self.concrete_rhs.Storage
    comptime read_address_space = Self.concrete_rhs.address_space
    comptime read_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime WriteTile = TileTensor[
        mut=True,
        Self.od,
        Self.write_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=True],
        Storage=Self.write_storage,
        address_space=Self.write_address_space,
        linear_idx_type=Self.write_linear_idx_type,
    ]
    comptime ReadTile = TileTensor[
        mut=False,
        Self.read_dtype,
        Self.read_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=False],
        Storage=Self.read_storage,
        address_space=Self.read_address_space,
        linear_idx_type=Self.read_linear_idx_type,
    ]

    var payload: Self.ConcretePayload

    def __init__(out self, payload: Self.ConcretePayload):
        self.payload = payload

    @always_inline
    def write_buffer(self) -> Self.WriteTile:
        return self.payload.lhs.as_unsafe_any_origin()

    @always_inline
    def read_buffer(self) -> Self.ReadTile:
        return self.payload.rhs.as_immut().as_unsafe_any_origin()


struct _FftRoutePingWriteRhs[
    od: DType,
    concrete_in: type_of(TileTensor[mut=False, ...]),
    concrete_lhs: type_of(TileTensor[mut=True, od, ...]),
    concrete_rhs: type_of(TileTensor[mut=True, od, ...]),
](TrivialRegisterPassable, _FftStageRoute):
    comptime out_dtype = Self.od
    comptime ConcretePayload = _FftStockhamPayload[
        Self.od, Self.concrete_in, Self.concrete_lhs, Self.concrete_rhs
    ]

    comptime in_layout = Self.concrete_in.LayoutType
    comptime in_storage = Self.concrete_in.Storage
    comptime in_address_space = Self.concrete_in.address_space
    comptime in_linear_idx_type = Self.concrete_in.linear_idx_type

    comptime lhs_layout = Self.concrete_lhs.LayoutType
    comptime lhs_storage = Self.concrete_lhs.Storage
    comptime lhs_address_space = Self.concrete_lhs.address_space
    comptime lhs_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime rhs_layout = Self.concrete_rhs.LayoutType
    comptime rhs_storage = Self.concrete_rhs.Storage
    comptime rhs_address_space = Self.concrete_rhs.address_space
    comptime rhs_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime write_layout = Self.concrete_rhs.LayoutType
    comptime write_storage = Self.concrete_rhs.Storage
    comptime write_address_space = Self.concrete_rhs.address_space
    comptime write_linear_idx_type = Self.concrete_rhs.linear_idx_type

    comptime read_dtype = Self.concrete_lhs.dtype
    comptime read_layout = Self.concrete_lhs.LayoutType
    comptime read_storage = Self.concrete_lhs.Storage
    comptime read_address_space = Self.concrete_lhs.address_space
    comptime read_linear_idx_type = Self.concrete_lhs.linear_idx_type

    comptime WriteTile = TileTensor[
        mut=True,
        Self.od,
        Self.write_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=True],
        Storage=Self.write_storage,
        address_space=Self.write_address_space,
        linear_idx_type=Self.write_linear_idx_type,
    ]
    comptime ReadTile = TileTensor[
        mut=False,
        Self.read_dtype,
        Self.read_layout,
        # FIXME(#6810)
        UnsafeAnyOrigin[mut=False],
        Storage=Self.read_storage,
        address_space=Self.read_address_space,
        linear_idx_type=Self.read_linear_idx_type,
    ]

    var payload: Self.ConcretePayload

    def __init__(out self, payload: Self.ConcretePayload):
        self.payload = payload

    @always_inline
    def write_buffer(self) -> Self.WriteTile:
        return self.payload.rhs.as_unsafe_any_origin()

    @always_inline
    def read_buffer(self) -> Self.ReadTile:
        return self.payload.lhs.as_immut().as_unsafe_any_origin()


comptime _FftStageRouteType[
    is_first: Bool, write_lhs: Bool, P: type_of(_FftStockhamPayload)
] = (
    _FftRouteFirstWriteLhs[
        P.out_dtype, P.in_tile, P.lhs_tile, P.rhs_tile
    ] if is_first
    and write_lhs else _FftRouteFirstWriteRhs[
        P.out_dtype, P.in_tile, P.lhs_tile, P.rhs_tile
    ] if is_first
    and not write_lhs else _FftRoutePingWriteLhs[
        P.out_dtype, P.in_tile, P.lhs_tile, P.rhs_tile
    ] if not is_first
    and write_lhs else _FftRoutePingWriteRhs[
        P.out_dtype, P.in_tile, P.lhs_tile, P.rhs_tile
    ]
)


# Conditional RouteType[Self.is_first, Self.write_lhs, …] does not monomorphize
# for exclusivity. These `R: _FftStageRoute` helpers check against abstract tiles.
@always_inline
def _run_butterfly_comptime_for_route[
    R: _FftStageRoute, exec: _Fft1dStageExec
](payload: R.ConcretePayload):
    var route = R(payload)
    comptime cfg = exec.config
    comptime for local_i in range(Int(cfg.length) // Int(cfg.base)):
        comptime func = _radix_n_fft_kernel_butterfly_comptime[
            R.out_dtype, cfg, UInt(local_i)
        ]
        var x_out = stack_allocation[R.out_dtype](row_major[Int(cfg.base), 2]())
        func(route.write_buffer(), route.read_buffer(), x_out)


@always_inline
def _run_butterfly_stage_for_route[
    R: _FftStageRoute, exec: _Fft1dStageExec, max_stack_seq_len: Int
](payload: R.ConcretePayload, twfs: TileTensor[mut=False, ...]):
    var route = R(payload)
    comptime x_out_layout = row_major[Int(exec.config.base), 2]()
    comptime iters = Int(exec.config.length) // Int(exec.config.base)
    comptime num_blocks = iters // Int(exec.config.processed)
    comptime full_unroll = min(Int(exec.config.processed), max_stack_seq_len)

    comptime width = simd_width_of[R.out_dtype]() // 2
    comptime for phase in range(full_unroll):
        @always_inline
        def _run_butterfly_ct[w: Int](local_i: Int) {mut route, imm twfs}:
            var x_out = stack_allocation[R.out_dtype](x_out_layout)
            var idx = UInt(local_i) * exec.config.processed + UInt(phase)
            _radix_n_fft_kernel_butterfly[
                R.out_dtype, exec.config, UInt(phase)
            ](
                route.write_buffer(),
                route.read_buffer(),
                idx,
                twfs,
                x_out,
            )

        vectorize[1, unroll_factor=width](Int(num_blocks), _run_butterfly_ct)

    for phase in range(full_unroll, Int(exec.config.processed)):
        @always_inline
        def _run_butterfly_rt[w: Int](local_i: Int) {mut route, imm twfs, imm phase}:
            var x_out = stack_allocation[R.out_dtype](x_out_layout)
            var idx = UInt(local_i) * exec.config.processed + UInt(phase)
            _radix_n_fft_kernel_butterfly[R.out_dtype, exec.config, None](
                route.write_buffer(),
                route.read_buffer(),
                idx,
                twfs,
                x_out,
            )

        vectorize[1, unroll_factor=width](Int(num_blocks), _run_butterfly_rt)


@always_inline
def _run_elem_comptime_for_route[
    R: _FftStageRoute, exec: _Fft1dStageExec
](payload: R.ConcretePayload):
    var route = R(payload)
    comptime for local_i in range(Int(exec.config.length)):
        comptime func = _radix_n_fft_kernel_elem_per_thread_comptime[
            R.out_dtype, exec.config, UInt(local_i)
        ]
        func(route.write_buffer(), route.read_buffer())


@always_inline
def _run_elem_per_thread_for_route[
    R: _FftStageRoute, exec: _Fft1dStageExec
](payload: R.ConcretePayload, twfs: TileTensor[mut=False, ...]):
    var route = R(payload)

    @always_inline
    def _run_elem[width: Int](local_i: Int) {imm}:
        _radix_n_fft_kernel_elem_per_thread[R.out_dtype, exec.config](
            route.write_buffer(), route.read_buffer(), UInt(local_i), twfs
        )

    comptime width = max(simd_width_of[R.out_dtype](), Int(exec.config.base))
    vectorize[1, unroll_factor=width](Int(exec.config.length), _run_elem)


@always_inline
def _run_elem_per_thread_once_for_route[
    R: _FftStageRoute, exec: _Fft1dStageExec
](payload: R.ConcretePayload, local_i: UInt, twfs: TileTensor[mut=False, ...],):
    var route = R(payload)
    _radix_n_fft_kernel_elem_per_thread[R.out_dtype, exec.config](
        route.write_buffer(), route.read_buffer(), local_i, twfs
    )


@fieldwise_init
struct _FftStageRouteParams[exec: _Fft1dStageExec](TrivialRegisterPassable):
    """Stage runner bound to an exec plan; payload tile types come from the arg.
    """

    comptime is_first = Self.exec.is_first
    comptime write_lhs = Self.exec.write_lhs

    @always_inline
    def run_butterfly_comptime(self, ref payload: _FftStockhamPayload):
        comptime Route = _FftStageRouteType[
            Self.is_first, Self.write_lhs, type_of(payload)
        ]
        _run_butterfly_comptime_for_route[Route, Self.exec](
            rebind[Route.ConcretePayload](payload)
        )

    @always_inline
    def run_butterfly_stage[
        max_stack_seq_len: Int
    ](
        self,
        ref payload: _FftStockhamPayload,
        twfs: TileTensor[mut=False, ...],
    ):
        comptime Route = _FftStageRouteType[
            Self.is_first, Self.write_lhs, type_of(payload)
        ]
        _run_butterfly_stage_for_route[Route, Self.exec, max_stack_seq_len](
            rebind[Route.ConcretePayload](payload), twfs
        )

    @always_inline
    def run_elem_comptime(self, ref payload: _FftStockhamPayload):
        comptime Route = _FftStageRouteType[
            Self.is_first, Self.write_lhs, type_of(payload)
        ]
        _run_elem_comptime_for_route[Route, Self.exec](
            rebind[Route.ConcretePayload](payload)
        )

    @always_inline
    def run_elem_per_thread(
        self, ref payload: _FftStockhamPayload, twfs: TileTensor[mut=False, ...]
    ):
        comptime Route = _FftStageRouteType[
            Self.is_first, Self.write_lhs, type_of(payload)
        ]
        _run_elem_per_thread_for_route[Route, Self.exec](
            rebind[Route.ConcretePayload](payload), twfs
        )

    @always_inline
    def run_elem_per_thread_once(
        self,
        ref payload: _FftStockhamPayload,
        local_i: UInt,
        twfs: TileTensor[mut=False, ...],
    ):
        comptime Route = _FftStageRouteType[
            Self.is_first, Self.write_lhs, type_of(payload)
        ]
        _run_elem_per_thread_once_for_route[Route, Self.exec](
            rebind[Route.ConcretePayload](payload), local_i, twfs
        )
