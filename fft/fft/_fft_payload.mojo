"""Stockham stage TileTensor payloads."""

from layout import TileTensor
from structured_kernels.tile_types import TilePayload


@fieldwise_init
struct _FftStockhamPayload[
    out_dtype: DType,
    in_tile: type_of(TileTensor[mut=False, ...]),
    lhs_tile: type_of(TileTensor[mut=True, out_dtype, ...]),
    rhs_tile: type_of(TileTensor[mut=True, out_dtype, ...]),
](TilePayload, TrivialRegisterPassable):
    """Ping-pong lhs/rhs tiles plus the stage input buffer for one radix stage.

    `out_dtype` is the canonical floating dtype for radix kernels. Tile types
    are wired from the pipeline or callsite; `in_tile` is the immutable input view.
    """

    var lhs: Self.lhs_tile
    var rhs: Self.rhs_tile
    var x_in: Self.in_tile
