# Major improvements kept (algorithmic)

Focus: make **ND 640×480** competitive without destroying **1D 1024**.
Code hubs: `fft/fft/_ndim_fft_gpu.mojo`, `fft/fft/fft.mojo`, butterflies in
`fft/fft/_fft.mojo`, layout helpers in `fft/fft/_fft_sm_layout.mojo`.

## 1. Skip full-volume transpose for rect 2D (SBRC column path)

**Idea:** After contiguous **row** FFTs along 480, run **column** FFTs along
640 with a tile of adjacent columns so global loads/stores coalesce on the
fast ortho axis — **no** standalone transpose + restore kernels.

**Gate today:** explicit `is_rect_sbrc_2d` when spatial shape is **640×480**
(see specialization doc). Equal-cube SBRC remains a separate, more general
pow2 path.

**Why it won:** Transpose/restore were a large fraction of ND time on Pascal.

## 2. Half-LDS Stockham for the column tile

**Idea:** One shared-memory line buffer (not ping-pong) with
load→regs→barrier→store, requiring **`n_work ≤ 1`** (every stage’s
`ceil(n_bfly / tpt) == 1`). Mid-stage multi-pass half-LDS **clobbers** inputs
— do not reintroduce without a correct multi-buffer scheme.

**Params kept:** **L=8** lines/block, **tpt=80**, bases **`[8,8,10]`** for 640.

**Why:** Dual-SM L=4 fit two tiles but lost to half-LDS L=8; L=9 and L=8@96
were slower; incomplete `n_work` looked fast but was **wrong**.

## 3. Stage0 from global + last stage write-through

**Idea:** For half-LDS SBRC:

1. **Stage 0** reads Stockham inputs from **global** with `axis_stride`
   (skip ept gather into SM).
2. Mid stages stay SM half-LDS.
3. **Last stage** stores butterflies to **global** with `out_complex_stride =
   axis_stride` (skip SM store + ept scatter).
4. Stage0: **no** barrier before SM store (loads are global-only); one barrier
   after SM store before mid stages.

**Why:** Gather/scatter ept loops and an extra SM round-trip were pure tax
once butterflies already touched global with the right stride.

## 4. Grid parallelism over serial batch loops

**Pascal lesson:** With ~1 resident SBRC block/SM, a small grid doing
**hundreds of serial tile-groups** pays barriers repeatedly on one SM.

**Keep:**

- Half-LDS SBRC: `sbrc_batch_size = sbrc_num_groups` (one group per block).
- Rect **row** axis: `batch_size = batches` (one FFT line per block).

**Failed:** packing 2 row lines/block (~25 ms); row half-LDS (~11.3 ms).

## 5. Factorization / thread map for 480 and 640

Hard-coded GPU bases (see specialization):

- **640 → `[8,8,10]`** — enables half-LDS `n_work≤1` at tpt=80.
- **480 → `[8,6,10]`** with **tpt=80** (`_stockham_intra` special in
  `_utils.mojo`) — fewer stages + exact/partial reg butterflies.

Generic estimator still exists for other lengths; these two lengths **bypass**
it on GPU.

## 6. Supporting keep pieces (less ND-specific)

- Closed / even-odd length-R DFTs for small R (fewer temps; tests green).
- Partial reg-bfly with **runtime** `n_work` loop (comptime unroll blew regs).
- Non-square transpose **TILE_Y=8**, skew-y, TILE+1 bank pad (when T/restore
  still used).
- Soft **1024** thread/block cap in SBRC sizing.
- `skip_volume_transpose` geo-wide when rect SBRC (or equal-cube / warp FS)
  so stage accounting matches.

## What did **not** stay

Fuse-back row+transpose, wide packing, dropping SBRC bank pad, warp-shuffle
Stockham, many radix permutations — see
[`05-failed-levers-summary.md`](05-failed-levers-summary.md).
