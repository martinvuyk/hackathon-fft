# Device baseline (GTX 1060 Max-Q) and measured numbers

## Device under test (optimization era)

| Field | Value |
|-------|-------|
| GPU | NVIDIA GeForce GTX 1060 with Max-Q Design |
| Compute capability | **6.1 (Pascal)** |
| VRAM | **6 GiB** |
| Typical SM count | **10** (Max-Q 1060) |
| Threads / SM (Pascal) | **2048** |
| Shared memory / block (device limit used in gates) | **48 KiB (49152 bytes)** hard-coded in several SBRC checks |
| Max threads / block (soft cap used) | **1024** |
| Warp size | 32 |
| No CUDA clusters | CC &lt; 9.0 — cluster sync path exists in code but is N/A here |
| Host | ~31 GiB RAM; overlapping heavy `mojo run` jobs can **exit 137** (OOM) |

**Important:** README or marketing tables for an RTX 5090 are **not** this baseline.
Re-measure everything on the 5090 before keep/revert decisions.

## What we optimized against

Primary ND pain shape: **`100 × 640 × 480`** (R2C in `pixi run bench` T05/quick packs;
C2C smoke via `check_640x480_c2c.mojo`).

Secondary: **`100000 × 1024`** 1D (near cuFFT already).

cuFFT reference on this DUT (from project comparator; treat as approximate):

| Shape (conceptual) | cuFFT ~ms | Notes |
|--------------------|-----------|-------|
| 100×640×480 | **~7.75** | Used as ND ratio denominator |
| 100k×1024 | **~10.66** | 1D nearly tied |

## Mojo keep numbers (end of GTX 1060 work)

| Shape | Mojo ~ms | vs cuFFT |
|-------|----------|----------|
| 100×640×480 | **~9.48** | **~1.22×** |
| 100k×1024 | **~10.8** | **~1.01×** |

Progression on 100×640×480 (approximate eras):

| Era | Approach | ~ms |
|-----|----------|-----|
| RTRT | Row FFT → transpose → col FFT → restore | ~13.3 |
| Dual-SM SBRC | L=4 @ tpt=96, skip T+restore | ~12.9 |
| Half-LDS SBRC | L=8 @ tpt=80 | ~12.2 |
| Exact-ept I/O | Drop bounds when `ept*tpt==dim` | ~11.55 |
| Grid parallelism | One SBRC group / one row line per block | ~10.35 |
| Last write-through | Last Stockham stage → global | ~10.21 |
| Stage0 from global | Skip coalesced gather | ~9.73 |
| Stage0 barrier trim | No barrier before stage0 SM store | **~9.48** |

## Device heuristics that repeatedly mattered (Pascal)

1. **48 KiB SMEM / block** — dual ping-pong for 640-pt lines cannot fit wide
   tiles; **half-LDS (one tile)** was required for L=8.
2. **SMEM-bound occupancy** — half-LDS L=8 @ ~41 KiB ⇒ **~1 block/SM**;
   serial group loops inside a block were catastrophic; **large grids** won.
3. **Register pressure** — large radices / wide ept → ptxas OOR or host OOM;
   `n_work≤1` (tpt ≥ max butterflies) kept half-LDS correct and launchable.
4. **No usable warp shuffle codegen** for needed `shfl.sync` forms — do not
   rely on shuffle-based Stockham on this DUT (and verify on 5090).
5. **Bank pad `dim+1` on SBRC lines** — dropping pad ~14.4 ms (bad); keep pad
   on Pascal.
6. **`_GPUTest.BLOCK` launch path** can OOR on heavy SBRC; production/bench
   uses `runtime_twfs=True` **without** forcing that test mode.

## Correctness note (640×480 float32)

CPU vs GPU C2C with ATOL `1e-2` may report `max_abs ≈ 0.12` and ~100 failing
bins; same under older RTRT. Treat **max_abs &gt; ~0.25** as a real fail;
`check_640x480_c2c.mojo` documents the expected smoke behavior.
