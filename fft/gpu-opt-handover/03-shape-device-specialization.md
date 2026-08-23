# Shape- and device-specific decisions (non-generality map)

This is the most important file for an RTX 5090 agent. Every item below is a
place where **generality was intentionally narrowed** for GTX 1060 / the
**640×480** pack. Prefer **parameterizing from `GPUInfo` + shape predicates**
over copying literal `640`/`480`/`49152` forever — but **re-bench** before
deleting a gate that still wins on 5090.

## A. Hard shape gates (highest arbitrariness)

### A1. `is_rect_sbrc_2d` — exactly 640×480, rank-2 spatial

**Where:** `fft/fft/_ndim_fft_gpu.mojo` (`_GPUExecConfig`).

**Condition (conceptual):** `amnt_dims == 2` and dims `[0]==640` and
`[1]==480`.

**What it enables:**

- Column **SBRC** even when not equal-cube pow2.
- **Half-LDS** (`sbrc_half_lds`).
- Fixed **L=8**, **tpt=80** for dim 640.
- **One group per block** SBRC grid.
- **One line per block** on the **row** (480) axis.
- Geo-wide **`skip_volume_transpose`**.

**Why arbitrary:** Other rectangles (e.g. 720×480, 512×512, 1920×1080) do
**not** get this path unless you generalize the predicate (SMEM fit, tpt,
bases with `n_work≤1`, grid policy).

**5090 direction:** Replace with something like:

- `can_rect_sbrc(d0,d1)` from SMEM budget, `max_threads`, and a bases plan
  with `max_n_bfly ≤ tpt`.
- Keep 640×480 as a **regression shape**, not the only enabler.

### A2. Forced bases for length 640 and 480 (GPU only)

**Where:** `fft/fft/fft.mojo` (estimate / ordered bases).

| Length | Forced bases | Motivation on 1060 |
|--------|--------------|--------------------|
| 640 | `[8,8,10]` | `n_bfly` max 80 → tpt=80 ⇒ half-LDS `n_work≤1` |
| 480 | `[8,6,10]` | Pair with tpt=80; beat longer factor lists |

**Bypasses** register-/warp-based `gpu_max_radix` estimator for these lengths.

**5090 direction:** Auto-search factorizations that satisfy
`max(N/R_i) ≤ chosen_tpt` and minimize stage count / measured time; keep
forced lists only as fallbacks.

### A3. Forced tpt=80 for dim 480

**Where:** `fft/fft/_utils.mojo` (`_stockham_intra_block` or sibling).

**Arbitrary:** other dims use occupancy/GCD logic; 480 is special-cased.

## B. Device-constant literals (Pascal-flavored)

| Literal | Where used | Meaning on 1060 | 5090 risk |
|---------|------------|-----------------|-----------|
| `49152` | SBRC smem gate / budget | 48 KiB block SMEM | 5090 often has **larger** per-block SMEM — dual-tile or larger L may win again |
| Soft `1024` block threads | SBRC max lines | Pascal-friendly | May raise toward device max |
| `dim_pad = dim + 1` | SBRC line stride | Bank conflicts | Re-validate; no-pad was **much** worse on 1060 |
| No cluster path | CC check | CC 6.1 | 5090 is SM90+ — cluster experiments become relevant |

Many other budgets correctly use `GPUInfo` (`sm_count`,
`threads_per_multiprocessor`, `shared_memory_per_multiprocessor`, etc.).
Prefer extending that pattern rather than new magic numbers.

## C. Algorithm policies tuned on this DUT (semi-general)

These are not hard-coded to 640×480 but were **accepted because they won on
1060 measurements**:

| Policy | Keep reason | Revisit on 5090 |
|--------|-------------|-----------------|
| Half-LDS requires `n_work≤1` | Correctness | Same rule; occupancy math changes |
| Prefer large grid over serial `batch_size` when SMEM-bound | Huge ND win | Still likely; thresholds differ |
| Stage0 global + last global write-through for half-LDS | Removed gather/scatter tax | Likely general for strided axes |
| Fuse-back / pack-multi-line **gated off** | Lost badly | May win with more SMEM/regs |
| Contig Bailey / warp four-step **gated** | Correct but slower than 1-tpt | Recheck |
| Twiddle inline vs runtime | Mixed; ND uses runtime in benches | Recheck |
| Transpose TILE 32×8, Y_REP=2 when used | Flat alternatives | Recheck if T returns |

## D. Launch / test hazards discovered on 1060

- **`_GPUTest.BLOCK`** can cause **launch OOR** on heavy SBRC kernels; benches
  and `check_640x480_c2c.mojo` avoid that path (`runtime_twfs=True`, normal
  plan).
- Host **exit 137** if tests + bench compile overlap with low free RAM —
  run serially.
- Never use **`git checkout`** on dirty `fft/fft/*.mojo` to “reset” — wiped
  keep trees more than once.

## E. Code search anchors for the 5090 agent

```text
is_rect_sbrc_2d
sbrc_half_lds
sbrc_lines_per_block
sbrc_batch_size
batch_size = Self.batches if
length == 640
length == 480
49152
sbrc_max_block_threads
skip_volume_transpose
```

Touch these deliberately when generalizing; add similar shapes one at a time
and only then merge predicates.
