# Notes for generalizing on RTX 5090

## Recommended working style

1. **Re-baseline** Mojo + cuFFT on the 5090 for `PACK_GPU_CORE` (at least
   100×640×480 and 100k×1024). Ignore 1060 absolute milliseconds.
2. **Leave keep gates intact** until you have a measured replacement.
3. For each new idea: **one shape rabbit hole** → extract a **predicate** →
   try **2–3 sibling shapes** → only then widen defaults.
4. Document new magic numbers the same way this pack documents `640`/`480`/
   `49152`.

## High-value generalization targets

### 1. Replace `is_rect_sbrc_2d` with a capability predicate

Inputs: `d0`, `d1`, dtype size, `shared_mem_per_block`, max threads, planned
bases, chosen `tpt`/`L`.

Outputs: whether half-LDS SBRC + skip-volume is legal and profitable.

Regression shapes: 640×480, plus 512×512, 720×480, 1024×768, etc.

### 2. Automatic bases for half-LDS

Search factor lists where `max_i (N / R_i) ≤ tpt` and stage count is small.
Use 640/`[8,8,10]` and 480/`[8,6,10]` as seeds, not as the only answers.

### 3. Revisit dual-tile SMEM on 5090

48 KiB forced half-LDS L=8. With more SMEM, dual ping-pong or larger L may
beat half-LDS — **re-bench**; do not assume Pascal’s winner.

### 4. Grid policy as a function of occupancy

Rule of thumb that won on 1060: if resident blocks/SM ≈ 1 due to SMEM, set
batch/group size so **grid carries work**. Encode with
`ceildiv(work, max_concurrent_blocks)` rather than `== num_groups` only for
one shape.

### 5. Stage0-global + last write-through

Likely **portable** for strided column FFTs once half-LDS (or ping-pong) is
chosen. Good candidate to keep even after shape gates widen.

### 6. SM90+ features

Cluster launch / distributed shared memory paths already exist in skeleton
form for newer GPUs — worth a dedicated track on 5090, separate from
640×480 rabbit holes.

## What “good” looks like

- ND 640×480 at or below cuFFT (or clearly closer than ~1.22×) **without**
  regressing 1D 1024 beyond noise.
- Same algorithm family helping **multiple** rectangles, not only 640×480.
- Fewer literal shape equals-checks; more `GPUInfo`-driven budgets.

## Collaboration with this pack

Update or add files under `fft/gpu-opt-handover/` when you introduce new
non-generic gates — future agents need the same honesty this pack tries to
provide. Still **do not commit** unless a human requests it.
