# Benchmarking and correctness guide

## Absolute rule: do not change git state

The optimizing agent **must not**:

- `git add` / `git commit` / `git commit --amend`
- `git checkout`, `git restore`, `git reset`, `git clean` on dirty FFT sources
- `git push` / force-push
- Change `git config`

Also stated in repo-root `AGENTS.md`. Record results in your own scratchpad
or in notes under this handover tree if appropriate — **not** via commits
unless a human explicitly asks later.

If you need a known-good tree, copy files manually (e.g. from a labeled
directory) — do **not** checkout over dirty work.

## Environment

From `fft/`:

```bash
# One-time / when GPU unlock needed
pixi run enable-nvidia-gpu   # writes fft/.nvidia-gpu-env (gitignored)

# Activation is hooked via pixi.toml [activation] scripts
pixi run tests               # mojo run -D ASSERT=warn tests.mojo
pixi run bench               # mojo run -O3 bench.mojo
```

Run **tests and bench serially** on machines with ≤32 GiB RAM; concurrent
heavy Mojo compiles have produced **exit 137**.

## Mojo bench switchboard (`fft/bench.mojo`)

Edit **only** the comptime switchboard in `main()`:

- `SUITE` — `GPU_BASELINE`, `GPU_QUICK`, `CPU_*`, or `TRACK`
- `TRACK` — `T00`…`T08` when `SUITE == TRACK`

Recent ND work used **`SUITE = TRACK`**, **`TRACK = T05`** (includes
100×640×480 and 100k×1024 among others — confirm pack registration in
`bench.mojo` before claiming).

For a full core pack claim: `SUITE = GPU_BASELINE`.

Do **not** invent one-off bench shapes inside `bench.mojo` without updating
the documented packs; use small standalone `.mojo` files for probes.

## Correctness gates

| Command / file | Role |
|----------------|------|
| `pixi run tests` | Standard suite (does **not** deeply stress 640×480 C2C the same way as the smoke) |
| `pixi run mojo run check_640x480_c2c.mojo` | 1×640×480 C2C CPU vs GPU; expect possible `max_abs≈0.12` at ATOL 1e-2; fail if ≫0.25 |
| Shape miniatures in `tests.mojo` | Smaller rect / cube coverage |

Keep wins only if **tests green** and C2C smoke acceptable under the above
rule.

## cuFFT / vendor comparator

Source: `cufft-benchmark-main/cufft_benchmark.cu` (+ Makefile).
The built binary is gitignored (`cufft-benchmark-main/.gitignore`); rebuild on
the target GPU with `make` — do not copy Pascal-linked executables.

- Align shapes with Mojo packs (`100×640×480`, `100000×1024`, cubes, etc.).
- Compare **same transform kind** (R2C vs C2C) as the Mojo bench you cite.

## Using Modular Mojo / MAX as reference

Allowed and encouraged:

- Mojo stdlib / GPU APIs (`DeviceContext`, `barrier`, `stack_allocation`,
  `TileTensor`, …)
- Skills under `.cursor/skills` (especially Mojo syntax + GPU fundamentals)
- Optional local Modular checkout if present under `.cursor/references/` for
  **API patterns only**

Prefer current Mojo conventions (`def`, `comptime`, `imm`/`mut`/`var`) over
outdated pretrained CUDA-in-Mojo guesses.

## Using other FFT reference code (generic guidance)

External GPU/CPU FFT trees may be available in the agent context. Use them
for **ideas**, not for copy-paste into this package:

1. **ND strategy taxonomy** — when to fuse transpose into a pass, when to use
   half shared memory, how to choose threads-per-transform vs lines-per-block.
2. **Shared-memory sizing** — bytes-aware limits, bank padding policies.
3. **Factorization / codelets** — radix sequences, twiddle policies.
4. **Do not** port CUDA `__shared__` / `<<<>>>` literally; map to Mojo GPU
   idioms via the Mojo GPU skill.

Always **re-measure on the 5090**; Pascal keep constants (48 KiB, L=8, tpt=80)
are hypotheses until proven.

## Profiling hooks already in `fft/pixi.toml`

- `pixi run profile-nsys` — builds `profile.mojo`, runs nsys (paths may need
  adjusting on the 5090 host).
- `pixi run profile-ncu` — Nsight Compute (hard-coded ncu path may differ).

Ad-hoc attach targets: `prof640.mojo`, `nd_phase_profile.mojo`.

Profiler outputs (`*.nsys-rep`, `*.ncu-rep`) should stay **untracked**.

## Suggested measurement contract for keep/revert

1. Note GPU name + driver + Mojo/MAX package versions.
2. Warmup + median (or harness `met (ms)`) on the same suite as the claim.
3. Run `pixi run tests` after any keep candidate.
4. For 640×480 path changes: run `check_640x480_c2c.mojo`.
5. Reject “wins” that break 1024 1D by more than noise unless ND gain is
   overwhelming and documented.
6. Prefer **clear** wins (roughly ≥2–3% and stable across re-runs) before
   replacing a keep.
