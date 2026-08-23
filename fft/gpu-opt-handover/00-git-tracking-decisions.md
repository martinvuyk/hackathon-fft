# Git tracking decisions (recommendations only)

**No git commits from the optimizing agent.** A human stages/commits when ready.

## Ignore rules (source of truth)

| Location | Ignores |
|----------|---------|
| `cufft-benchmark-main/.gitignore` | Built `cufft_benchmark` binary + `*.o` / `*.out` |
| `fft/.gitignore` | `.pixi`, `build/*`, `*.ll`, `*.mojoc`, `.nvidia-gpu-env`, profiler dumps |
| Root `.gitignore` | `.cursor/*`, `scratchpad/*`, traces, common profiler extensions |

Do **not** document binaries as “please ignore” elsewhere — put them in `.gitignore`.

## Recommend **track** (source / tooling)

| Path | Rationale |
|------|-----------|
| `fft/gpu-opt-handover/` | Handover for 5090 work |
| `fft/fft/_fft_sm_layout.mojo` | Live ND/SM layout helpers |
| `fft/fft/_fft_large_1d.mojo` | Two-upload / four-step 1D |
| `fft/check_640x480_c2c.mojo` | Smoke for rect-2D hot path |
| `fft/nd_phase_profile.mojo` | Multi-shape profiler attach |
| `fft/prof640.mojo` | Minimal 640×480 profiler attach |
| `AGENTS.md` | Agent rules (incl. no-commit) |
| `cufft-benchmark-main/*.cu`, Makefile, Dockerfile, … | Comparator **source** only |

`.cursorignore` is optional IDE hygiene.

## Already-tracked assets that are fine

| Path | Why keep |
|------|----------|
| `multi_thread_cpu.png`, `single_thread_cpu.png` | Referenced from root `README.md` |
| `cufft-benchmark-main/*.cu` (+ Makefile / Docker) | Rebuild comparator on each GPU |

## Cleanup already in the working tree

| Path | Status |
|------|--------|
| `fft/Butterfly 8 Input Example.jpg` | Deleted in WT (was a diagram asset; duplicate under `original_submission/`) |

## Modified keep sources (optimization surface)

`fft/fft/_ndim_fft_gpu.mojo`, `fft.mojo`, `_fft*.mojo`, `_utils.mojo`, `bench.mojo`, `tests.mojo`, and `cufft_benchmark.cu` source edits.
