# GPU FFT optimization handover (GTX 1060 → RTX 5090)

This directory is a **handover pack** for an agent continuing GPU FFT work on a
different device (target: **RTX 5090**). It documents what was optimized on a
**GTX 1060 Max-Q (CC 6.1)**, where the code is **shape- or device-specific**,
what to keep in git, and how to benchmark without changing git state.

| File | Purpose |
|------|---------|
| [`00-git-tracking-decisions.md`](00-git-tracking-decisions.md) | Untracked files: recommend track vs ignore |
| [`01-device-baseline.md`](01-device-baseline.md) | DUT specs, measured Mojo vs cuFFT numbers |
| [`02-major-improvements.md`](02-major-improvements.md) | Keep algorithms and why they won |
| [`03-shape-device-specialization.md`](03-shape-device-specialization.md) | **Arbitrary / non-generic gates** (primary map for 5090) |
| [`04-benchmarking-guide.md`](04-benchmarking-guide.md) | How to bench + correctness; **no git commits** |
| [`05-failed-levers-summary.md`](05-failed-levers-summary.md) | Condensed “do not blindly retry” list |
| [`06-5090-generalization-notes.md`](06-5090-generalization-notes.md) | Suggested approach: specialize → generalize |

**Hard rule for the receiving agent:** do **not** `git add`, `git commit`,
`git checkout`, amend, push, or otherwise change git state. See
[`04-benchmarking-guide.md`](04-benchmarking-guide.md) and repo `AGENTS.md`.

**Do not `git checkout` dirty FFT sources.** Local keep snapshots may exist
outside this tree; restoring via checkout has wiped keep work before.

This pack does **not** include private scratchpad notes. You may create your
own `scratchpad/` (typically gitignored) for experiments.
