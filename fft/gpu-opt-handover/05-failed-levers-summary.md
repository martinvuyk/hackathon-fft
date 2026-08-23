# Failed / reverted levers (condensed)

Full chronology lived in a gitignored scratchpad. Below is enough for an
RTX 5090 agent to **avoid naive retries** of known-bad ideas on the **same
algorithm family**, while still re-testing when hardware limits change
(more SMEM, more regs, clusters).

## Do not retry without a new hypothesis

| Attempt | Outcome on GTX 1060 |
|---------|---------------------|
| Mid-stage multi-pass **half-LDS** with `n_work>1` | **Incorrect** (clobber); looked fast once |
| Drop SBRC **bank pad** (`dim_pad=dim`) | ~14.4 ms (vs ~10) |
| Row **half-LDS** | ~11.3 ms (barriers &gt; occupancy) |
| Pack **2 row lines**/block + full grid | ~25.8 ms |
| Fuse-back row+transpose (many L/tpt) | Near-miss or OOR; packing tax |
| Wide SBRC gather **comptime unroll** (dual era) | Slower |
| SBRC **4× serial** batch in-block | Slower than grid parallel |
| Warp **shuffle** Stockham | NVPTX could not select needed shfl forms |
| Single-warp stage sync elision (`pass`) | Incorrect |
| Coalesced gather-before-Stockham on contig path | Extra barrier tax |
| Many 480/640 **radix permutations** | Usually worse than keep bases |
| Runtime ept / capping comptime unroll to cut regs | ILP loss on 1D/ND |
| `ptxas -maxrregcount` via compiler PATH wrapper | Mojo did not invoke wrapper |
| ND bank-pad for multi-warp everywhere | Large ND regressions |
| Equal-cube-style SBRC on all mixed ND | Large regressions |
| Skip-restore strided last store of dim0 | Correct but ~29 ms |

## Near-misses (worth **re**-probing on 5090 only)

- Fuse-back L≈6–12 with partial reg-bfly (~14.1–14.2 vs ~13.3 RTRT era) —
  closer if SMEM/regs improve.
- Dual-SM SBRC L=4 — correct; lost to half-LDS L=8 on 48 KiB.
- Contig Bailey / SM FFT-32 — correct; slower than keep 1-tpt row path.
- Larger L with more than 48 KiB/block SMEM.

## Correctness landmines

- Half-LDS **requires** `n_work≤1` with current in-place store pattern.
- `_GPUTest.BLOCK` launches can OOR on heavy kernels.
- Float32 640×480 CPU vs GPU ATOL 1e-2 may show ~0.12 max_abs without a
  logic bug — use a looser smoke bound, not silent ignore of huge errors.
