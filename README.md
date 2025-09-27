# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

The performance beats cufft in most scenarios for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n[[UInt(16), UInt(8)], 128], 6.125515560793777
bench_intra_block_radix_n[[UInt(16), UInt(4), UInt(2)], 128], 4.981225745425455
bench_intra_block_radix_n[[UInt(8), UInt(8), UInt(2)], 128], 4.405644800986574
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(4)], 128], 3.327901031492003
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 3.2252928801940803
bench_intra_block_radix_n[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 3.1733623852118633
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 2.589278351005437
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 2.570056582356039
bench_intra_block_radix_n[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 2.491094167631473
bench_intra_block_radix_n[[UInt(2)], 128], 2.1865113248311907
```

So the performance improvement depending on the radix used is ~ 83 - 94 %
(5.9 - 16.5 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.
