# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

The performance beats cufft in most scenarios for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n[[UInt(16), UInt(8)], 128], 41.94357820714286
bench_intra_block_radix_n[[UInt(16), UInt(4), UInt(2)], 128], 36.39850653124999
bench_intra_block_radix_n[[UInt(8), UInt(8), UInt(2)], 128], 24.8389934152039
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(4)], 128], 21.928405524074073
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 22.33799034528302
bench_intra_block_radix_n[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 22.92205127503771
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 16.762296832152916
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 17.652676974626864
bench_intra_block_radix_n[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 17.893813246969696
bench_intra_block_radix_n[[UInt(2)], 128], 15.934853541891892
```

So the performance improvement depending on the radix used is ~ 10 - 55 %
(1.1 - 2.25 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.
