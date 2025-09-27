# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

## intra-block on an RTX 5090

The performance beats cufft for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n_rfft[[UInt(16), UInt(8)], 128], 5.182051050126884
bench_intra_block_radix_n_rfft[[UInt(16), UInt(4), UInt(2)], 128], 4.32487476126847
bench_intra_block_radix_n_rfft[[UInt(8), UInt(8), UInt(2)], 128], 5.031442015010036
bench_intra_block_radix_n_rfft[[UInt(8), UInt(4), UInt(4)], 128], 3.001333106250981
bench_intra_block_radix_n_rfft[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 3.0087947918684463
bench_intra_block_radix_n_rfft[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 2.896257316511454
bench_intra_block_radix_n_rfft[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 3.2195414843013075
bench_intra_block_radix_n_rfft[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 2.9446249520569197
bench_intra_block_radix_n_rfft[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 2.582607618687464
bench_intra_block_radix_n_rfft[[UInt(2)], 128], 2.3323706165611897
```

So the performance improvement depending on the radix used is ~ 85 - 93 %
(7 - 15 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.
