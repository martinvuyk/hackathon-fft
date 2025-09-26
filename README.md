# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

The performance beats cufft in most scenarios for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n[[UInt(16), UInt(8)], 128], 7.097101896504841
bench_intra_block_radix_n[[UInt(16), UInt(4), UInt(2)], 128], 5.977541498412071
bench_intra_block_radix_n[[UInt(8), UInt(8), UInt(2)], 128], 4.76041410243476
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(4)], 128], 3.91074582249146
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 3.6132732441107818
bench_intra_block_radix_n[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 3.509325970245853
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 3.61365104406267
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 3.1815356535965753
bench_intra_block_radix_n[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 3.089634219563748
bench_intra_block_radix_n[[UInt(2)], 128], 2.6155104162517215
```

So the performance improvement depending on the radix used is ~ 80 - 92 %
(5 - 13.7 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.
