# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

The performance beats cufft in most scenarios for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n[[UInt(16), UInt(8)], 128], 15.277498781835268
bench_intra_block_radix_n[[UInt(16), UInt(4), UInt(2)], 128], 15.403353916408344
bench_intra_block_radix_n[[UInt(8), UInt(8), UInt(2)], 128], 8.062209865381824
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(4)], 128], 7.1380963145067
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 7.229129097039065
bench_intra_block_radix_n[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 7.173479453892216
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 5.590830209010177
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 5.872535374275572
bench_intra_block_radix_n[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 5.308980826841691
bench_intra_block_radix_n[[UInt(2)], 128], 5.225372688246764
```

So the performance improvement depending on the radix used is ~ 57 - 85 %
(2.3 - 6.8 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.
