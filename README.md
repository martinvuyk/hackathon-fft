# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

The performance beats cufft in most scenarios for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):
```terminal
bench_intra_block_radix_n[[UInt(16), UInt(8)], 128], 44.08921483703704
bench_intra_block_radix_n[[UInt(16), UInt(4), UInt(2)], 128], 40.64003441724138
bench_intra_block_radix_n[[UInt(8), UInt(8), UInt(2)], 128], 27.23606703081395
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(4)], 128], 23.717161614
bench_intra_block_radix_n[[UInt(8), UInt(4), UInt(2), UInt(2)], 128], 28.01864506428571
bench_intra_block_radix_n[[UInt(8), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 32.51976655833333
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(4), UInt(2)], 128], 19.606640216393448
bench_intra_block_radix_n[[UInt(4), UInt(4), UInt(2), UInt(2), UInt(2)], 128], 24.126215451360544
bench_intra_block_radix_n[[UInt(4), UInt(2), UInt(2), UInt(2), UInt(2), UInt(2)], 128], 28.409911514285717
bench_intra_block_radix_n[[UInt(2)], 128], 28.2544759
```