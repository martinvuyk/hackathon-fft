# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

## Sequential intra-block on an RTX 5090

The performance beats cufft for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):

```terminal
bench_intra_block_radix_n_rfft[[64, 2], 128], 33.56
bench_intra_block_radix_n_rfft[[32, 4], 128], 10.104
bench_intra_block_radix_n_rfft[[16, 8], 128], 5.199
bench_intra_block_radix_n_rfft[[16, 4, 2], 128], 4.421
bench_intra_block_radix_n_rfft[[8, 8, 2], 128], 5.009
bench_intra_block_radix_n_rfft[[8, 4, 4], 128], 3.009
bench_intra_block_radix_n_rfft[[8, 4, 2, 2], 128], 3.004
bench_intra_block_radix_n_rfft[[8, 2, 2, 2, 2], 128], 2.892
bench_intra_block_radix_n_rfft[[4, 4, 4, 2], 128], 3.211
bench_intra_block_radix_n_rfft[[4, 4, 2, 2, 2], 128], 2.938
bench_intra_block_radix_n_rfft[[4, 2, 2, 2, 2, 2], 128], 2.576
bench_intra_block_radix_n_rfft[[2], 128], 2.338
```

So the performance improvement depending on the radix used is ~ 85 - 93 %
(7 - 15 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.

## CPU implementation on an Intel i5-12600KF

These are several implementations with different default behaviour. Some use
multi-threading on by default depending on the platform's linalg libs.

```terminal
lib   100_000x128  1_000_000x128
numpy     44.663    431.364
scipy     35.113    345.956
pyfftw    31.148    805.163
```

Bellow are the results for different radix sizes, but for the sake of brevity
if we choose the best multi-threaded combination we get:

```terminal
bench_cpu_radix_n_rfft[[16, 8], 100_000, 128, workers=n], 34.431
bench_cpu_radix_n_rfft[[16, 8], 1_000_000, 128, workers=n], 271.565
```

Which means a ~ 21.5 % (1.27 x) improvement for long sequences compared to the
fastest implementation of the 3 (scipy). This is highly dependent on the chosen
radix factors.

This is all ignoring the cost of interop between Python and the C/C++
implementations. I tried setting up fftw locally but it was a headache, and
this is just to give an estimate overview of the benefits of this
implementation.

## Conclusions

We can setup a simple function to estimate the best bases using the learnings
of the benchmarking. In the case of GPUs, the more threads the better. In the
case of CPUs, the closer `length // base` to the amount of logical
cores the better.

It is worth noting that these benchmarks highlight the strengths of the other
implementations. This implementation runs a generic Cooley-Tukey algorithm, and
thus leaves the other implementations in the dust when the sequence length is
not a power of the common prime factors (usually 2, 3, 5, and 7), because it
lets the user choose other prime factors when the default ones don't have all
those needed by the given sequence length instead of falling back to a slower
algorithm.

The only limitation is that a radix factor bigger than 100 leads to compilation
issues, it might be solvable by adding one of those slower fallback algorithms;
but I would rather se users using padding or other common ways to adjust the
sequence to the more efficient versions. Other fallback algorithms can be
implemented independently from this one and shouldn't be called fft given they
aren't running the Cooley-Tukey algorithm.

### Other results

Results for the Mojo fft implementation when using a single thread on CPU:

```terminal
bench_cpu_radix_n_rfft[[64, 2], 100_000, 128, workers=1], 354.506
bench_cpu_radix_n_rfft[[64, 2], 1_000_000, 128, workers=1], 3438.661
bench_cpu_radix_n_rfft[[32, 4], 100_000, 128, workers=1], 106.815
bench_cpu_radix_n_rfft[[32, 4], 1_000_000, 128, workers=1], 1029.569
bench_cpu_radix_n_rfft[[16, 8], 100_000, 128, workers=1], 73.957
bench_cpu_radix_n_rfft[[16, 8], 1_000_000, 128, workers=1], 744.5
bench_cpu_radix_n_rfft[[16, 4, 2], 100_000, 128, workers=1], 77.536
bench_cpu_radix_n_rfft[[16, 4, 2], 1_000_000, 128, workers=1], 799.327
bench_cpu_radix_n_rfft[[8, 8, 2], 100_000, 128, workers=1], 81.54
bench_cpu_radix_n_rfft[[8, 8, 2], 1_000_000, 128, workers=1], 817.486
bench_cpu_radix_n_rfft[[8, 4, 4], 100_000, 128, workers=1], 83.961
bench_cpu_radix_n_rfft[[8, 4, 4], 1_000_000, 128, workers=1], 736.94
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 100_000, 128, workers=1], 98.817
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 1_000_000, 128, workers=1], 922.319
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 100_000, 128, workers=1], 113.178
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 1_000_000, 128, workers=1], 1194.745
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 100_000, 128, workers=1], 96.392
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 1_000_000, 128, workers=1], 936.58
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 100_000, 128, workers=1], 111.692
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 1_000_000, 128, workers=1], 1191.075
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 100_000, 128, workers=1], 136.179
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 1_000_000, 128, workers=1], 1420.303
bench_cpu_radix_n_rfft[[2], 100_000, 128, workers=1], 162.594
bench_cpu_radix_n_rfft[[2], 1_000_000, 128, workers=1], 1695.749
```

And these are the results when using multiple threads on CPU:

```terminal
bench_cpu_radix_n_rfft[[64, 2], 100_000, 128, workers=n], 87.337
bench_cpu_radix_n_rfft[[64, 2], 1_000_000, 128, workers=n], 821.523
bench_cpu_radix_n_rfft[[32, 4], 100_000, 128, workers=n], 29.619
bench_cpu_radix_n_rfft[[32, 4], 1_000_000, 128, workers=n], 288.772
bench_cpu_radix_n_rfft[[16, 8], 100_000, 128, workers=n], 34.431
bench_cpu_radix_n_rfft[[16, 8], 1_000_000, 128, workers=n], 271.565
bench_cpu_radix_n_rfft[[16, 4, 2], 100_000, 128, workers=n], 46.793
bench_cpu_radix_n_rfft[[16, 4, 2], 1_000_000, 128, workers=n], 385.535
bench_cpu_radix_n_rfft[[8, 8, 2], 100_000, 128, workers=n], 49.381
bench_cpu_radix_n_rfft[[8, 8, 2], 1_000_000, 128, workers=n], 411.286
bench_cpu_radix_n_rfft[[8, 4, 4], 100_000, 128, workers=n], 41.265
bench_cpu_radix_n_rfft[[8, 4, 4], 1_000_000, 128, workers=n], 418.261
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 100_000, 128, workers=n], 52.641
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 1_000_000, 128, workers=n], 534.925
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 100_000, 128, workers=n], 67.329
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 1_000_000, 128, workers=n], 669.196
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 100_000, 128, workers=n], 55.902
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 1_000_000, 128, workers=n], 574.819
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 100_000, 128, workers=n], 70.171
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 1_000_000, 128, workers=n], 700.39
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 100_000, 128, workers=n], 78.673
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 1_000_000, 128, workers=n], 799.775
bench_cpu_radix_n_rfft[[2], 100_000, 128, workers=n], 95.492
bench_cpu_radix_n_rfft[[2], 1_000_000, 128, workers=n], 987.712
```
