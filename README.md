# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

## Sequential intra-block on an RTX 5090

The performance beats cufft for sequential fft executions.
cufft takes ~36 miliseconds running 10k 128-point fft. These are the results
for my algorithm (the lists are the radixes used):

```terminal
bench_intra_block_radix_n_rfft[[64, 2], 128], 34.393
bench_intra_block_radix_n_rfft[[32, 4], 128], 9.551
bench_intra_block_radix_n_rfft[[16, 8], 128], 4.793
bench_intra_block_radix_n_rfft[[16, 4, 2], 128], 4.698
bench_intra_block_radix_n_rfft[[8, 8, 2], 128], 4.618
bench_intra_block_radix_n_rfft[[8, 4, 4], 128], 2.563
bench_intra_block_radix_n_rfft[[8, 4, 2, 2], 128], 2.899
bench_intra_block_radix_n_rfft[[8, 2, 2, 2, 2], 128], 2.657
bench_intra_block_radix_n_rfft[[4, 4, 4, 2], 128], 2.525
bench_intra_block_radix_n_rfft[[4, 4, 2, 2, 2], 128], 2.649
bench_intra_block_radix_n_rfft[[4, 2, 2, 2, 2, 2], 128], 2.356
bench_intra_block_radix_n_rfft[[2], 128], 1.976
```

So the performance improvement depending on the radix used is ~ 85 - 94 %
(7 - 18 x) in sequential executions. Plus the benefit of no dynamic
allocation or planning step which wasn't included in the benchmark.

## CPU implementation on an Intel i5-12600KF

These are several implementations with different default behavior. Some use
multi-threading on by default depending on the platform's linalg libs.

### Single-threaded

```terminal
Shape                     | NumPy (ms)   | SciPy (ms)   | PyFFTW (ms) 
----------------------------------------------------------------------
(100000, 128)             |       80.419 |       18.948 |       14.668
(100000, 1024)            |      721.269 |      161.764 |      177.246
(100, 16384)              |       17.529 |        3.997 |        4.287
(100, 640, 480)           |      427.481 |       91.605 |      107.713
(10, 1920, 1080)          |      334.407 |       70.449 |       94.191
(1, 3840, 2160)           |      151.040 |       35.203 |       55.885
(1, 7680, 4320)           |      682.552 |      186.163 |      307.622
(100, 64, 64, 64)         |      510.898 |       80.757 |       46.992
(10, 128, 128, 128)       |      417.480 |       69.197 |       44.463
(1, 256, 256, 256)        |      352.040 |       67.794 |      100.129
(1, 512, 512, 512)        |     3512.941 |      640.898 |     1362.859
```

This FFT implementation:
```terminal
bench_cpu_radix_n_rfft[(100000, 128), workers=1], 59.532
bench_cpu_radix_n_rfft[(100000, 1024), workers=1], 659.04
bench_cpu_radix_n_rfft[(100, 16384), workers=1], 15.716
bench_cpu_radix_n_rfft[(100, 640, 480), workers=1], 640.201
bench_cpu_radix_n_rfft[(10, 1920, 1080), workers=1], 541.857
bench_cpu_radix_n_rfft[(1, 3840, 2160), workers=1], 462.305
bench_cpu_radix_n_rfft[(1, 7680, 4320), workers=1], 3220.419
bench_cpu_radix_n_rfft[(100, 64, 64, 64), workers=1], 806.693
bench_cpu_radix_n_rfft[(10, 128, 128, 128), workers=1], 900.437
bench_cpu_radix_n_rfft[(1, 256, 256, 256), workers=1], 1201.321
```

### Multi-threaded

```terminal
Shape                     | NumPy (ms)   | SciPy (ms)   | PyFFTW (ms)
----------------------------------------------------------------------
(100000, 128)             |       78.862 |       19.667 |        7.649
(100000, 1024)            |      697.248 |      163.304 |       55.119
(100, 16384)              |       13.660 |        4.065 |        1.483
(100, 640, 480)           |      432.518 |       91.246 |       33.092
(10, 1920, 1080)          |      351.934 |       68.214 |       22.578
(1, 3840, 2160)           |      158.390 |       37.060 |       14.928
(1, 7680, 4320)           |      675.804 |      183.549 |       57.947
(100, 64, 64, 64)         |      509.366 |       83.495 |       31.403
(10, 128, 128, 128)       |      419.321 |       69.900 |       23.582
(1, 256, 256, 256)        |      357.371 |       76.605 |       27.078
(1, 512, 512, 512)        |     3547.772 |      648.791 |      230.020
```

This FFT implementation:
```terminal
bench_cpu_radix_n_rfft[(100000, 128), workers=n], 14.539
bench_cpu_radix_n_rfft[(100000, 1024), workers=n], 147.371
bench_cpu_radix_n_rfft[(100, 16384), workers=n], 4.674
bench_cpu_radix_n_rfft[(100, 640, 480), workers=n], 175.633
bench_cpu_radix_n_rfft[(10, 1920, 1080), workers=n], 155.772
bench_cpu_radix_n_rfft[(1, 3840, 2160), workers=n], 109.511
bench_cpu_radix_n_rfft[(1, 7680, 4320), workers=n], 622.908
bench_cpu_radix_n_rfft[(100, 64, 64, 64), workers=n], 220.444
bench_cpu_radix_n_rfft[(10, 128, 128, 128), workers=n], 371.172
bench_cpu_radix_n_rfft[(1, 256, 256, 256), workers=n], 372.938
```

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
but I would rather see users using padding or other common ways to adjust the
sequence to the more efficient versions. Other fallback algorithms can be
implemented independently from this one and shouldn't be called fft given they
aren't running the Cooley-Tukey algorithm.

### Other results

Results for the Mojo fft implementation when using a single thread on CPU:

```terminal
bench_cpu_radix_n_rfft[[64, 2], 100_000, 128, workers=1], 434.76
bench_cpu_radix_n_rfft[[64, 2], 1_000_000, 128, workers=1], 4261.257
bench_cpu_radix_n_rfft[[32, 4], 100_000, 128, workers=1], 134.495
bench_cpu_radix_n_rfft[[32, 4], 1_000_000, 128, workers=1], 1301.183
bench_cpu_radix_n_rfft[[16, 8], 100_000, 128, workers=1], 115.614
bench_cpu_radix_n_rfft[[16, 8], 1_000_000, 128, workers=1], 1148.454
bench_cpu_radix_n_rfft[[16, 4, 2], 100_000, 128, workers=1], 88.025
bench_cpu_radix_n_rfft[[16, 4, 2], 1_000_000, 128, workers=1], 805.408
bench_cpu_radix_n_rfft[[8, 8, 2], 100_000, 128, workers=1], 110.555
bench_cpu_radix_n_rfft[[8, 8, 2], 1_000_000, 128, workers=1], 1082.624
bench_cpu_radix_n_rfft[[8, 4, 4], 100_000, 128, workers=1], 68.18
bench_cpu_radix_n_rfft[[8, 4, 4], 1_000_000, 128, workers=1], 686.936
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 100_000, 128, workers=1], 82.91
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 1_000_000, 128, workers=1], 808.935
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 100_000, 128, workers=1], 97.149
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 1_000_000, 128, workers=1], 956.262
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 100_000, 128, workers=1], 89.543
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 1_000_000, 128, workers=1], 843.657
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 100_000, 128, workers=1], 95.985
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 1_000_000, 128, workers=1], 1004.276
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 100_000, 128, workers=1], 121.523
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 1_000_000, 128, workers=1], 1127.333
bench_cpu_radix_n_rfft[[2], 100_000, 128, workers=1], 145.242
bench_cpu_radix_n_rfft[[2], 1_000_000, 128, workers=1], 1336.504
```

And these are the results when using multiple threads on CPU:

```terminal
bench_cpu_radix_n_rfft[[64, 2], 100_000, 128, workers=n], 101.729
bench_cpu_radix_n_rfft[[64, 2], 1_000_000, 128, workers=n], 992.453
bench_cpu_radix_n_rfft[[32, 4], 100_000, 128, workers=n], 41.277
bench_cpu_radix_n_rfft[[32, 4], 1_000_000, 128, workers=n], 356.703
bench_cpu_radix_n_rfft[[16, 8], 100_000, 128, workers=n], 28.802
bench_cpu_radix_n_rfft[[16, 8], 1_000_000, 128, workers=n], 306.103
bench_cpu_radix_n_rfft[[16, 4, 2], 100_000, 128, workers=n], 47.942
bench_cpu_radix_n_rfft[[16, 4, 2], 1_000_000, 128, workers=n], 412.996
bench_cpu_radix_n_rfft[[8, 8, 2], 100_000, 128, workers=n], 43.685
bench_cpu_radix_n_rfft[[8, 8, 2], 1_000_000, 128, workers=n], 430.616
bench_cpu_radix_n_rfft[[8, 4, 4], 100_000, 128, workers=n], 41.552
bench_cpu_radix_n_rfft[[8, 4, 4], 1_000_000, 128, workers=n], 399.767
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 100_000, 128, workers=n], 52.064
bench_cpu_radix_n_rfft[[8, 4, 2, 2], 1_000_000, 128, workers=n], 507.537
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 100_000, 128, workers=n], 76.505
bench_cpu_radix_n_rfft[[8, 2, 2, 2, 2], 1_000_000, 128, workers=n], 646.568
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 100_000, 128, workers=n], 63.228
bench_cpu_radix_n_rfft[[4, 4, 4, 2], 1_000_000, 128, workers=n], 571.997
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 100_000, 128, workers=n], 72.065
bench_cpu_radix_n_rfft[[4, 4, 2, 2, 2], 1_000_000, 128, workers=n], 699.373
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 100_000, 128, workers=n], 76.774
bench_cpu_radix_n_rfft[[4, 2, 2, 2, 2, 2], 1_000_000, 128, workers=n], 771.757
bench_cpu_radix_n_rfft[[2], 100_000, 128, workers=n], 96.144
bench_cpu_radix_n_rfft[[2], 1_000_000, 128, workers=n], 939.783
```
