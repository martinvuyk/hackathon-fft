# An evolution of the Fast fourier transform for the Modular hackathon.

This is the improved and generalized version.

## GPU implementation

This is still very much a work in progress. The sequential execution is faster
than cufft, but the batched one is slower by factors of magnitude. Most of the
rework seems to be around twiddle factors and other memory-related design
decisions that I'm working through. I might also rework the whole threading
logic.

## CPU implementation on an Intel i5-12600KF

There is still some work to do around workload scheduling, and possibly adapt
some of the learnings of the GPU rework once it's done.

### Single-threaded (C2C transforms)

```terminal
Shape                     | NumPy (ms)   | SciPy (ms)   | PyFFTW (ms) 
----------------------------------------------------------------------
(1000000, 93)             |     2135.127 |      492.254 |      700.002
(1000000, 128)            |     1754.112 |      412.807 |      186.791
(100000, 1024)            |     1486.417 |      456.084 |      199.525
(100, 16384)              |       25.611 |        8.188 |        5.957
(100, 640, 480)           |      980.167 |      223.876 |      132.532
(10, 1920, 1080)          |      717.253 |      186.949 |      135.662
(1, 3840, 2160)           |      360.358 |       87.282 |       57.653
(1, 7680, 4320)           |     1431.354 |      399.102 |      220.496
(100, 64, 64, 64)         |     1069.090 |      232.883 |       93.738
(10, 128, 128, 128)       |      956.972 |      232.840 |      107.677
(1, 256, 256, 256)        |      957.737 |      206.384 |      105.353
(1, 512, 512, 512)        |     8437.041 |     2142.148 |      795.725
(1, 64, 64, 64, 64)       |     1008.089 |      202.837 |       88.391
(1, 25, 160, 160, 48)     |     1739.727 |      315.957 |      156.214
```

This FFT implementation:
```terminal
bench_cpu_radix_n_rfft[(1000000, 93), workers=1]         | 803.546582 
bench_cpu_radix_n_rfft[(1000000, 128), workers=1]        | 623.863241 
bench_cpu_radix_n_rfft[(100000, 1024), workers=1]        | 743.339861 
bench_cpu_radix_n_rfft[(100, 16384), workers=1]          | 11.69622   
bench_cpu_radix_n_rfft[(100, 640, 480), workers=1]       | 945.469851 
bench_cpu_radix_n_rfft[(10, 1920, 1080), workers=1]      | 748.498751 
bench_cpu_radix_n_rfft[(1, 3840, 2160), workers=1]       | 413.001765 
bench_cpu_radix_n_rfft[(1, 7680, 4320), workers=1]       | 3017.106932
bench_cpu_radix_n_rfft[(100, 64, 64, 64), workers=1]     | 858.177628 
bench_cpu_radix_n_rfft[(10, 128, 128, 128), workers=1]   | 1323.986322
bench_cpu_radix_n_rfft[(1, 256, 256, 256), workers=1]    | 1564.728128
bench_cpu_radix_n_rfft[(1, 512, 512, 512), workers=1]    | 27187.02211
bench_cpu_radix_n_rfft[(1, 64, 64, 64, 64), workers=1]   | 1266.103523
bench_cpu_radix_n_rfft[(1, 25, 160, 160, 48), workers=1] | 3777.050719
```

### Multi-threaded (C2C transforms)

```terminal
Shape                     | NumPy (ms)   | SciPy (ms)   | PyFFTW (ms)
----------------------------------------------------------------------
(1000000, 93)             |     1869.060 |      451.815 |      104.136
(1000000, 128)            |     1591.919 |      409.703 |      111.844
(100000, 1024)            |     1328.997 |      379.331 |       90.852
(100, 16384)              |       25.012 |        6.994 |        2.119
(100, 640, 480)           |      924.827 |      219.604 |       54.627
(10, 1920, 1080)          |      681.492 |      176.150 |       44.019
(1, 3840, 2160)           |      316.209 |       84.085 |       19.423
(1, 7680, 4320)           |     1420.042 |      380.953 |       90.482
(100, 64, 64, 64)         |     1064.321 |      245.154 |       40.413
(10, 128, 128, 128)       |      975.358 |      251.574 |       41.574
(1, 256, 256, 256)        |      967.775 |      211.001 |       47.859
(1, 512, 512, 512)        |     9316.329 |     2251.690 |      347.590
(1, 64, 64, 64, 64)       |     1015.117 |      210.924 |       40.835
(1, 25, 160, 160, 48)     |     1736.233 |      319.680 |       73.128
```

This FFT implementation:
```terminal
bench_cpu_radix_n_rfft[(1000000, 93), workers=n]         | 937.095326 
bench_cpu_radix_n_rfft[(1000000, 128), workers=n]        | 542.679772 
bench_cpu_radix_n_rfft[(100000, 1024), workers=n]        | 726.707623 
bench_cpu_radix_n_rfft[(100, 16384), workers=n]          | 11.66103   
bench_cpu_radix_n_rfft[(100, 640, 480), workers=n]       | 231.832635 
bench_cpu_radix_n_rfft[(10, 1920, 1080), workers=n]      | 158.007667 
bench_cpu_radix_n_rfft[(1, 3840, 2160), workers=n]       | 93.416824  
bench_cpu_radix_n_rfft[(1, 7680, 4320), workers=n]       | 772.801927 
bench_cpu_radix_n_rfft[(100, 64, 64, 64), workers=n]     | 276.27824  
bench_cpu_radix_n_rfft[(10, 128, 128, 128), workers=n]   | 188.769021 
bench_cpu_radix_n_rfft[(1, 256, 256, 256), workers=n]    | 322.044562 
bench_cpu_radix_n_rfft[(1, 512, 512, 512), workers=n]    | 6582.235364
bench_cpu_radix_n_rfft[(1, 64, 64, 64, 64), workers=n]   | 546.713272 
bench_cpu_radix_n_rfft[(1, 25, 160, 160, 48), workers=n] | 1076.988621
```

## Conclusions

Until I've made this faster I've got nothing to say, other than Mojo is awesome
and this code turned out to be incredibly generic. In one of my experiments I
also plugged in a fast hartley transform implementation and it worked like a
charm for 1D transforms (N-D isn't separable like the FFT).