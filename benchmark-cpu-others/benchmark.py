import os
import numpy as np
import scipy.fft
import pyfftw
from timeit import repeat
import multiprocessing

from threadpoolctl import threadpool_limits

def run_batched_nd_benchmark(num_threads):
    shapes = [
        (100_000, 128),
        (100_000, 1024),
        (100, 16384),
        (100, 640, 480),
        (10, 1920, 1080),
        (1, 3840, 2160),
        (1, 7680, 4320),
        (100, 64, 64, 64),
        (10, 128, 128, 128),
        (1, 256, 256, 256),
        (1, 512, 512, 512),
    ]

    print(f"\n--- Running Benchmark (Threads: {num_threads}) ---")
    header = f"{'Shape':<25} | {'NumPy (ms)':<12} | {'SciPy (ms)':<12} | {'PyFFTW (ms)':<12}"
    print(header)
    print("-" * 70)

    for shape in shapes:
        data = np.random.randn(*shape).astype(np.float32)
        axes = tuple(range(1, len(shape)))
        
        # Adjust repeats based on size: large arrays = fewer repeats
        num_repeats = 2 if np.prod(shape) > 5e7 else 5

        # --- NumPy ---
        with threadpool_limits(limits=num_threads):
            t_numpy = (sum(repeat(lambda: np.fft.rfftn(data, axes=axes), repeat=num_repeats, number=1))/num_repeats)*1000

        # --- SciPy ---
        with threadpool_limits(limits=num_threads):
            t_scipy = (sum(repeat(lambda: scipy.fft.rfftn(data, axes=axes), repeat=num_repeats, number=1))/num_repeats)*1000

        # --- PyFFTW ---
        a = pyfftw.empty_aligned(shape, dtype='float32')
        a[:] = data
        fft_obj = pyfftw.builders.rfftn(a, axes=axes, planner_effort='FFTW_ESTIMATE', threads=num_threads)
        t_fftw = (sum(repeat(lambda: fft_obj(), repeat=num_repeats, number=1))/num_repeats)*1000

        print(f"{str(shape):<25} | {t_numpy:>12.3f} | {t_scipy:>12.3f} | {t_fftw:>12.3f}")

if __name__ == "__main__":
    pyfftw.interfaces.cache.enable()
    
    # 1. Single Threaded (workers=1)
    run_batched_nd_benchmark(num_threads=1)
    
    # 2. Multi-Threaded (workers=n)
    total_cores = multiprocessing.cpu_count()
    run_batched_nd_benchmark(num_threads=total_cores)