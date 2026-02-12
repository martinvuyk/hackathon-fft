import os
import numpy as np
import scipy.fft
import pyfftw
from timeit import repeat
import multiprocessing
from threadpoolctl import threadpool_limits

def run_batched_nd_benchmark(num_threads):
    shapes = [
        (1_000_000, 93),
        (1_000_000, 128),
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
        (1, 64, 64, 64, 64),
        (1, 25, 160, 160, 48),
    ]

    print(f"\n--- Running C2C Benchmark (Threads: {num_threads}) ---")
    header = f"{'Shape':<25} | {'NumPy (ms)':<12} | {'SciPy (ms)':<12} | {'PyFFTW (ms)':<12}"
    print(header)
    print("-" * 70)

    for shape in shapes:
        # 1. Create Complex64 data (C2C requirement)
        # Using complex64 (2x float32) matches standard MRI reconstruction precision
        data = (np.random.randn(*shape) + 1j*np.random.randn(*shape)).astype(np.complex64)
        
        # We benchmark the FFT over the spatial dimensions (X, Y, Z), keeping Time as the batch
        axes = tuple(range(1, len(shape)))
        
        # Adjust repeats based on size (Complex data is 2x heavier than Real)
        num_repeats = 2 if np.prod(shape) > 2e7 else 5

        # --- NumPy (Uses fftn for C2C) ---
        with threadpool_limits(limits=num_threads):
            t_numpy = (sum(repeat(lambda: np.fft.fftn(data, axes=axes), repeat=num_repeats, number=1))/num_repeats)*1000

        # --- SciPy (Uses fftn for C2C) ---
        with threadpool_limits(limits=num_threads):
            t_scipy = (sum(repeat(lambda: scipy.fft.fftn(data, axes=axes), repeat=num_repeats, number=1))/num_repeats)*1000

        # --- PyFFTW (Uses fftn for C2C) ---
        # Note: 'complex64' in numpy maps to 'complex float' in FFTW
        a = pyfftw.empty_aligned(shape, dtype='complex64')
        a[:] = data
        # Using FFTW_MEASURE for more accurate (though slower to plan) results
        fft_obj = pyfftw.builders.fftn(a, axes=axes, planner_effort='FFTW_MEASURE', threads=num_threads)
        t_fftw = (sum(repeat(lambda: fft_obj(), repeat=num_repeats, number=1))/num_repeats)*1000

        print(f"{str(shape):<25} | {t_numpy:>12.3f} | {t_scipy:>12.3f} | {t_fftw:>12.3f}")

if __name__ == "__main__":
    # Enable cache to avoid re-planning if the same shape is called multiple times
    pyfftw.interfaces.cache.enable()
    
    # 1. Single Threaded
    run_batched_nd_benchmark(num_threads=1)
    
    # 2. Multi-Threaded
    total_cores = multiprocessing.cpu_count()
    run_batched_nd_benchmark(num_threads=total_cores)