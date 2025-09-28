from timeit import repeat
import numpy as np
import scipy
import pyfftw 

small = np.random.rand(100_000, 128)
big = np.random.rand(1_000_000, 128)

pyfftw.interfaces.cache.enable()

def numpy_fft(a):
    np.fft.rfft(a, axis=1)

def scipy_fft(a):
    scipy.fft.rfft(a)

def fftw_fft(a):
    pyfftw.interfaces.numpy_fft.rfft(a)

def run(fft):
    return (1000*sum(repeat(fft+'(small)', repeat=20, number=1, globals=globals()))/20,
            1000*sum(repeat(fft+'(big)', repeat=3, number=1, globals=globals()))/3)

def main():
    print('lib   100_000x128  1_000_000x128')
    print('numpy    %7.3f    %7.3f' % run('numpy_fft'))
    print('scipy    %7.3f    %7.3f' % run('scipy_fft'))
    print('pyfftw   %7.3f    %7.3f' % run('fftw_fft'))

if __name__ == "__main__":
    main()