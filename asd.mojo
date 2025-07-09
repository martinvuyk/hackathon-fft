from bit import bit_reverse
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import log2, exp, pi, cos, sin, iota, sqrt
from sys import sizeof, argv
from sys.info import is_gpu


fn _get_last_twiddle_factors[
    length: UInt, dtype: DType, base: UInt
](out res: InlineArray[ComplexSIMD[dtype, 1], length - 1]):
    """Get the twiddle factors for the last stage."""
    alias C = ComplexSIMD[dtype, 1]
    res = __type_of(res)(uninitialized=True)
    alias N = length
    for n in range(1, N):
        # exp((-j * 2 * pi * n) / N)
        theta = Scalar[dtype]((-2 * pi * n) / N)
        # TODO: this could be more generic using fputils
        res[n - 1] = C(cos(theta).__round__(15), sin(theta).__round__(15))
        print(res[n - 1].re, res[n - 1].im)


fn main():
    alias stages = 3
    alias length = 8
    alias out_dtype = DType.float64
    _ = _get_last_twiddle_factors[length, out_dtype, 2]()
