from bit import bit_reverse
from complex import ComplexSIMD
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from math import log2, exp, pi, cos, sin, iota, sqrt
from sys import sizeof, argv
from sys.info import is_gpu


fn _get_twiddle_factors[
    stages: UInt, length: UInt, dtype: DType, base: UInt
](out res: InlineArray[ComplexSIMD[dtype, 1], length - 2]):
    alias C = ComplexSIMD[dtype, 1]
    res = __type_of(res)(uninitialized=True)
    alias N = length
    # for N in [base**i for i in range(1, stages + 1)]:
    #     for n in range(N // base):
    for n in range(1, N - 1):
        # exp((-j * 2 * pi * n) / N)
        theta = Scalar[dtype]((-2 * pi * n) / N)
        # TODO: this could be more generic using fputils
        res[n - 1] = C(cos(theta).__round__(15), sin(theta).__round__(15))
        print(res[n - 1].re, res[n - 1].im)


fn main():
    alias stages = 4
    alias length = 16
    alias out_dtype = DType.float64
    _ = _get_twiddle_factors[stages, length, out_dtype, 2]()
