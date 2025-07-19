from math import pi, cos, sin
from complex import ComplexSIMD


fn _get_twiddle_factors[
    length: UInt, dtype: DType
](out res: InlineArray[ComplexSIMD[dtype, 1], length - 1]):
    """Get the twiddle factors for the length.

    Examples:
        for a signal with 8 datapoints:
        the result is: [W_1_8, W_2_8, W_3_8, W_4_8, W_5_8, W_6_8, W_7_8]
    """
    alias C = ComplexSIMD[dtype, 1]
    res = __type_of(res)(uninitialized=True)
    alias N = length
    for n in range(1, N):
        # exp((-j * 2 * pi * n) / N)
        theta = Scalar[dtype]((-2 * pi * n) / N)
        # TODO: this could be more generic using fputils
        res[n - 1] = C(cos(theta).__round__(15), sin(theta).__round__(15))


fn main():
    var twfs = _get_twiddle_factors[8, DType.float64]()
    for i in range(len(twfs)):
        print(i, ":", twfs[i].re, twfs[i].im)
