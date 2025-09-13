from bit import bit_reverse
from complex import ComplexSIMD
from math import pi

from fft.utils import _get_ordered_items, _get_twiddle_factors


fn main():
    # alias items = _get_ordered_items[8, List[UInt](4, 2)]()
    # alias items = _get_twiddle_factors[8, DType.float64]()
    # print(0, "", ComplexSIMD[DType.float64, 1](1, 0))
    # for i in range(len(items)):
    #     print(i + 1, "", items[i])
    alias items = _get_ordered_items[6, [3, 2]]()
    for i in range(len(materialize[items]())):
        print(items[i])
