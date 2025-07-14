from complex import ComplexSIMD


fn _get_test_values_8[
    complex_dtype: DType
](out res: List[Tuple[List[Int], List[ComplexSIMD[complex_dtype, 1]]]]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 8.
    """
    alias Complex = ComplexSIMD[complex_dtype, 1]
    res = [
        (
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(4),
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(4),
                Complex(0),
                Complex(0),
                Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11, 10),
            List(
                Complex(178, 0),
                Complex(-55.113, -10.929),
                Complex(20, -4),
                Complex(7.113, 25.071),
                Complex(22, 0),
                Complex(7.113, -25.071),
                Complex(20, 4),
                Complex(-55.113, 10.929),
            ),
        ),
        (
            List(4, 8, 15, 16, 23, 42, 0, 0),
            List(
                Complex(108, 0),
                Complex(-54.355, -2.272),
                Complex(12, -34),
                Complex(16.355, 27.728),
                Complex(-24, 0),
                Complex(16.355, -27.728),
                Complex(12, 34),
                Complex(-54.355, 2.272),
            ),
        ),
        (
            List(1, -1, 1, -1, 5, 4, 3, 2),
            List(
                Complex(14, 0),
                Complex(-5.414, 7.657),
                Complex(2, -2),
                Complex(-2.586, 3.657),
                Complex(6, 0),
                Complex(-2.586, -3.657),
                Complex(2, 2),
                Complex(-5.414, -7.657),
            ),
        ),
    ]


fn _get_test_values_4[
    complex_dtype: DType
](out res: List[Tuple[List[Int], List[ComplexSIMD[complex_dtype, 1]]]]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 4.
    """
    alias Complex = ComplexSIMD[complex_dtype, 1]
    res = [
        (
            List(0, 0, 0, 0),
            List(
                Complex(0),
                Complex(0),
                Complex(0),
                Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0),
            List(
                Complex(2),
                Complex(0),
                Complex(2),
                Complex(0),
            ),
        ),
        (
            List(1, -1, 1, -1),
            List(
                Complex(0),
                Complex(0),
                Complex(4),
                Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27),
            List(
                Complex(81, 0),
                Complex(-11, 20),
                Complex(13, 0),
                Complex(-11, -20),
            ),
        ),
        (
            List(4, 8, 15, 16),
            List(
                Complex(43, 0),
                Complex(-11, 8),
                Complex(5, 0),
                Complex(-11, -8),
            ),
        ),
        (
            List(5, 4, 3, 2),
            List(
                Complex(14, 0),
                Complex(2, -2),
                Complex(2, 0),
                Complex(2, 2),
            ),
        ),
    ]
