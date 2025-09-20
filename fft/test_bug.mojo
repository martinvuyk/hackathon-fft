from complex import ComplexSIMD
from gpu import thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple
from testing import assert_almost_equal


fn _kernel[
    out_dtype: DType,
    out_layout: Layout,
    out_origin: MutableOrigin,
    address_space: AddressSpace,
    base: UInt,
    length: UInt,
    processed: UInt,
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=address_space
    ],
    local_i: UInt,
):
    alias offset = processed
    var n = local_i % offset + (local_i // offset) * (offset * base)

    alias Co = ComplexSIMD[out_dtype, 1]
    var x_out = InlineArray[Co, 2](uninitialized=True)
    if local_i == 0:

        @parameter
        if processed == 1:
            x_out[0] = Co(1, 0)
            x_out[1] = Co(1, 0)
        elif processed == 2:
            x_out[0] = Co(2, 0)
            x_out[1] = Co(2, 0)
        elif processed == 4:
            x_out[0] = Co(4, 0)
            x_out[1] = Co(4, 0)
    elif local_i == 1:
        x_out[0] = Co(0, 0)
        x_out[1] = Co(0, 0)
    elif local_i == 2:
        x_out[0] = Co(0, 0)
        x_out[1] = Co(0, 0)
    else:
        x_out[0] = Co(0, 0)
        x_out[1] = Co(0, 0)

    @parameter
    for i in range(base):
        # TODO: make sure this is the most efficient
        var idx = n + i * offset
        # var ptr = x_out.unsafe_ptr().bitcast[Scalar[out_dtype]]()
        # var res = (ptr + 2 * i).load[width=2]()
        # output.store(Int(idx), 0, res)
        output[idx, 0] = x_out[i].re
        output[idx, 1] = x_out[i].im


fn _test[target: StaticString]() raises:
    alias inverse = False
    alias dtype = DType.float64
    alias Co = ComplexSIMD[dtype, 1]
    var x_out_1 = InlineArray[Co, 8](
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
    )
    var x_out_2 = InlineArray[Co, 8](
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
    )
    var x_out_3 = InlineArray[Co, 8](
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
        Co(4, 0),
        Co(0, 0),
        Co(0, 0),
        Co(0, 0),
    )

    alias SIZE = 8
    alias TPB = SIZE
    alias BLOCKS_PER_GRID = (1, 1)
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(
        SIZE, 2
    ) if inverse else Layout.row_major(SIZE)
    alias out_layout = Layout.row_major(SIZE, 2)
    alias calc_dtype = dtype

    @parameter
    fn _eval(
        result: LayoutTensor[mut=True, out_dtype, out_layout],
        complex_out: InlineArray[Co, 8],
    ) raises:
        print("out: ", end="")
        for i in range(SIZE):
            if i == 0:
                print("[", result[i, 0], ", ", result[i, 1], sep="", end="")
            else:
                print(",", result[i, 0], ",", result[i, 1], end="")
        print("]")
        print("expected: ", end="")

        # gather all real parts and then the imaginary parts
        for i in range(SIZE):
            if i == 0:
                print("[", complex_out[i].re, ", ", sep="", end="")
            else:
                print(", ", complex_out[i].re, ", ", sep="", end="")
            print(complex_out[i].im, end="")
        print("]")
        for i in range(SIZE):
            assert_almost_equal(
                result[i, 0],
                complex_out[i].re.cast[out_dtype](),
                atol=1e-3,
                rtol=1e-5,
            )
            assert_almost_equal(
                result[i, 1],
                complex_out[i].im.cast[out_dtype](),
                atol=1e-3,
                rtol=1e-5,
            )

    with DeviceContext() as ctx:

        @parameter
        if target == "cpu":
            var out = List[Scalar[in_dtype]](length=SIZE * 2, fill=1)
            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                Span(out)
            )

            @parameter
            for j in range(3):
                for i in range(8 // 2):
                    _kernel[base=2, processed = 2**j, length=8](out_tensor, i)

                @parameter
                if j == 1:
                    _eval(out_tensor, x_out_1)
                elif j == 2:
                    _eval(out_tensor, x_out_2)
                elif j == 3:
                    _eval(out_tensor, x_out_3)
        else:
            var out = ctx.enqueue_create_buffer[out_dtype](
                SIZE * 2
            ).enqueue_fill(1)
            var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
                out.unsafe_ptr()
            )

            @parameter
            fn _gpu_launch[
                out_dtype: DType,
                out_layout: Layout,
                out_origin: MutableOrigin,
                address_space: AddressSpace,
            ](
                out_tensor: LayoutTensor[
                    out_dtype,
                    out_layout,
                    out_origin,
                    address_space=address_space,
                ]
            ):
                @parameter
                for i in range(3):
                    _kernel[
                        out_dtype,
                        out_layout,
                        out_origin,
                        address_space,
                        base=2,
                        processed = 2**i,
                        length=8,
                    ](out_tensor, thread_idx.x)

            ctx.enqueue_function[
                _gpu_launch[
                    out_dtype,
                    out_layout,
                    out_tensor.origin,
                    out_tensor.address_space,
                ]
            ](out_tensor, grid_dim=1, block_dim=8 // 2)

            ctx.synchronize()

            with out.map_to_host() as out_host:
                var tmp = LayoutTensor[mut=True, out_dtype, out_layout](
                    out_host.unsafe_ptr()
                )
                _eval(tmp, x_out_3)


fn main() raises:
    _test["gpu"]()
    print("gpu passed")
    _test["cpu"]()
    print("cpu passed")
