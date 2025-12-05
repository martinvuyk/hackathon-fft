from algorithm import parallelize, vectorize
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.info import Vendor, is_cpu
from layout import Layout, LayoutTensor, IntTuple
from utils.index import IndexList
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, size_of, simd_width_of
from math import ceil

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _max,
    _min,
    _mixed_radix_digit_reverse,
)
from ._fft import (
    _radix_n_fft_kernel,
    _launch_inter_or_intra_multiprocessor_fft,
    _intra_block_fft_kernel_radix_n,
    _inter_multiprocessor_fft_kernel_radix_n,
)


@always_inline
@parameter
fn _product_of_dims(dims: IntTuple) -> Int:
    """Calculates the product of a tuple of dimensions."""
    var prod = 1
    for i in range(len(dims)):
        prod *= dims[i].value()
    return prod


fn _run_cpu_nd_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    inverse: Bool,
    bases: List[List[UInt]],
](
    output: LayoutTensor[out_dtype, out_layout, out_origin, **_],
    x: LayoutTensor[in_dtype, in_layout, in_origin, **_],
    *,
    cpu_workers: Optional[UInt] = None,
):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime last_dim_idx = rank - 3

    comptime batches = out_layout.shape[0].value()
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    # print("GLOBAL DEBUG: x.layout:", x.layout, "output.layout:", output.layout)

    @no_inline
    @parameter
    fn _run_1d_fft[
        dtype_in: DType,
        layout_out: Layout,
        layout_in: Layout,
        x_in_origin: ImmutOrigin, //,
        dim_idx: Int,
    ](
        shared_f: LayoutTensor[out_dtype, layout_out, out_origin, **_],
        x_in: LayoutTensor[dtype_in, layout_in, x_in_origin, **_],
        enable_debug: Bool = False,
    ):
        comptime length = UInt(layout_in.shape[0].value())

        # print(
        #     "1D FFT: dim_idx:",
        #     dim_idx,
        #     "FFT Length N:",
        #     length,
        #     "x_in.layout:",
        #     layout_in,
        #     "shared_f.layout:",
        #     layout_out,
        # )

        comptime bases_idx = bases[dim_idx]
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases_idx
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]

        @parameter
        fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
            comptime last_base = ordered_bases[len(ordered_bases) - 1]
            var mat_bases = materialize[ordered_bases]()
            var c = Int((length // last_base) * (last_base - 1))
            var offsets = List[UInt](capacity=c * len(mat_bases))
            var val = UInt(0)
            for base in mat_bases:
                offsets.append(val)
                val += (length // base) * (base - 1)
            return val, offsets^

        comptime total_offsets = _calc_total_offsets()
        comptime total_twfs = total_offsets[0]
        comptime twf_offsets = total_offsets[1]
        comptime twfs_array = _get_flat_twfs[
            out_dtype,
            length,
            total_twfs,
            ordered_bases,
            processed_list,
            inverse,
        ]()
        comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
        var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
            twfs_array.unsafe_ptr()
        )
        comptime is_last_dim = dim_idx == last_dim_idx
        """We are running the ffts from right to left in the layout."""

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime func = _radix_n_fft_kernel[
                do_rfft = is_last_dim and x_complex_in == 1,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
                ordered_bases=ordered_bases,
            ]

            for local_i in range(length // base):
                var x_out_array = InlineArray[Scalar[out_dtype], Int(base) * 2](
                    uninitialized=True
                )
                var x_out = LayoutTensor[
                    out_dtype, Layout.row_major(Int(base), 2)
                ](x_out_array.unsafe_ptr())

                func(
                    shared_f,
                    x_in,
                    UInt(local_i),
                    twfs,
                    x_out,
                    enable_debug=enable_debug,
                    i_to_debug=0,
                )

    var inter_layer_buffer: List[Scalar[out_dtype]]

    @parameter
    if len(dims) == 1:
        inter_layer_buffer = {}
    else:
        inter_layer_buffer = {capacity = output.size()}

    @no_inline
    @parameter
    fn _run_batch(block_num: Int):
        var base_out_ptr = output.ptr + output.stride[0]() * block_num
        var base_inter_layer_ptr = (
            inter_layer_buffer.unsafe_ptr() + output.stride[0]() * block_num
        )

        # print("BATCHING DEBUG: dims:", dims.__str__(), "prod:", prod)

        @parameter
        for idx in reversed(range(len(dims))):
            comptime dim_tuple = dims[idx]
            comptime dim = dim_tuple.value()

            @parameter
            if dim == 1:
                continue

            comptime out_dim_offset = output.stride[rank - 2 - idx]()
            comptime out_row_stride = output.stride[idx + 1]()
            comptime batch_prod = prod // dim
            comptime dim_out_layout = Layout(
                IntTuple(batch_prod, dim, 2),
                IntTuple(out_dim_offset, out_row_stride, 1),
            )
            comptime dim_out_tensor = LayoutTensor[
                out_dtype, dim_out_layout, origin_of(inter_layer_buffer)
            ]
            var out_copy = dim_out_tensor(base_inter_layer_ptr)

            @parameter
            if idx != last_dim_idx:
                comptime T = LayoutTensor[
                    out_dtype, dim_out_layout, address_space=_
                ]
                out_copy.copy_from(T(base_out_ptr))

            @parameter
            for inner_batch_n in range(batch_prod):
                comptime dim_batch_out_layout = Layout(
                    IntTuple(dim, 2), IntTuple(out_row_stride, 1)
                )
                var dim_batch_out = LayoutTensor[
                    out_dtype, dim_batch_out_layout, address_space=_
                ](base_out_ptr + out_dim_offset * inner_batch_n)

                # print(
                #     "out_dim_offset:",
                #     out_dim_offset,
                #     "base_out_ptr offset:",
                #     out_dim_offset * inner_batch_n,
                # )

                # We are running the ffts from right to left in the layout
                @parameter
                if idx == last_dim_idx:
                    comptime x_dim_offset = x.stride[rank - 2 - idx]()
                    comptime x_row_stride = x.stride[idx + 1]()
                    comptime dim_batch_x_layout = Layout(
                        IntTuple(dim, x_complex_in), IntTuple(x_row_stride, 1)
                    )

                    var base_x_ptr = x.ptr + x.stride[0]() * block_num

                    var dim_batch_x = LayoutTensor[
                        in_dtype, dim_batch_x_layout, address_space=_
                    ](base_x_ptr + x_dim_offset * inner_batch_n)

                    # print(
                    #     "x_dim_offset:",
                    #     x_dim_offset,
                    #     "base_x_ptr offset:",
                    #     x_dim_offset * inner_batch_n,
                    # )

                    _run_1d_fft[idx](
                        dim_batch_out,
                        dim_batch_x,
                        enable_debug=inner_batch_n == 0,
                    )
                else:
                    var dim_batch_out_input = LayoutTensor[
                        out_dtype,
                        dim_batch_out_layout,
                        origin_of(inter_layer_buffer),
                        address_space=_,
                    ](out_copy.ptr + out_dim_offset * inner_batch_n)

                    _run_1d_fft[idx](
                        dim_batch_out,
                        dim_batch_out_input.get_immutable(),
                        enable_debug=inner_batch_n == 0,
                    )

    parallelize[func=_run_batch](
        batches, 1  # Int(cpu_workers.or_else(UInt(parallelism_level())))
    )
