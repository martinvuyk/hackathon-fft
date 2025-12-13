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
from memory import memcpy

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _mixed_radix_digit_reverse,
    _product_of_dims,
    _get_cascade_idxes,
)
from ._fft import (
    _radix_n_fft_kernel,
    _launch_inter_or_intra_multiprocessor_fft,
    _intra_block_fft_kernel_radix_n,
    _inter_multiprocessor_fft_kernel_radix_n,
)


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
    comptime last_dim_idx = len(dims) - 1

    comptime batches = out_layout.shape[0].value()
    comptime x_complex_in = in_layout.shape[rank - 1].value()

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

                func(shared_f, x_in, local_i, twfs, x_out)

    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    var inter_layer_buf: List[Scalar[out_dtype]]

    @parameter
    if len(dims) == 1:
        inter_layer_buf = {}
    else:
        inter_layer_buf = {unsafe_uninit_length = output.size()}

    @parameter
    fn _run_batch(block_num: Int):
        var block_offset = output.stride[0]() * block_num
        var base_out = LayoutTensor[
            out_dtype,
            Layout.row_major(output.layout.shape[1:]),
            address_space=_,
        ](output.ptr + block_offset)
        var base_inter_out = LayoutTensor[
            out_dtype,
            Layout.row_major(output.layout.shape[1:]),
            address_space=_,
        ](inter_layer_buf.unsafe_ptr() + block_offset)
        var base_x = LayoutTensor[
            in_dtype,
            Layout.row_major(x.layout.shape[1:]),
            address_space=_,
        ](x.ptr + x.stride[0]() * block_num)

        @parameter
        for idx in reversed(range(len(dims))):
            comptime dim_tuple = dims[idx]
            comptime dim = dim_tuple.value()
            comptime batch_prod = prod // dim
            __comptime_assert dim != 1, "no inner dimension should be of size 1"

            @parameter
            if idx != last_dim_idx:
                memcpy(
                    dest=base_inter_out.ptr,
                    src=base_out.ptr.address_space_cast[AddressSpace.GENERIC](),
                    count=prod * 2,
                )

            @parameter
            for inner_batch_n in range(batch_prod):
                comptime idxes = _get_cascade_idxes[
                    output.layout.shape[1:], (idx, rank - 2)
                ](inner_batch_n)
                var dim_batch_out = base_out.slice[
                    Slice(0, dim), Slice(0, 2), (idx, rank - 2)
                ](idxes)

                # We are running the ffts from right to left in the layout
                @parameter
                if idx == last_dim_idx:
                    var dim_batch_x = base_x.slice[
                        Slice(0, dim),
                        Slice(0, x_complex_in),
                        (idx, rank - 2),
                    ](idxes)
                    _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                else:
                    var dim_batch_inter_out = base_inter_out.slice[
                        Slice(0, dim), Slice(0, 2), (idx, rank - 2)
                    ](idxes).get_immutable()
                    _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)

    parallelize[func=_run_batch](
        batches, Int(cpu_workers.or_else(UInt(parallelism_level())))
    )
