from algorithm import parallelize, vectorize
from builtin.globals import global_constant
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
    _get_flat_twfs_inline,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _mixed_radix_digit_reverse,
    _product_of_dims,
    _get_cascade_idxes,
    _get_flat_twfs_total_offsets,
    _max,
)
from ._fft import _radix_n_fft_kernel, _radix_n_fft_kernel_exp_twfs_runtime

from benchmark import keep


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
    output: LayoutTensor[out_dtype, out_layout, out_origin, ...],
    x: LayoutTensor[in_dtype, in_layout, in_origin, ...],
    *,
    cpu_workers: Optional[UInt] = None,
):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime batches = UInt(out_layout.shape[0].value())
    comptime x_complex_in = in_layout.shape[rank - 1].value()

    @always_inline
    @parameter
    fn _run_1d_fft[
        dtype_in: DType,
        layout_out: Layout,
        layout_in: Layout,
        x_in_origin: ImmutOrigin,
        //,
        dim_idx: Int,
    ](
        shared_f: LayoutTensor[out_dtype, layout_out, out_origin, ...],
        x_in: LayoutTensor[dtype_in, layout_in, x_in_origin, ...],
        enable_debug: Bool = False,
    ):
        comptime length = UInt(layout_in.shape[0].value())
        comptime bases_idx = bases[dim_idx]
        comptime bases_processed = _get_ordered_bases_processed_list[
            length, bases_idx
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        comptime total_offsets = _get_flat_twfs_total_offsets(
            ordered_bases, length
        )
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
        # FIXME(#5686): replace with this once it's solved
        # ref twfs_array_runtime = global_constant[twfs_array]()
        # var twfs_array_runtime = materialize[twfs_array]()
        # var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        #     twfs_array_runtime.unsafe_ptr()
        # )
        comptime max_base = Int(_max(ordered_bases))
        var x_out_array = InlineArray[Scalar[out_dtype], max_base * 2](
            uninitialized=True
        )
        var x_out = LayoutTensor[out_dtype, Layout.row_major(max_base, 2)](
            x_out_array.unsafe_ptr()
        )

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime func = _radix_n_fft_kernel_exp_twfs_runtime[
                do_rfft = dim_idx == start_dim_idx and x_complex_in == 1,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                # twf_offset = twf_offsets[b],
                ordered_bases=ordered_bases,
            ]
            comptime num_iters = length // base

            for local_i in range(num_iters):
                # func(shared_f, x_in, local_i, twfs, x_out)
                func(shared_f, x_in, local_i, x_out)

    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    var inter_layer_buf = List[Scalar[out_dtype]](
        unsafe_uninit_length=output.size() * Int(len(dims) > 1)
    )
    comptime o_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, o_layout, address_space=_]
    var max_threads_available = cpu_workers.or_else(UInt(parallelism_level()))

    @parameter
    fn _run_batch(block_num: Int):
        var block_offset = output.stride[0]() * block_num
        var base_out = out_t(output.ptr + block_offset)
        var base_inter_out = out_t(inter_layer_buf.unsafe_ptr() + block_offset)
        comptime x_out_layout = Layout.row_major(x.layout.shape[1:])
        var base_x = LayoutTensor[in_dtype, x_out_layout, address_space=_](
            x.ptr + x.stride[0]() * block_num
        )

        @parameter
        if len(dims) == 1:
            _run_1d_fft[start_dim_idx](base_out, base_x)
        else:

            @parameter
            for idx in reversed(range(len(dims))):
                comptime dim_tuple = dims[idx]
                comptime dim = dim_tuple.value()
                comptime batch_prod = UInt(prod // dim)

                @parameter
                if idx != start_dim_idx:
                    comptime G = AddressSpace.GENERIC
                    var b_o = base_out.ptr.address_space_cast[G]()
                    memcpy(dest=base_inter_out.ptr, src=b_o, count=prod * 2)

                @parameter
                fn _run_dim_batch(flat_idx: Int):
                    comptime exclude = (idx, rank - 2)
                    comptime exclude_t = IntTuple(idx, rank - 2)
                    comptime dim_sl = Slice(0, dim)
                    comptime o_comp = Slice(0, 2)
                    comptime dims_comp = base_out.layout.shape
                    var idxes = _get_cascade_idxes[dims_comp, exclude_t](
                        flat_idx
                    )
                    var dim_batch_out = base_out.slice[dim_sl, o_comp, exclude](
                        idxes
                    )

                    @parameter
                    if idx == start_dim_idx:
                        comptime x_comp = Slice(0, x_complex_in)
                        var dim_batch_x = base_x.slice[dim_sl, x_comp, exclude](
                            idxes
                        )
                        _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                    else:
                        var dim_batch_inter_out = base_inter_out.slice[
                            dim_sl, o_comp, exclude
                        ](idxes).get_immutable()
                        _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)

                var num_workers = Int(min(max_threads_available, batch_prod))
                parallelize[func=_run_dim_batch](Int(batch_prod), num_workers)

    @parameter
    if len(dims) == 1:
        var num_workers = min(max_threads_available, batches)
        parallelize[func=_run_batch](Int(batches), Int(num_workers))
    else:
        for block_num in range(Int(batches)):
            _run_batch(block_num)
