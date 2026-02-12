from algorithm import parallelize, vectorize, parallel_memcpy
from builtin.globals import global_constant
from complex import ComplexScalar
from layout import Layout, LayoutTensor, IntTuple
from utils.index import IndexList
from layout.int_tuple import IntArray
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, size_of, simd_width_of
from math import ceil
from memory import memcpy, ArcPointer

from ._utils import (
    _get_dtype,
    _get_twiddle_factors,
    _mixed_radix_digit_reverse,
    _get_ordered_bases_processed_list,
    _mixed_radix_digit_reverse,
    _product_of_dims,
    _get_cascade_idxes,
    _max,
    _min,
)
from ._fft import _radix_n_fft_kernel


struct _CPUPlan[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
](Copyable):
    comptime dims = Self.out_layout.shape[1 : Self.out_layout.rank() - 1]
    comptime L = List[ComplexScalar[Self.out_dtype]]

    var inter_layer_buf: ArcPointer[Self.L]
    var twiddle_factors: ArcPointer[List[Self.L]]

    fn __init__(out self):
        comptime l_size = Self.out_layout.size()
        comptime size = Self.out_layout.size() if len(Self.dims) > 1 else 0
        self.inter_layer_buf = {Self.L(capacity=size)}
        self.twiddle_factors = {
            _get_dims_twfs[
                Self.out_dtype, Self.out_layout, Self.inverse, Self.bases
            ]()
        }


fn _get_dims_twfs[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
](out twfs: List[List[ComplexScalar[out_dtype]]]):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime amnt_dims = len(dims)

    twfs = {capacity = amnt_dims}

    @parameter
    for dim_idx in range(amnt_dims):
        comptime length = UInt(out_layout.shape[dim_idx + 1].value())
        twfs.append(_get_twiddle_factors[length, out_dtype, inverse]())


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
    plan: _CPUPlan[out_dtype, out_layout, inverse, bases],
    cpu_workers: Optional[UInt] = None,
):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime batches = UInt(out_layout.shape[0].value())
    comptime x_complex_in = in_layout.shape[rank - 1].value()
    # NOTE: extract the unsafe pointer to avoid the arcpointer refcount
    var twfs_runtime_ptr = plan.twiddle_factors[].unsafe_ptr()

    @parameter
    fn _find_max_threads(out max_threads: UInt):
        max_threads = 0

        @parameter
        for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(dims[i].value())
            comptime threads = dim // val
            max_threads = max(threads, max_threads)

    comptime max_threads = _find_max_threads()

    var threads = cpu_workers.or_else(UInt(parallelism_level()))
    var per_batch_workers = min(threads, max_threads)
    threads = max(threads - per_batch_workers, 1)
    var parallel_batches = min(threads, batches)

    @always_inline
    @parameter
    fn _run_1d_fft[
        dtype_in: DType,
        layout_out: Layout,
        layout_in: Layout,
        shared_origin: MutOrigin,
        x_in_origin: ImmutOrigin,
        //,
        dim_idx: Int,
    ](
        shared_f: LayoutTensor[out_dtype, layout_out, shared_origin, ...],
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
        # FIXME(#5686): maybe replace with this once it's solved
        # comptime twfs_array = _get_flat_twfs[
        #     out_dtype,
        #     length,
        #     total_twfs,
        #     ordered_bases,
        #     processed_list,
        #     inverse,
        # ]()
        # ref twfs_array_runtime = global_constant[twfs_array]()
        comptime twfs_layout = Layout.row_major(Int(length), 2)
        var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
            twfs_runtime_ptr[dim_idx].unsafe_ptr().bitcast[Scalar[out_dtype]]()
        )

        @parameter
        for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime func = _radix_n_fft_kernel[
                do_rfft = x_complex_in == 1 and dim_idx == start_dim_idx,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                ordered_bases=ordered_bases,
                runtime_twfs=False,
                inline_twfs = Int(length) * size_of[out_dtype]() * 2
                <= 64 * 1024,
            ]
            comptime num_iters = length // base

            @always_inline
            fn _run[
                width: Int
            ](local_i: Int) unified {mut shared_f, read x_in, read twfs}:
                var x_out = LayoutTensor[
                    out_dtype, Layout.row_major(Int(base), 2), MutExternalOrigin
                ].stack_allocation()
                func(shared_f, x_in, UInt(local_i), twfs, x_out)

            # TODO: replace with unroll once we have it again
            comptime factor = simd_width_of[out_dtype]()
            vectorize[1, unroll_factor=factor](Int(num_iters), _run)

    # When running ffts on multiple dimensions, we need to copy the output of
    # each dimension into an intermediate buffer for reordering
    # NOTE: extract the unsafe pointer to avoid the arcpointer refcount
    var inter_layer_buf_ptr = (
        plan.inter_layer_buf[]
        .unsafe_ptr()
        .mut_cast[True]()
        .bitcast[Scalar[out_dtype]]()
    )
    comptime o_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, o_layout, address_space=_]

    @always_inline
    @parameter
    fn _run_batch(block_num: Int):
        var block_offset = output.stride[0]() * block_num
        var base_out = out_t(output.ptr + block_offset)
        comptime x_out_layout = Layout.row_major(x.layout.shape[1:])
        var base_x = LayoutTensor[in_dtype, x_out_layout, address_space=_](
            x.ptr + x.stride[0]() * block_num
        )

        @parameter
        if len(dims) == 1:
            _run_1d_fft[start_dim_idx](base_out, base_x)
        else:
            var base_inter_out = out_t(inter_layer_buf_ptr + block_offset)

            @parameter
            for idx in reversed(range(len(dims))):
                comptime dim_tuple = dims[idx]
                comptime dim = dim_tuple.value()
                comptime batch_prod = UInt(prod // dim)

                @parameter
                if idx != start_dim_idx:
                    comptime G = AddressSpace.GENERIC
                    var b_o = base_out.ptr.address_space_cast[G]()
                    comptime count = prod * 2
                    parallel_memcpy(
                        dest=base_inter_out.ptr,
                        src=b_o,
                        count=count,
                        count_per_task=count // Int(per_batch_workers),
                        num_tasks=Int(per_batch_workers),
                    )

                @always_inline
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
                        var dim_batch_x = base_x.slice[
                            dim_sl, Slice(0, x_complex_in), exclude
                        ](idxes)
                        _run_1d_fft[idx](dim_batch_out, dim_batch_x)
                    else:
                        var dim_batch_inter_out = base_inter_out.slice[
                            dim_sl, o_comp, exclude
                        ](idxes).get_immutable()
                        _run_1d_fft[idx](dim_batch_out, dim_batch_inter_out)

                parallelize[func=_run_dim_batch](
                    Int(batch_prod), Int(per_batch_workers)
                )

    parallelize[func=_run_batch](Int(batches), Int(parallel_batches))
