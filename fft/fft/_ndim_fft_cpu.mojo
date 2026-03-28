from std.algorithm import parallelize, vectorize, parallel_memcpy
from std.complex import ComplexScalar
from layout import Layout, LayoutTensor, IntTuple
from std.runtime.asyncrt import parallelism_level
from std.sys.info import size_of, simd_width_of
from std.memory import memcpy, ArcPointer
from std.math import ceildiv

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
    _num_stages_end_of,
)
from ._fft import (
    _radix_n_fft_kernel_stockham,
    _radix_n_fft_kernel_stockham_comptime,
)


struct _CPUPlan[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
](Copyable):
    comptime L = List[ComplexScalar[Self.out_dtype]]

    var twiddle_factors: ArcPointer[List[Self.L]]
    var calc_buf: ArcPointer[Self.L]

    def __init__(out self):
        self.twiddle_factors = {
            _get_dims_twfs[
                Self.out_dtype, Self.out_layout, Self.inverse, Self.bases
            ]()
        }
        comptime size = Self.out_layout.size() // 2  # Self.L is already complex
        self.calc_buf = {Self.L(capacity=size)}


def _get_dims_twfs[
    out_dtype: DType,
    out_layout: Layout,
    inverse: Bool,
    bases: List[List[UInt]],
](out twfs: List[List[ComplexScalar[out_dtype]]]):
    comptime rank = out_layout.rank()
    comptime dims = out_layout.shape[1 : rank - 1]
    comptime amnt_dims = len(dims)

    twfs = {capacity = amnt_dims}

    comptime for dim_idx in range(amnt_dims):
        comptime length = UInt(out_layout.shape[dim_idx + 1].value())
        twfs.append(_get_twiddle_factors[length, out_dtype, inverse]())


def _run_cpu_nd_fft[
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
    comptime amnt_dims = len(dims)
    comptime prod = _product_of_dims(dims)
    comptime start_dim_idx = len(dims) - 1
    """We are running the ffts from right to left in the layout."""

    comptime batches = UInt(out_layout.shape[0].value())
    comptime x_complex_in = in_layout.shape[rank - 1].value()
    # NOTE: extract the unsafe pointer to avoid the arcpointer refcount
    var twfs_runtime_ptr = plan.twiddle_factors[].unsafe_ptr()

    @parameter
    def _find_max_threads(out max_threads: UInt):
        max_threads = 0

        comptime for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(dims[i].value())
            comptime threads = dim // val
            max_threads = max(threads, max_threads)

    comptime max_threads = _find_max_threads()

    var threads = cpu_workers.or_else(UInt(parallelism_level()))
    var per_batch_workers = min(threads, max_threads) if amnt_dims > 1 else 1
    var parallel_batches = min(
        max(threads - (per_batch_workers - 1), 1), batches
    )

    @always_inline
    @parameter
    def _run_1d_fft[
        dtype_in: DType, //, dim_idx: Int
    ](
        shared_f_lhs: LayoutTensor[mut=True, out_dtype, ...],
        shared_f_rhs: LayoutTensor[mut=True, out_dtype, ...],
        x_in: LayoutTensor[mut=False, dtype_in, ...],
    ):
        comptime length = UInt(x_in.layout.shape[0].value())
        comptime bases_idx = bases[dim_idx]
        comptime bases_processed = materialize[
            _get_ordered_bases_processed_list[length, bases_idx]()
        ]()
        comptime ordered_bases = bases_processed[0]
        comptime processed_list = bases_processed[1]
        comptime twfs_layout = Layout.row_major(Int(length), 2)
        var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
            twfs_runtime_ptr[dim_idx].unsafe_ptr().bitcast[Scalar[out_dtype]]()
        )

        comptime num_stages = _num_stages_end_of[bases, dims, dim_idx + 1]()

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            # TODO: once we can stop this from fully unrolling we should use it
            # for every length
            comptime if length <= 128:
                comptime func = _radix_n_fft_kernel_stockham_comptime[
                    ...,
                    do_rfft=x_complex_in == 1 and dim_idx == start_dim_idx,
                    base=base,
                    length=length,
                    processed=processed,
                    inverse=inverse,
                    ordered_bases=ordered_bases,
                ]

                comptime for local_i in range(length):
                    comptime if _num_stages_end_of[bases, dims, 0]() % 2 == 0:
                        comptime if b == 0 and dim_idx == start_dim_idx:
                            func[local_i=local_i](shared_f_rhs, x_in)
                        elif (num_stages + b) % 2 == 0:
                            func[local_i=local_i](shared_f_rhs, shared_f_lhs)
                        else:
                            func[local_i=local_i](shared_f_lhs, shared_f_rhs)
                    else:
                        comptime if b == 0 and dim_idx == start_dim_idx:
                            func[local_i=local_i](shared_f_lhs, x_in)
                        elif (num_stages + b) % 2 == 0:
                            func[local_i=local_i](shared_f_lhs, shared_f_rhs)
                        else:
                            func[local_i=local_i](shared_f_rhs, shared_f_lhs)
            else:
                comptime func = _radix_n_fft_kernel_stockham[
                    ...,
                    do_rfft=x_complex_in == 1 and dim_idx == start_dim_idx,
                    base=base,
                    length=length,
                    processed=processed,
                    inverse=inverse,
                    ordered_bases=ordered_bases,
                    inline_twfs=(
                        Int(length) * size_of[out_dtype]() * 2 <= 64 * 1024
                    ),
                    runtime_twfs=False,
                ]

                @always_inline
                def _run[width: Int](local_i: Int) unified {read}:
                    var idx = UInt(local_i)
                    comptime if _num_stages_end_of[bases, dims, 0]() % 2 == 0:
                        comptime if b == 0 and dim_idx == start_dim_idx:
                            func(shared_f_rhs, x_in, idx, twfs)
                        elif (num_stages + b) % 2 == 0:
                            func(shared_f_rhs, shared_f_lhs, idx, twfs)
                        else:
                            func(shared_f_lhs, shared_f_rhs, idx, twfs)
                    else:
                        comptime if b == 0 and dim_idx == start_dim_idx:
                            func(shared_f_lhs, x_in, idx, twfs)
                        elif (num_stages + b) % 2 == 0:
                            func(shared_f_lhs, shared_f_rhs, idx, twfs)
                        else:
                            func(shared_f_rhs, shared_f_lhs, idx, twfs)

                # TODO: replace with unroll once we have it again
                comptime width = simd_width_of[out_dtype]()
                vectorize[1, unroll_factor=width](Int(length), _run)

    # NOTE: extract the unsafe pointer to avoid the arcpointer refcount
    var calc_buf_ptr = (
        plan.calc_buf[]
        .unsafe_ptr()
        .unsafe_mut_cast[True]()
        .bitcast[Scalar[out_dtype]]()
    )
    comptime o_layout = Layout.row_major(output.layout.shape[1:])
    comptime out_t = LayoutTensor[out_dtype, o_layout, ...]

    @always_inline
    @parameter
    def _run_batch(block_num: Int):
        var block_offset = output.stride[0]() * block_num
        var base_out = out_t(output.ptr + block_offset)
        var base_calc = out_t(calc_buf_ptr + block_offset)
        comptime x_out_layout = Layout.row_major(x.layout.shape[1:])
        var base_x = LayoutTensor[in_dtype, x_out_layout, address_space=_](
            x.ptr + x.stride[0]() * block_num
        )

        comptime if amnt_dims == 1:
            _run_1d_fft[start_dim_idx](base_out, base_calc, base_x)
        else:
            comptime for idx in reversed(range(amnt_dims)):
                comptime dim_tuple = dims[idx]
                comptime dim = dim_tuple.value()
                comptime batch_prod = UInt(prod // dim)

                @always_inline
                @parameter
                def _run_dim_batch(flat_idx: Int):
                    comptime exclude = (idx, rank - 2)
                    comptime exclude_t = IntTuple(idx, rank - 2)
                    comptime dim_sl = Slice(0, dim)
                    comptime o_comp = Slice(0, 2)
                    comptime dims_comp = base_out.layout.shape
                    var idxes = _get_cascade_idxes[dims_comp, exclude_t](
                        flat_idx
                    )
                    var dim_batch_x = base_x.slice[
                        dim_sl, Slice(0, x_complex_in), exclude
                    ](idxes)
                    var dim_batch_out = base_out.slice[dim_sl, o_comp, exclude](
                        idxes
                    )
                    var dim_base_calc = base_calc.slice[
                        dim_sl, o_comp, exclude
                    ](idxes)
                    _run_1d_fft[idx](dim_batch_out, dim_base_calc, dim_batch_x)

                parallelize[func=_run_dim_batch](
                    Int(batch_prod), Int(per_batch_workers)
                )

    parallelize[func=_run_batch](Int(batches), Int(parallel_batches))
