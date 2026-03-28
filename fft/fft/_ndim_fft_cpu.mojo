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


@always_inline
def transpose[
    *, into_: Int, from_: Int
](
    dst: LayoutTensor[mut=True, ...],
    src: LayoutTensor[mut=False, dst.dtype, dst.layout, ...],
    num_workers: Int,
):
    comptime dims = src.layout.shape[: src.rank - 1]

    @parameter
    def _calc_sizes() -> Tuple[Int, Int, Int]:
        # Always anchor the math to the outer-most index being swapped
        var target_idx = min(into_, from_)

        # 1. Batches: Product of all dimensions outside the swap zone
        var batch_val = 1
        for i in range(0, target_idx + 1):
            batch_val *= dims[i].value()

        # 2. M: The size of the specific dimension we are shifting
        var m_val = dims[target_idx + 1].value()

        # 3. N: Product of all inner dimensions inside the swap zone
        var n_val = 1
        for i in range(target_idx + 2, len(dims)):
            n_val *= dims[i].value()

        # If we are restoring the layout (from_ < into_), the source memory
        # is already flipped [N, M], so we swap what M and N mean for the loop.
        comptime if into_ < from_:
            return batch_val, m_val, n_val
        else:
            return batch_val, n_val, m_val

    comptime sizes = _calc_sizes()
    comptime batches = sizes[0]
    comptime M = sizes[1]
    comptime N = sizes[2]
    comptime TILE = simd_width_of[ComplexScalar[src.dtype]]()

    @parameter
    def _transpose_batch(b: Int):
        var src_base = src.ptr + b * M * N * 2
        var dst_base = dst.ptr + b * M * N * 2

        for i in range(0, M, TILE):
            for j in range(0, N, TILE):
                var i_max = min(i + TILE, M)
                var j_max = min(j + TILE, N)

                for ii in range(i, i_max):
                    for jj in range(j, j_max):
                        var src_offset = (ii * N + jj) * 2
                        var dst_offset = (jj * M + ii) * 2
                        var val = src_base.load[2](src_offset)
                        dst_base.store(dst_offset, val)

    parallelize[_transpose_batch](batches, min(num_workers, batches, TILE))


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
    # TODO: this should be dependent on the CPU cache architecture
    comptime MAX_STACK_SEQ_LEN = 128
    """Maximum length to fully unroll on the stack."""
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
    def _find_max_batch_prod(out max_batch_prod: UInt):
        max_batch_prod = 0

        comptime for i, base_set in enumerate(bases):
            comptime val = _min(base_set)
            comptime dim = UInt(dims[i].value())
            max_batch_prod = max(dim // val, max_batch_prod)

    comptime max_batch_prod = _find_max_batch_prod()

    var threads = cpu_workers.or_else(UInt(parallelism_level()))
    var per_batch_workers = min(threads, max_batch_prod) if amnt_dims > 1 else 1
    var parallel_batches = min(
        max(threads - (per_batch_workers - 1), 1), batches
    )
    comptime total_stages = _num_stages_end_of[
        bases, dims, 0
    ]() + 2 * start_dim_idx

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

        comptime fft_stages = _num_stages_end_of[bases, dims, dim_idx + 1]()
        comptime prev_stages = fft_stages + (start_dim_idx - dim_idx)

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime do_rfft = x_complex_in == 1 and (
                dim_idx == start_dim_idx
            ) and b == 0
            comptime s = prev_stages + b
            comptime write_lhs = (total_stages - (s + 1)) % 2 == 0

            comptime if length <= MAX_STACK_SEQ_LEN:
                comptime func = _radix_n_fft_kernel_stockham_comptime[
                    ...,
                    do_rfft=do_rfft,
                    base=base,
                    length=length,
                    processed=processed,
                    inverse=inverse,
                    ordered_bases=ordered_bases,
                ]

                comptime for local_i in range(length):
                    comptime if b == 0 and dim_idx == start_dim_idx:
                        comptime if write_lhs:
                            func[local_i=local_i](shared_f_lhs, x_in)
                        else:
                            func[local_i=local_i](shared_f_rhs, x_in)
                    else:
                        comptime if write_lhs:
                            func[local_i=local_i](shared_f_lhs, shared_f_rhs)
                        else:
                            func[local_i=local_i](shared_f_rhs, shared_f_lhs)
            else:
                comptime func = _radix_n_fft_kernel_stockham[
                    ...,
                    do_rfft=do_rfft,
                    base=base,
                    length=length,
                    processed=processed,
                    inverse=inverse,
                    ordered_bases=ordered_bases,
                    inline_twfs=False,
                    runtime_twfs=False,
                ]

                @always_inline
                def _run[width: Int](local_i: Int) unified {read}:
                    var idx = UInt(local_i)
                    comptime if b == 0 and dim_idx == start_dim_idx:
                        comptime if write_lhs:
                            func(shared_f_lhs, x_in, idx, twfs)
                        else:
                            func(shared_f_rhs, x_in, idx, twfs)
                    else:
                        comptime if write_lhs:
                            func(shared_f_lhs, shared_f_rhs, idx, twfs)
                        else:
                            func(shared_f_rhs, shared_f_lhs, idx, twfs)

                # TODO: replace with unroll once we have it again
                comptime width = max(simd_width_of[out_dtype](), Int(base))
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

                comptime if idx != start_dim_idx:
                    comptime fft_stages = _num_stages_end_of[
                        bases, dims, idx + 1
                    ]()
                    comptime s = fft_stages + (start_dim_idx - (idx + 1))
                    comptime write_lhs = (total_stages - (s + 1)) % 2 == 0
                    comptime if write_lhs:
                        transpose[from_=idx + 1, into_=idx](
                            base_out, base_calc, Int(per_batch_workers)
                        )
                    else:
                        transpose[from_=idx + 1, into_=idx](
                            base_calc, base_out, Int(per_batch_workers)
                        )

                comptime dim_x_layout = Layout.row_major(dim, x_complex_in)
                comptime x_offset = dim * x_complex_in
                comptime dim_out_layout = Layout.row_major(dim, 2)
                comptime out_offset = dim * 2

                @always_inline
                @parameter
                def _run_dim_batch(flat_idx: Int):
                    var dim_batch_x = LayoutTensor[_, dim_x_layout, ...](
                        base_x.ptr + flat_idx * x_offset
                    )
                    var dim_batch_out = LayoutTensor[_, dim_out_layout, ...](
                        base_out.ptr + flat_idx * out_offset
                    )
                    var dim_batch_calc = LayoutTensor[_, dim_out_layout, ...](
                        base_calc.ptr + flat_idx * out_offset
                    )
                    _run_1d_fft[idx](dim_batch_out, dim_batch_calc, dim_batch_x)

                parallelize[func=_run_dim_batch](
                    Int(batch_prod), Int(per_batch_workers)
                )

            comptime fft_stages = _num_stages_end_of[bases, dims, 0]()
            comptime for idx in range(amnt_dims - 1):
                comptime s = fft_stages + start_dim_idx + idx
                comptime write_lhs = (total_stages - (s + 1)) % 2 == 0
                comptime if write_lhs:
                    transpose[from_=idx, into_=idx + 1](
                        base_out, base_calc, Int(per_batch_workers)
                    )
                else:
                    transpose[from_=idx, into_=idx + 1](
                        base_calc, base_out, Int(per_batch_workers)
                    )

    parallelize[func=_run_batch](Int(batches), Int(parallel_batches))
