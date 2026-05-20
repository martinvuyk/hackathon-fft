from std.algorithm import parallelize, vectorize
from std.complex import ComplexScalar
from layout import Layout, LayoutTensor, IntTuple
from std.runtime.asyncrt import parallelism_level
from std.sys.info import size_of, simd_width_of
from std.memory import memcpy, ArcPointer
from std.math import ceildiv, sqrt
from std.utils.index import IndexList
from std.bit import prev_power_of_two

from ._utils import (
    _get_twiddle_factors,
    _get_ordered_bases_processed_list,
    _product_of_dims,
    _max,
    _min,
    _num_stages_end_of,
    _calc_batches_M_N,
)
from ._fft import (
    _radix_n_fft_kernel_elem_per_thread,
    _radix_n_fft_kernel_elem_per_thread_comptime,
    _radix_n_fft_kernel_butterfly,
    _radix_n_fft_kernel_butterfly_comptime,
)


struct _CPUPlan[
    out_dtype: DType, out_layout: Layout, inverse: Bool, bases: List[List[UInt]]
](Copyable):
    comptime rank = Self.out_layout.rank()
    comptime dims = Self.out_layout.shape[1 : Self.rank - 1]
    # TODO: this should somehow be dependent on the CPU register size
    comptime max_stack_seq_len = 128
    """Maximum sequence length to fully unroll on the stack."""
    # TODO: this should somehow be dependent on the CPU register size
    comptime max_butterfly_base = 16
    """Maximum radix base to use the butterfly algorithm."""

    comptime L = List[ComplexScalar[Self.out_dtype]]

    var twiddle_factors: ArcPointer[List[Optional[Self.L]]]
    var calc_buf: ArcPointer[Self.L]

    def __init__(out self):
        self.twiddle_factors = {Self._get_dims_twfs()}
        comptime size = Self.out_layout.size() // 2  # Self.L is already complex
        self.calc_buf = {Self.L(capacity=size)}

    @staticmethod
    def _get_dims_twfs(out twfs: List[Optional[Self.L]]):
        comptime amnt_dims = len(Self.dims)

        twfs = {capacity = amnt_dims}

        comptime for dim_idx in range(amnt_dims):
            comptime length = UInt(Self.dims[dim_idx].value())
            comptime if length <= Self.max_stack_seq_len:
                twfs.append(None)
                continue
            twfs.append(
                _get_twiddle_factors[length, Self.out_dtype, Self.inverse]()
            )


def _complex_transpose_mask[TILE: Int]() -> IndexList[TILE * TILE * 2]:
    var mask = IndexList[TILE * TILE * 2](fill=0)
    for i in range(TILE):
        for j in range(TILE):
            var src_re = i * (2 * TILE) + j * 2
            var src_im = src_re + 1

            var dst_re = j * (2 * TILE) + i * 2
            var dst_im = dst_re + 1

            mask[dst_re] = src_re
            mask[dst_im] = src_im
    return mask


@always_inline
def _transpose[
    *, into_: Int, from_: Int
](
    dst: LayoutTensor[mut=True, ...],
    src: LayoutTensor[mut=False, dst.dtype, dst.layout, ...],
    num_workers: Int,
):
    comptime dims = src.layout.shape[: src.rank - 1]

    comptime sizes = _calc_batches_M_N[dims, into_, from_]()
    comptime intra_fft_batches = Int(sizes[0])
    comptime M = Int(sizes[1])
    comptime N = Int(sizes[2])
    # We assume the L1 -> registers bus size is ~ 2 * simd_width
    comptime TILE = min(
        prev_power_of_two(min(M, N)), simd_width_of[src.dtype]()
    )
    comptime mask = _complex_transpose_mask[TILE]()

    @parameter
    def _transpose_batch(b: Int):
        var src_base = src.ptr + b * M * N * 2
        var dst_base = dst.ptr + b * M * N * 2

        for i in range(0, M, TILE):
            for j in range(0, N, TILE):
                if i + TILE <= M and j + TILE <= N:
                    var tile_in = SIMD[src.dtype, TILE * TILE * 2]()

                    comptime for row in range(TILE):
                        var val = src_base.load[TILE * 2](
                            ((i + row) * N + j) * 2
                        )
                        UnsafePointer(to=tile_in).bitcast[
                            Scalar[src.dtype]
                        ]().store(row * TILE * 2, val)

                    var tile_out = tile_in.shuffle[mask]()

                    comptime for col in range(TILE):
                        var val = (
                            UnsafePointer(to=tile_out)
                            .bitcast[Scalar[src.dtype]]()
                            .load[TILE * 2](col * TILE * 2)
                        )
                        dst_base.store(((j + col) * M + i) * 2, val)
                else:
                    for ii in range(i, min(i + TILE, M)):
                        for jj in range(j, min(j + TILE, N)):
                            var val = src_base.load[2]((ii * N + jj) * 2)
                            dst_base.store((jj * M + ii) * 2, val)

    parallelize[_transpose_batch](
        intra_fft_batches, min(num_workers, intra_fft_batches, TILE)
    )


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
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
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

        var twfs: LayoutTensor[
            mut=False, out_dtype, twfs_layout, origin_of(plan)
        ]
        comptime if length <= plan.max_stack_seq_len:
            twfs = {unsafe_ptr = {}}
        else:
            twfs = {
                twfs_runtime_ptr[dim_idx]
                .value()
                .unsafe_ptr()
                .bitcast[Scalar[out_dtype]]()
                .mut_cast[False]()
                .unsafe_origin_cast[origin_of(plan)]()
            }

        comptime fft_stages = _num_stages_end_of[bases, dims, dim_idx + 1]()
        comptime prev_stages = fft_stages + (start_dim_idx - dim_idx)
        comptime run_butterfly = ordered_bases[0] < plan.max_butterfly_base

        comptime for b in range(len(ordered_bases)):
            comptime base = ordered_bases[b]
            comptime processed = processed_list[b]
            comptime do_rfft = x_complex_in == 1 and (
                dim_idx == start_dim_idx
            ) and b == 0
            comptime s = prev_stages + b
            comptime write_lhs = (total_stages - (s + 1)) % 2 == 0
            comptime x_out_layout = Layout.row_major(Int(base), 2)

            # comptime if run_butterfly and length <= plan.max_stack_seq_len:
            #     comptime func = _radix_n_fft_kernel_butterfly_comptime[
            #         ...,
            #         do_rfft=do_rfft,
            #         base=base,
            #         length=length,
            #         processed=processed,
            #         inverse=inverse,
            #         ordered_bases=ordered_bases,
            #         run_inplace=False,
            #     ]

            #     comptime for local_i in range(length // base):
            #         var x_out = LayoutTensor[
            #             out_dtype, x_out_layout, MutExternalOrigin
            #         ].stack_allocation()
            #         comptime if b == 0 and dim_idx == start_dim_idx:
            #             comptime if write_lhs:
            #                 func[local_i=local_i](shared_f_lhs, x_in, x_out)
            #             else:
            #                 func[local_i=local_i](shared_f_rhs, x_in, x_out)
            #         else:
            #             comptime if write_lhs:
            #                 func[local_i=local_i](
            #                     shared_f_lhs, shared_f_rhs, x_out
            #                 )
            #             else:
            #                 func[local_i=local_i](
            #                     shared_f_rhs, shared_f_lhs, x_out
            #                 )
            # elif run_butterfly:
            comptime if run_butterfly:
                comptime iters = length // base
                comptime num_blocks = iters // processed

                @always_inline
                @parameter
                def run_phase[phase: Optional[UInt]](runtime_phase: UInt):
                    comptime func = _radix_n_fft_kernel_butterfly[
                        ...,
                        do_rfft=do_rfft,
                        base=base,
                        length=length,
                        processed=processed,
                        inverse=inverse,
                        ordered_bases=ordered_bases,
                        inline_twfs=length <= plan.max_stack_seq_len,
                        runtime_twfs=False,
                        run_inplace=False,
                        phase=phase,
                    ]

                    @always_inline
                    def _run_butterfly[width: Int](local_i: Int) unified {read}:
                        var x_out = LayoutTensor[
                            out_dtype, x_out_layout, MutExternalOrigin
                        ].stack_allocation()
                        var idx = UInt(local_i) + runtime_phase * num_blocks
                        comptime if b == 0 and dim_idx == start_dim_idx:
                            comptime if write_lhs:
                                func(shared_f_lhs, x_in, idx, twfs, x_out)
                            else:
                                func(shared_f_rhs, x_in, idx, twfs, x_out)
                        else:
                            comptime if write_lhs:
                                func(
                                    shared_f_lhs, shared_f_rhs, idx, twfs, x_out
                                )
                            else:
                                func(
                                    shared_f_rhs, shared_f_lhs, idx, twfs, x_out
                                )

                    # TODO: replace with unroll once we have it again
                    comptime width = simd_width_of[out_dtype]() // 2
                    vectorize[1, unroll_factor=width](
                        Int(num_blocks), _run_butterfly
                    )

                comptime full_unroll = min(processed, plan.max_stack_seq_len)
                comptime for phase in range(full_unroll):
                    run_phase[phase](phase)
                for phase in range(full_unroll, processed):
                    run_phase[None](phase)
            elif length <= plan.max_stack_seq_len:
                comptime func = _radix_n_fft_kernel_elem_per_thread_comptime[
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

                @always_inline
                def _run_elem[width: Int](local_i: Int) unified {read}:
                    comptime func = _radix_n_fft_kernel_elem_per_thread[
                        ...,
                        do_rfft=do_rfft,
                        base=base,
                        length=length,
                        processed=processed,
                        inverse=inverse,
                        ordered_bases=ordered_bases,
                        inline_twfs=length <= plan.max_stack_seq_len,
                        runtime_twfs=False,
                    ]

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
                vectorize[1, unroll_factor=width](Int(length), _run_elem)

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
                        _transpose[from_=idx + 1, into_=idx](
                            base_out, base_calc, Int(per_batch_workers)
                        )
                    else:
                        _transpose[from_=idx + 1, into_=idx](
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
                    _transpose[from_=idx, into_=idx + 1](
                        base_out, base_calc, Int(per_batch_workers)
                    )
                else:
                    _transpose[from_=idx, into_=idx + 1](
                        base_calc, base_out, Int(per_batch_workers)
                    )

    parallelize[func=_run_batch](Int(batches), Int(parallel_batches))
