from std.compile import compile_info
from layout import Layout, LayoutTensor

from fft.fft._ndim_fft_cpu import _run_cpu_nd_fft, _CPUPlan


def run_fft():
    comptime SIZE = 2**14
    comptime in_layout = Layout.row_major(1, SIZE, 2)
    comptime in_size = in_layout.size()
    comptime out_layout = Layout.row_major(1, SIZE, 2)
    comptime out_size = out_layout.size()
    comptime in_dtype = DType.float32
    comptime out_dtype = DType.float32

    var out_data = List[Scalar[in_dtype]](length=out_size, fill=0)
    var x_data = List[Scalar[out_dtype]](length=in_size, fill=0)
    var output = LayoutTensor[mut=True, out_dtype, out_layout](Span(out_data))
    var x = LayoutTensor[mut=False, in_dtype, in_layout](Span(x_data))
    var plan = _CPUPlan[out_dtype, out_layout, False, [[2]]]()
    _run_cpu_nd_fft(output, x, plan=plan)


def main() raises:
    with open("dump_cpu.ll", "w") as f:
        f.write(compile_info[run_fft, emission_kind="llvm-opt"]())
