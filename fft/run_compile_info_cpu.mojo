from std.compile import compile_info
from layout import TileTensor, row_major

from fft.fft._ndim_fft_cpu import _run_cpu_nd_fft, _CPUPlan


def run_fft():
    comptime SIZE = 2**14
    comptime in_layout = row_major[1, SIZE, 2]()
    comptime out_layout = row_major[1, SIZE, 2]()
    comptime in_size = in_layout.static_cosize
    comptime out_size = out_layout.static_cosize
    comptime in_dtype = DType.float32
    comptime out_dtype = DType.float32

    var out_data = List[Scalar[in_dtype]](length=out_size, fill=0)
    var x_data = List[Scalar[out_dtype]](length=in_size, fill=0)
    var output = TileTensor(Span(out_data), out_layout)
    var x = TileTensor(Span(x_data), in_layout)
    var plan = _CPUPlan[out_dtype, type_of(out_layout), False, [[2]]]()
    _run_cpu_nd_fft(output, x, plan=plan)


def main() raises:
    with open("run_compile_info_cpu.ll", "w") as f:
        f.write(compile_info[run_fft, emission_kind="llvm-opt"]())
