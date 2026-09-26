"""Softmax: fused Triton vs torch.softmax vs naive unfused vs torch.compile(naive), in GB/s.

GB/s here is "effective bandwidth": the minimum bytes a softmax must move
(read the matrix once, write it once) divided by runtime. The naive version
actually moves more bytes than this, which is exactly why it scores lower.
"""
import torch
import triton

from common import RESULTS, bench, compiled
from kernels.softmax import naive_softmax, softmax

N_ROWS = 4096
naive_compiled = compiled(naive_softmax)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(1, 33)] + [6144, 8192, 12288, 16384],
        line_arg="provider",
        line_vals=["triton", "torch", "naive", "torch_compile"],
        line_names=["Triton fused", "torch.softmax", "Naive PyTorch (unfused)", "torch.compile(naive)"],
        styles=[("tab:blue", "-"), ("tab:green", "-"), ("tab:red", "-"), ("tab:orange", "--")],
        ylabel="GB/s",
        plot_name="softmax",
        args={"M": N_ROWS},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    fn = {
        "triton": lambda: softmax(x),
        "torch": lambda: torch.softmax(x, dim=-1),
        "naive": lambda: naive_softmax(x),
        "torch_compile": lambda: naive_compiled(x),
    }[provider]
    ms, lo, hi = bench(fn)
    gbps = lambda t: 2 * x.numel() * x.element_size() * 1e-9 / (t * 1e-3)
    return gbps(ms), gbps(hi), gbps(lo)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=RESULTS)
