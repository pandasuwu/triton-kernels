"""Matmul + GELU: fused Triton vs cuBLAS + separate GELU vs cuBLAS alone vs torch.compile, in TFLOPS.

TFLOPS counts only the 2*M*N*K matmul FLOPs (the activation is negligible),
so "cuBLAS alone" is an upper reference: it does less work than the others.
"""
import torch
import triton

from common import RESULTS, bench, compiled
from kernels.matmul import matmul, torch_reference

ACT = "gelu"
ref_compiled = compiled(lambda a, b: torch_reference(a, b, ACT))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch_unfused", "cublas", "torch_compile"],
        line_names=["Triton matmul+GELU (fused)", "torch.matmul + GELU (unfused)", "cuBLAS matmul only", "torch.compile(matmul+GELU)"],
        styles=[("tab:blue", "-"), ("tab:red", "-"), ("tab:green", ":"), ("tab:orange", "--")],
        ylabel="TFLOPS",
        plot_name="matmul_gelu_fp16",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    fn = {
        "triton": lambda: matmul(a, b, ACT),
        "torch_unfused": lambda: torch_reference(a, b, ACT),
        "cublas": lambda: torch.matmul(a, b),
        "torch_compile": lambda: ref_compiled(a, b),
    }[provider]
    ms, lo, hi = bench(fn)
    tflops = lambda t: 2 * M * N * K * 1e-12 / (t * 1e-3)
    return tflops(ms), tflops(hi), tflops(lo)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=RESULTS)
