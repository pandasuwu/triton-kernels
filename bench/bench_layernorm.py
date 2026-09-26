"""LayerNorm forward: fused Triton vs F.layer_norm vs torch.compile(manual layernorm), in GB/s.

Effective bandwidth counts reading x and writing y once (weights/bias are
small and ignored).
"""
import torch
import torch.nn.functional as F
import triton

from common import RESULTS, bench, compiled
from kernels.layernorm import layernorm

N_ROWS = 4096


def manual_layernorm(x, w, b, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * w + b


manual_compiled = compiled(manual_layernorm)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256 * i for i in range(1, 17)] + [6144, 8192, 12288, 16384],
        line_arg="provider",
        line_vals=["triton", "torch", "naive", "torch_compile"],
        line_names=["Triton fused", "F.layer_norm", "Naive PyTorch (unfused)", "torch.compile(naive)"],
        styles=[("tab:blue", "-"), ("tab:green", "-"), ("tab:red", "-"), ("tab:orange", "--")],
        ylabel="GB/s",
        plot_name="layernorm_fp16",
        args={"M": N_ROWS},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    fn = {
        "triton": lambda: layernorm(x, w, b),
        "torch": lambda: F.layer_norm(x, (N,), w, b),
        "naive": lambda: manual_layernorm(x, w, b),
        "torch_compile": lambda: manual_compiled(x, w, b),
    }[provider]
    ms, lo, hi = bench(fn)
    gbps = lambda t: 2 * x.numel() * x.element_size() * 1e-9 / (t * 1e-3)
    return gbps(ms), gbps(hi), gbps(lo)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=RESULTS)
