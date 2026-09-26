"""Vector add: Triton vs PyTorch eager vs torch.compile, in GB/s."""
import torch
import triton

from common import RESULTS, bench, compiled
from kernels.vector_add import add

torch_add_compiled = compiled(lambda x, y: x + y)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 29)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "PyTorch eager", "torch.compile"],
        styles=[("tab:blue", "-"), ("tab:green", "-"), ("tab:orange", "--")],
        ylabel="GB/s",
        plot_name="vector_add",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    if provider == "triton":
        ms, lo, hi = bench(lambda: add(x, y))
    elif provider == "torch":
        ms, lo, hi = bench(lambda: x + y)
    else:
        ms, lo, hi = bench(lambda: torch_add_compiled(x, y))
    # Bytes moved: read x, read y, write out.
    gbps = lambda t: 3 * x.numel() * x.element_size() * 1e-9 / (t * 1e-3)
    return gbps(ms), gbps(hi), gbps(lo)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=RESULTS)
