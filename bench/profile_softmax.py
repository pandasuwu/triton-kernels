"""Profile fused Triton softmax vs naive unfused PyTorch.

Nsight Compute is not installed, and torch.profiler cannot record GPU
activity on this machine (CUPTI fails to initialise with
CUPTI_ERROR_INVALID_DEVICE under torch 2.11 + driver 610.91 on Windows).
So this script measures the same things another way:

  * Kernels per call: each implementation is captured into a CUDA graph and
    the kernel nodes in the graph are counted (CUDAGraph.debug_dump). This is
    the exact list of kernels the GPU runs for one call.
  * GPU time: triton.testing.do_bench for the whole call, and for each step of
    the naive version on its own.
  * Bytes moved: computed from the shapes of the tensors each kernel reads and
    writes. This is the algorithmic minimum per kernel, not a hardware
    counter reading.

If torch.profiler does work (e.g. on Linux), its kernel list is printed too.
Writes results/profile_softmax.txt and results/profile_softmax.csv.
"""
import csv
import os
import re
import tempfile

import torch
import triton

from common import RESULTS
from kernels.softmax import naive_softmax, softmax

M, N = 4096, 4096


def graph_kernels(fn, x):
    """Kernel names launched by one call of fn(x), from a captured CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # warm up: compile Triton kernels, fill allocator
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # keep_graph=True keeps the captured cudaGraph_t alive so it can be dumped.
    g = torch.cuda.CUDAGraph(keep_graph=True)
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        fn(x)
    path = os.path.join(tempfile.mkdtemp(), "graph.dot").replace("\\", "/")
    g.debug_dump(path)
    with open(path) as f:
        dot = f.read()
    # Each kernel node in the DOT dump looks like
    #   label="{KERNEL\n| {ID | 0 (topoId: 2) | <mangled kernel name>\<\<\<grid,block,smem\>\>\>}
    n_nodes = len(re.findall(r'label="\{KERNEL', dot))
    names = re.findall(r"\(topoId: \d+\) \| ([^\\}]+)", dot)
    assert len(names) == n_nodes, (len(names), n_nodes)
    return names


def short(name):
    """Shorten mangled C++ kernel names for display."""
    for pat in (r"softmax_\w*kernel", r"[a-z]+_kernel_cuda", r"Max\w*Ops", r"func_wrapper", r"(Div|Sub|Mul|Add)\w*Functor",
                r"reduce_kernel", r"elementwise_kernel"):
        m = re.search(pat, name)
        if m:
            tag = m.group(0)
            if tag == "reduce_kernel" and "Max" in name:
                tag = "reduce_kernel(max)"
            return tag
    return name[:60]


def ms(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=300)


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    mat, vec = M * N * 4, M * 4
    x_max = x.max(dim=1, keepdim=True).values
    z = x - x_max
    num = torch.exp(z)
    den = num.sum(dim=1, keepdim=True)

    naive_steps = [
        ("max(x)",     lambda: x.max(dim=1, keepdim=True), mat + vec),
        ("x - max",    lambda: x - x_max,                  mat + vec + mat),
        ("exp(z)",     lambda: torch.exp(z),               mat + mat),
        ("sum(num)",   lambda: num.sum(dim=1, keepdim=True), mat + vec),
        ("num / den",  lambda: num / den,                  mat + vec + mat),
    ]
    fused_bytes = 2 * mat

    k_fused = graph_kernels(softmax, x)
    k_naive = graph_kernels(naive_softmax, x)
    t_fused = ms(lambda: softmax(x))
    t_naive = ms(lambda: naive_softmax(x))
    t_steps = [(n, ms(f), b) for n, f, b in naive_steps]
    naive_bytes = sum(b for *_, b in t_steps)

    L = [f"Softmax profile: {M} x {N} fp32, {torch.cuda.get_device_name(0)}", ""]
    L.append("Kernels per call (counted from a captured CUDA graph):")
    L.append(f"  Triton fused : {len(k_fused)}  {', '.join(short(k) for k in k_fused)}")
    L.append(f"  Naive PyTorch: {len(k_naive)}  {', '.join(short(k) for k in k_naive)}")
    L.append("")
    L.append("GPU time per call (triton.testing.do_bench, median):")
    L.append(f"  Triton fused : {t_fused * 1e3:8.1f} us")
    L.append(f"  Naive PyTorch: {t_naive * 1e3:8.1f} us   ({t_naive / t_fused:.2f}x slower)")
    for n, t, b in t_steps:
        L.append(f"     {n:<10} {t * 1e3:8.1f} us   {b / 1e6:6.1f} MB")
    L.append("")
    L.append("Bytes moved per call (computed from tensor shapes, minimum per kernel):")
    L.append(f"  Triton fused : {fused_bytes / 1e6:6.1f} MB")
    L.append(f"  Naive PyTorch: {naive_bytes / 1e6:6.1f} MB   ({naive_bytes / fused_bytes:.2f}x)")
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "profile_softmax.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(RESULTS, "profile_softmax.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementation", "step", "kernels", "gpu_us", "bytes_MB"])
        w.writerow(["triton_fused", "total", len(k_fused), round(t_fused * 1e3, 1), round(fused_bytes / 1e6, 1)])
        w.writerow(["naive_pytorch", "total", len(k_naive), round(t_naive * 1e3, 1), round(naive_bytes / 1e6, 1)])
        for n, t, b in t_steps:
            w.writerow(["naive_pytorch", n, 1, round(t * 1e3, 1), round(b / 1e6, 1)])
    if len(k_fused) == 0 or len(k_naive) == 0:
        raise RuntimeError("No kernels found in captured graph; profile is invalid")


if __name__ == "__main__":
    main()
