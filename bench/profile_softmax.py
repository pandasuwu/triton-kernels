"""Profile fused Triton softmax vs naive unfused PyTorch with torch.profiler.

Records, per softmax call on a 4096 x 4096 fp32 matrix:
  * the GPU kernels actually launched (measured by the profiler),
  * their total GPU time (measured),
  * the global-memory bytes each kernel must read/write, computed from the
    tensor shapes each kernel consumes and produces (not a hardware counter;
    Nsight Compute is not available on this machine).
Writes results/profile_softmax.txt and results/profile_softmax.csv.
"""
import csv
import os

import torch
from torch.profiler import ProfilerActivity, profile

from common import RESULTS
from kernels.softmax import naive_softmax, softmax

M, N = 4096, 4096
ITERS = 20


def profile_fn(fn, x):
    for _ in range(5):  # warm up / compile
        fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn(x)
        torch.cuda.synchronize()
    kernels = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0:
            t = getattr(evt, "device_time_total", None)
            if t is None:
                t = evt.cuda_time_total
            kernels[evt.key] = (evt.count / ITERS, t / ITERS)
    return kernels


def main():
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    mat = M * N * 4          # bytes in one full fp32 M x N tensor
    vec = M * 4              # bytes in one per-row fp32 vector
    # Bytes each step must move given its inputs/outputs (see naive_softmax):
    naive_bytes = {
        "max":  mat + vec,          # read x, write row max
        "sub":  mat + vec + mat,    # read x + row max, write z
        "exp":  mat + mat,          # read z, write num
        "sum":  mat + vec,          # read num, write row sum
        "div":  mat + vec + mat,    # read num + row sum, write out
    }
    fused_bytes = mat + mat        # read x once, write out once

    results = {"triton_fused": profile_fn(softmax, x), "naive_pytorch": profile_fn(naive_softmax, x)}

    lines = [f"Softmax profile, {M} x {N} fp32, averaged over {ITERS} calls (torch.profiler)\n"]
    rows = []
    for name, kernels in results.items():
        n_launch = sum(c for c, _ in kernels.values())
        total_us = sum(t for _, t in kernels.values())
        lines.append(f"[{name}] kernels per call: {n_launch:g}, total GPU time per call: {total_us:.1f} us")
        for k, (c, t) in sorted(kernels.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"    {c:g} x {t:8.1f} us  {k[:110]}")
            rows.append([name, k, c, round(t, 2)])
        lines.append("")
    lines.append(f"Bytes moved per call (computed from tensor shapes):")
    lines.append(f"    fused : {fused_bytes / 1e6:.1f} MB  (1 read + 1 write of x)")
    nb = sum(naive_bytes.values())
    lines.append(f"    naive : {nb / 1e6:.1f} MB  ({', '.join(f'{k} {v / 1e6:.1f}' for k, v in naive_bytes.items())})")
    lines.append(f"    ratio : {nb / fused_bytes:.2f}x")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "profile_softmax.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(RESULTS, "profile_softmax.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementation", "kernel", "launches_per_call", "gpu_us_per_call"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
