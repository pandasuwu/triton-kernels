# triton-kernels

Hand-written Triton kernels (vector add, fused softmax, tiled matmul with a fused activation, fused LayerNorm), each tested against PyTorch and benchmarked against PyTorch eager and `torch.compile` on a laptop RTX 5060.

## Environment

All results in `results/` were produced on this setup (`results/environment.txt`, written by `bench/env_info.py`):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, compute capability 12.0, 26 SMs, 8 GB, 32 MB L2) |
| Driver | 610.91 (nvidia-smi reports CUDA 13.3) |
| CUDA (PyTorch build) | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 (`triton-windows` 3.6.0.post26) |
| Python | 3.12.14 |
| OS | Windows 11 (10.0.26300), native, no WSL |

The laptop was on AC power. The GPU is power-capped (nvidia-smi showed SM clocks from 1357 to 2767 MHz and the software power-cap throttle flag during matmul runs), so repeated runs differ by a few percent.

## Layout

```
kernels/   vector_add.py  softmax.py  matmul.py  layernorm.py
tests/     pytest correctness tests against PyTorch
bench/     bench_*.py, profile_softmax.py, env_info.py, run_all.py
results/   CSV + PNG per benchmark, profile output, environment.txt
WALKTHROUGH.md   per-kernel explanation and interview questions
```

## Running

```
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "triton-windows>=3.6,<3.7" numpy pytest matplotlib pandas   # on Linux: triton
.venv\Scripts\activate

pytest                     # 74 tests
python bench/run_all.py    # regenerates everything in results/
```

On Windows, `triton-windows` bundles its own CUDA toolchain and C compiler; only the Visual C++ 2015-2022 redistributable is needed.

## Results

Timing uses `triton.testing.do_bench` (median, L2 flushed between runs). `torch.compile` uses static shapes.

### Vector add (fp32)

![vector add](results/vector_add.png)

| Elements | Triton | PyTorch eager | torch.compile |
|---|---|---|---|
| 2^20 | 336 GB/s | 320 GB/s | 323 GB/s |
| 2^28 | 345 GB/s | 345 GB/s | 347 GB/s |

All three saturate at about 345 GB/s. Below about 2^18 elements the time is dominated by launch overhead.

### Fused softmax (fp32, 4096 rows)

![softmax](results/softmax.png)

| Columns | Triton fused | torch.softmax | Naive PyTorch (5 kernels) | torch.compile(naive) |
|---|---|---|---|---|
| 1024 | 333 GB/s | 335 GB/s | 140 GB/s | 336 GB/s |
| 4096 | 336 GB/s | 312 GB/s | 86 GB/s | 336 GB/s |
| 16384 | 335 GB/s | 333 GB/s | 86 GB/s | 339 GB/s |

GB/s is effective bandwidth (one read and one write of the matrix divided by time). The fused kernel is 3.9x faster than the naive version at 4096 columns. `torch.softmax` drops to 230-310 GB/s between 2176 and 4096 columns. At 128 columns the Triton kernel reaches only 181 GB/s against 319 GB/s for `torch.softmax`: one program per 128-element row leaves most of its 4 warps idle.

### Matmul + GELU (fp16 in, fp32 accumulate, M = N = K)

![matmul](results/matmul_gelu_fp16.png)

| Size | Triton fused | torch.matmul + GELU | cuBLAS matmul only | torch.compile(matmul + GELU) |
|---|---|---|---|---|
| 512 | 15.8 TFLOPS | 12.2 TFLOPS | 15.3 TFLOPS | 12.1 TFLOPS |
| 2048 | 32.8 TFLOPS | 31.4 TFLOPS | 32.4 TFLOPS | 32.2 TFLOPS |
| 4096 | 32.5 TFLOPS | 30.0 TFLOPS | 29.5 TFLOPS | 31.0 TFLOPS |

TFLOPS counts the 2*M*N*K matmul FLOPs only. The fused kernel is 1.08x faster than the unfused PyTorch version at 4096 and 1.3x at 512. `@triton.autotune` chooses among 6 tile configurations per shape. Inductor printed "Not enough SMs to use max_autotune_gemm mode", so `torch.compile` keeps the cuBLAS matmul and adds a separate fused activation kernel; it does not fuse GELU into the GEMM.

The "cuBLAS only" line falling below "cuBLAS + GELU" at 3840-4096 appeared in both full sweeps. A standalone re-measurement at 4096 gave cuBLAS 32.9-34.1 TFLOPS, so the dip comes from the GPU's power/clock state during the sweep, not from cuBLAS itself. Treat differences of a few percent at large sizes as noise.

### LayerNorm forward (fp16, 4096 rows)

![layernorm](results/layernorm_fp16.png)

| Columns | Triton fused | F.layer_norm | Naive PyTorch | torch.compile(naive) |
|---|---|---|---|---|
| 256 | 176 GB/s | 132 GB/s | 41 GB/s | 315 GB/s |
| 4096 | 332 GB/s | 322 GB/s | 42 GB/s | 332 GB/s |
| 16384 | 345 GB/s | 249 GB/s | 37 GB/s | 340 GB/s |

## Profiling: fused vs unfused softmax

Nsight Compute is not installed, and `torch.profiler` records no GPU activity on this machine (CUPTI fails with `CUPTI_ERROR_INVALID_DEVICE`). `bench/profile_softmax.py` therefore captures one call of each implementation into a CUDA graph and counts the kernel nodes, times each step with `do_bench`, and computes bytes from the tensor shapes each kernel reads and writes. For a 4096 x 4096 fp32 input, the fused Triton softmax runs 1 kernel in 408 us and moves 134.2 MB. The naive PyTorch version runs 5 kernels (max, subtract, exp, sum, divide) in 1550 us and moves 536.9 MB. That is 4.00x the bytes for a 3.80x slowdown, consistent with both versions being memory bound. Full output: `results/profile_softmax.txt`.

## Notes

