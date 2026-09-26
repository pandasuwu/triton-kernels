# Walkthrough

One section per kernel: the idea, the code path, the measured numbers, and
questions that are likely to come up, with answers. All numbers are from
`results/` (RTX 5060 Laptop GPU, see README for the exact environment).

## Triton in one paragraph

A Triton kernel is written from the point of view of one *program instance*
(roughly a CUDA thread block). You launch a grid of them; each finds out who it
is with `tl.program_id`, builds a block of offsets with `tl.arange`, and does
`tl.load` / compute / `tl.store` on whole blocks at once. The compiler decides
how a block is split across threads, how loads are vectorised and coalesced,
and how shared memory and tensor cores are used. You decide the decomposition
of the problem into blocks, what lives on-chip, and the memory traffic.

---

## 1. Vector add (`kernels/vector_add.py`)

**Idea.** `out = x + y`. Purely memory bound: 2 loads and 1 store per element,
one add. The only question is how close we get to DRAM bandwidth.

**Code path.**
1. `add()` launches `cdiv(n, BLOCK_SIZE)` programs on a 1-D grid.
2. Program `pid` owns elements `[pid*BLOCK_SIZE, (pid+1)*BLOCK_SIZE)`.
3. `offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`; `x_ptr + offsets` is a
   block of addresses.
4. `mask = offsets < n` protects the last, partial block.
5. Load, add, store with the same mask.

**Numbers.** At 2^28 fp32 elements: Triton 345 GB/s, PyTorch eager 345 GB/s,
torch.compile 347 GB/s. All three reach the same plateau, which is the practical
DRAM bandwidth of this GPU. At 4096 elements every version gets about 9 GB/s:
the kernel takes a few microseconds whatever it does, so the time is launch
overhead.

**Likely questions.**
- *Why is BLOCK_SIZE a `tl.constexpr`?* `tl.arange` needs a compile-time size,
  and the compiler specialises the kernel (register allocation, vector width)
  for it. Each distinct value compiles a separate kernel.
- *What happens without the mask?* The last program reads and writes past the
  end of the allocation: garbage values at best, an illegal-address fault or
  silent corruption of a neighbouring tensor at worst.
- *Why does Triton not beat PyTorch here?* Both are at the bandwidth limit. A
  memory-bound kernel with no reuse cannot do better than moving the minimum
  bytes, and both already do that. Small sizes are dominated by launch
  overhead, not bandwidth.
- *How do you compute GB/s?* `3 * n * bytes_per_element / time`. It counts the
  bytes the algorithm must move, so it is comparable to the GPU's peak DRAM
  bandwidth.

---

## 2. Fused softmax (`kernels/softmax.py`)

**Idea.** The naive PyTorch version runs max, subtract, exp, sum and divide as
separate kernels. Each one reads a full-size tensor from DRAM and most write
one back. The fused kernel loads a row into registers once, does all five steps
on-chip and writes once. Softmax is memory bound, so fewer bytes means faster.

**Code path (`softmax_kernel`, rows up to 16384 columns).**
1. One program per row (grid = `n_rows`). The `tl.range(row_start, n_rows,
   row_step)` loop lets one program take several rows if the grid were smaller.
2. `BLOCK_SIZE = next_power_of_2(n_cols)`; the mask hides the padding columns.
3. Load with `other=-inf`, so padding never wins the max and `exp(-inf) = 0`
   adds nothing to the sum.
4. Subtract the row max before `exp` (numerical stability), sum, divide, store.
5. fp16 input is upcast to fp32 for the arithmetic and cast back on store.

**Long rows (`softmax_online_kernel`).** If the row is longer than 16384 the
whole row cannot sit in registers. The kernel then walks the row in 4096-wide
chunks and keeps a running max `m` and running sum `s`. When a chunk raises the
max, the old sum is rescaled by `exp(m_old - m_new)`. A second pass normalises
and writes. That is 2 reads + 1 write instead of 1 + 1, still far less than the
naive version. This is the same recurrence FlashAttention uses.

**Numbers.** 4096 rows, fp32, effective GB/s:

| Columns | Triton fused | torch.softmax | Naive (unfused) | torch.compile(naive) |
|---|---|---|---|---|
| 128 | 181 | 319 | 101 | 327 |
| 4096 | 336 | 312 | 86 | 336 |
| 16384 | 335 | 333 | 86 | 339 |

Two things to be ready to explain:
- At 128 columns the Triton kernel is slow (181 GB/s). Each program handles one
  128-element row with 4 warps (128 threads), so each thread holds a single
  element and the launch has 4096 tiny programs. Processing several rows per
  program, or using 1 warp for short rows, would fix it.
- `torch.softmax` dips to 230-310 GB/s between 2176 and 4096 columns, then
  recovers, in both runs. My guess (not verified, it would need a profiler) is
  that PyTorch switches between internal softmax kernels around that width;
  the Triton kernel does not change strategy there.

**Profiling.** (`bench/profile_softmax.py`, 4096 x 4096 fp32.) Kernels counted
from a captured CUDA graph: fused = 1 (`softmax_kernel`), naive = 5 (max
reduce, subtract, exp, sum reduce, divide). GPU time: 408 us vs 1550 us (3.80x).
Bytes from tensor shapes: 134.2 MB vs 536.9 MB (4.00x). The time ratio tracks
the byte ratio, which is what you expect from a memory-bound op: the work is
the same, only the traffic changes. The per-step times show it directly: the
full-tensor elementwise steps (subtract, exp, divide) each take about 400-425 us,
and the two reductions, which read the matrix but write only one value per row,
about 211-216 us.

Why not torch.profiler? On this machine CUPTI fails with
`CUPTI_ERROR_INVALID_DEVICE`, so the profiler records no GPU events. The CUDA
graph capture gives the same kernel list without CUPTI.

**Likely questions.**
- *Why subtract the max?* `exp(x)` overflows fp32 for x > ~88. `softmax(x) ==
  softmax(x - c)` for any c, and choosing c = max makes every exponent <= 0.
  The test `test_softmax_large_values_stable` uses inputs of about 1000.
- *Why `other=-inf` and not 0?* A 0 could be larger than every real value in
  the row (all negative), corrupting the max, and `exp(0 - max)` would add to
  the sum.
- *Why does performance drop for very wide rows?* Each program holds the whole
  row in registers. As BLOCK_SIZE grows, registers per thread grow and fewer
  programs fit per SM (lower occupancy), or values spill to local memory.
- *How does `torch.softmax` compare?* It is already a fused, hand-tuned CUDA
  kernel, so the gap there is small. The big gap is against the unfused
  version, which is what a naive model implementation pays.
- *What does torch.compile do here?* Inductor fuses the naive graph into a
  generated Triton reduction kernel, so it should land close to the hand-written
  fused kernel. That is the point of the comparison.

---

## 3. Matmul with fused GELU (`kernels/matmul.py`)

**Idea.** `C = gelu(A @ B)` with fp16 inputs and fp32 accumulation. Matmul is
compute bound for large sizes: each tile of A and B loaded on-chip is reused
many times. The fused epilogue applies GELU to the accumulator in registers, so
C is written once instead of written, re-read and re-written.

**Code path.**
1. 1-D grid of `cdiv(M, BM) * cdiv(N, BN)` programs, each owning one BM x BN
   tile of C.
2. *Grouped ordering.* `pid` is remapped so that consecutive programs cover a
   GROUP_M-tall band of tiles, column by column. Programs running at the same
   time then share rows of A and columns of B, and those stay in L2.
3. Pointer blocks `a_ptrs[i, k] = A + m_i*stride_am + k*stride_ak` (and
   similarly for B) are built by broadcasting `[:, None]` against `[None, :]`.
   The strides make transposed or sliced inputs work without a copy.
4. K loop: masked loads (out-of-range rows, columns and the ragged last K tile
   load 0), `acc = tl.dot(a, b, acc)` (tensor cores, fp32 accumulate), advance
   the pointers by BLOCK_K.
5. Epilogue: LeakyReLU or GELU on `acc`, cast to fp16, masked store.
6. `@triton.autotune(key=["M","N","K"])` benchmarks 6 tile configs the first
   time it sees a shape and caches the winner.

**Numbers.** Square fp16, TFLOPS:

| Size | Triton fused | torch.matmul + GELU | cuBLAS only | torch.compile |
|---|---|---|---|---|
| 256 | 3.9 | 2.5 | 3.7 | 2.4 |
| 512 | 15.8 | 12.2 | 15.3 | 12.1 |
| 2048 | 32.8 | 31.4 | 32.4 | 32.2 |
| 4096 | 32.5 | 30.0 | 29.5 | 31.0 |

- Fusion helps most at small sizes (1.3-1.5x at 256-512), where the extra GELU
  launch and the extra round trip of C are a large share of the total time.
  At 4096 the gain is 1.08x.
- The Triton matmul is roughly level with cuBLAS on this GPU from 1024 up.
- torch.compile does not fuse GELU into the matmul here. Its generated code
  (`TORCH_LOGS=output_code`) is `extern_kernels.mm` (cuBLAS) followed by a
  separate `triton_poi_fused_gelu` kernel. Inductor's Triton GEMM templates are
  only tried in max-autotune mode, and it printed "Not enough SMs to use
  max_autotune_gemm mode" (this GPU has 26 SMs).
- The cuBLAS-only line dips below cuBLAS + GELU at 3840-4096 in both sweeps,
  but a standalone run at 4096 gave 32.9-34.1 TFLOPS. The laptop GPU runs into
  its power cap (clocks moved between 1357 and 2767 MHz), so a few percent at
  the large end is noise. Do not read that dip as a real effect.

**Likely questions.**
- *Why accumulate in fp32?* fp16 has a 10-bit mantissa. Adding thousands of
  products in fp16 loses low-order bits and, for large K, the error grows with
  K. Tensor cores take fp16 inputs and accumulate in fp32 at full speed anyway.
- *What does GROUP_M buy you?* L2 hit rate. With row-major ordering, the
  programs running at once span one row of tiles, so they need all of B before
  anything is reused. Grouping makes the working set roughly square, so more
  loads hit in L2. It changes nothing about correctness.
- *What do BLOCK_K and num_stages control?* BLOCK_K is how much of K each
  iteration consumes (bigger = fewer loop iterations, more shared memory).
  num_stages is how many future K tiles are prefetched into shared memory while
  the current one is being multiplied (software pipelining to hide DRAM
  latency).
- *Why isn't the gain bigger at large sizes?* A large matmul is compute bound:
  for N = 4096 the GEMM does 2*N^3 = 137 GFLOP, while the extra GELU pass only
  moves 2 * 4096^2 * 2 bytes = 67 MB. The epilogue saving is a fixed cost that
  shrinks relative to the N^3 work as N grows.
- *How is TFLOPS computed?* `2*M*N*K / time` (one multiply and one add per
  inner-product term). The activation FLOPs are not counted.

---

## 4. LayerNorm forward (`kernels/layernorm.py`)

**Idea.** Same pattern as softmax: one program per row, load once, compute mean
and variance in registers, normalise, apply weight and bias, write once.

**Code path.** Load the row (padding = 0), `mean = sum / n_cols`, centre, and
re-zero the padding lanes with `tl.where` (otherwise each one would add
`mean^2` to the variance). `rstd = 1/sqrt(var + eps)`, then `y = xc * rstd * w
+ b`.

**Numbers.** 4096 rows, fp16, effective GB/s:

| Columns | Triton fused | F.layer_norm | Naive (unfused) | torch.compile(naive) |
|---|---|---|---|---|
| 256 | 176 | 132 | 41 | 315 |
| 4096 | 332 | 322 | 42 | 332 |
| 16384 | 345 | 249 | 37 | 340 |

The naive version is about 8x slower here: it launches many more kernels than
softmax (mean, subtract, square, mean, add eps, sqrt, divide, multiply, add),
and each one moves a full tensor. Short rows have the same weakness as in
softmax: at 256 columns torch.compile is 1.8x faster than this
one-row-per-program kernel. Inductor's generated reduction kernel can map
several short rows to one program; reading its `output_code` for that shape
would confirm how it splits the work.

**Likely questions.**
- *Why divide by n_cols and not BLOCK_SIZE?* BLOCK_SIZE includes padding lanes.
- *Biased or unbiased variance?* Biased (divide by N), matching
  `torch.nn.functional.layer_norm`.
- *What would the backward pass need?* Saving `mean` and `rstd` per row, and a
  reduction over rows for dW and dB, usually done with a two-stage partial-sum
  scheme to avoid atomics contention.
