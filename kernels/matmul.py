"""Tiled matmul with a fused activation epilogue: C = act(A @ B).

A is (M, K), B is (K, N), both fp16. Accumulation is in fp32, output fp16.

Ideas shown here:
  * Block tiling: each program computes one BLOCK_M x BLOCK_N tile of C by
    walking along K in BLOCK_K steps, so every A/B tile it loads is reused
    BLOCK_N (resp. BLOCK_M) times from on-chip memory.
  * Grouped program ordering: programs are re-numbered so that consecutive
    programs work on a GROUP_M-tall band of C, which makes them share the same
    rows of A and columns of B while those are still in L2 cache.
  * Fused epilogue: the activation is applied to the fp32 accumulator while it
    is still in registers, so C is written once. The unfused PyTorch version
    writes C, then reads it back and writes it again for the activation.
  * @triton.autotune picks the best tile shape per (M, N, K).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _configs():
    # A handful of tile shapes / pipeline depths. num_stages controls how many
    # K-tiles are prefetched ahead (software pipelining through shared memory).
    cfg = triton.Config
    return [
        cfg({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        cfg({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        cfg({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        cfg({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
        cfg({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        cfg({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    ]


@triton.jit
def _gelu_tanh(x):
    # GELU, tanh approximation (matches F.gelu(approximate="tanh")):
    #   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    # Written with sigmoid because tanh(u) = 2*sigmoid(2u) - 1, so
    # 0.5 * (1 + tanh(u)) = sigmoid(2u).
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * tl.sigmoid(2.0 * u)


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,      # how far to move in memory for +1 row / +1 col of A
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACTIVATION: tl.constexpr,  # "none", "leaky_relu" or "gelu"
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- Which output tile does this program own? ----------------------
    # The grid is 1-D with num_pid_m * num_pid_n programs. The naive mapping
    # (pid_m = pid // num_pid_n) walks C row-by-row, so consecutive programs
    # need all of B before they reuse any of it. Grouped ordering walks
    # GROUP_M tile-rows column-by-column instead: a group of programs shares
    # a few rows of A and a few columns of B, which stay hot in L2.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)  # last group may be short
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Pointers to the first K-tile of A and B --------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows of C (and A) this tile covers
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols of C (and B)
    offs_k = tl.arange(0, BLOCK_K)
    # Broadcasting a column vector [:, None] against a row vector [None, :]
    # gives a 2-D block of addresses: a_ptrs[i, j] = &A[offs_m[i], offs_k[j]].
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- Main loop over K -------------------------------------------------
    # Accumulate in fp32: fp16 has only ~3 decimal digits, and summing K
    # products in fp16 would lose precision badly for large K.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Masks: rows past M, cols past N and the ragged last K-tile load 0,
        # which adds nothing to the dot product.
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # tl.dot lowers to tensor-core MMA instructions; fp16 in, fp32 out.
        acc = tl.dot(a, b, acc)
        # Slide both tiles BLOCK_K along the K dimension.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Fused epilogue ---------------------------------------------------
    # acc is still in registers; applying the activation here costs no extra
    # memory traffic. The unfused version pays a full extra read + write of C.
    if ACTIVATION == "leaky_relu":
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    elif ACTIVATION == "gelu":
        acc = _gelu_tanh(acc)
    c = acc.to(tl.float16)

    # ---- Write the tile of C once ----------------------------------------
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "none") -> torch.Tensor:
    """C = activation(a @ b) for fp16 CUDA matrices. activation: none | leaky_relu | gelu."""
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert activation in ("none", "leaky_relu", "gelu")
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # The grid depends on the tile size the autotuner chose, so it is a lambda.
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


def torch_reference(a: torch.Tensor, b: torch.Tensor, activation: str = "none") -> torch.Tensor:
    """Unfused PyTorch: cuBLAS matmul, then a separate activation kernel."""
    c = torch.matmul(a, b)
    if activation == "leaky_relu":
        return F.leaky_relu(c, 0.01)
    if activation == "gelu":
        return F.gelu(c, approximate="tanh")
    return c
