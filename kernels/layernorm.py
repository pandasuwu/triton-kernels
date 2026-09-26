"""Fused LayerNorm forward: y = (x - mean) / sqrt(var + eps) * w + b, per row.

Unfused PyTorch would compute mean, subtract, square, mean again, rsqrt,
multiply, add - each a separate pass over the data. Here each program loads
its row once, computes both statistics in registers, and writes y once.
Rows up to MAX_COLS are supported (whole row in one block).
"""
import torch
import triton
import triton.language as tl

MAX_COLS = 16384


@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    x_row_stride, y_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,   # power of two >= n_cols
):
    # One program per row.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols                          # ignore padding columns

    # Single read of the row, promoted to fp32 for accurate statistics.
    # Padding lanes load 0 so they do not change the sum.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean over the real columns only (divide by n_cols, not BLOCK_SIZE).
    mean = tl.sum(x, axis=0) / n_cols
    # Centre; re-zero padding lanes, otherwise they would contribute
    # (0 - mean)^2 to the variance.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols       # biased variance, as torch uses
    rstd = 1.0 / tl.sqrt(var + eps)

    # The affine parameters are shared by all rows, so they are small and
    # mostly served from L2 after the first programs touch them.
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b

    # Single write of the row.
    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    assert n_cols <= MAX_COLS, f"layernorm supports rows up to {MAX_COLS} columns"
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    num_warps = 16 if block >= 8192 else 8 if block >= 2048 else 4
    layernorm_kernel[(n_rows,)](
        x, y, weight, bias, x.stride(0), y.stride(0), n_cols, eps,
        BLOCK_SIZE=block, num_warps=num_warps,
    )
    return y
