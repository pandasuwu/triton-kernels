"""Fused row-wise softmax: out[i, :] = softmax(x[i, :]).

Why fuse? The naive PyTorch version
    m = x.max(dim=1); z = x - m; e = exp(z); s = e.sum(dim=1); out = e / s
launches ~5 kernels and writes/reads an intermediate full-size tensor
between each one. The fused kernel loads each row once into on-chip memory
(registers), does every step there, and writes the result once:
one read + one write of the matrix, the minimum possible.

Two code paths:
  * softmax_kernel          - the whole row fits in one block (n_cols <= MAX_FUSED_COLS).
                              Exactly 1 read + 1 write per element.
  * softmax_online_kernel   - row too long for one block. Loops over the row in
                              chunks using the "online softmax" recurrence
                              (running max and running sum), then a second pass
                              writes the output. 2 reads + 1 write per element.
"""
import torch
import triton
import triton.language as tl

# Largest row handled by the single-block path. BLOCK_SIZE is the next power
# of two >= n_cols, and the whole block lives in registers, so very large
# blocks spill. 16384 fp32 values per row is a practical limit.
MAX_FUSED_COLS = 16384


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,   # elements between the starts of consecutive rows
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,        # power of two >= n_cols
):
    # One program per row... but if there are more rows than programs we
    # launched, each program strides through several rows. The grid is sized
    # to the number of rows, so in practice this loop usually runs once.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # Pointer to the first element of this row. The stride lets us handle
        # non-contiguous inputs (e.g. a slice of a bigger matrix).
        row_in_ptr = in_ptr + row_idx * in_row_stride

        # Column offsets 0..BLOCK_SIZE-1. BLOCK_SIZE is rounded up to a power
        # of two (tl.arange requires it), so it can exceed n_cols.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # Mask off the padding columns beyond n_cols.
        mask = col_offsets < n_cols

        # THE ONE READ: the whole row comes from global memory into registers.
        # Padding lanes get -inf so they cannot win the max and contribute
        # exp(-inf) = 0 to the sum. Compute in fp32 even for fp16 input.
        row = tl.load(row_in_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

        # Numerical stability: softmax(x) == softmax(x - max(x)). Subtracting
        # the max makes the largest exponent exp(0) = 1, so exp never overflows.
        row_max = tl.max(row, axis=0)
        z = row - row_max

        # All of these stay on-chip: no intermediate tensor is written to
        # global memory (this is the saving over the unfused version).
        num = tl.exp(z)
        den = tl.sum(num, axis=0)
        out = num / den

        # THE ONE WRITE: the finished row goes back to global memory.
        row_out_ptr = out_ptr + row_idx * out_row_stride
        tl.store(row_out_ptr + col_offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def softmax_online_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,        # chunk width; the row is processed in chunks
):
    # One program per row; the row is too long to hold in registers at once.
    row_idx = tl.program_id(0)
    row_in_ptr = in_ptr + row_idx * in_row_stride
    row_out_ptr = out_ptr + row_idx * out_row_stride
    col = tl.arange(0, BLOCK_SIZE)

    # Pass 1: online softmax. Keep a per-lane running max m and running sum
    # s of exp(x - m). When a new chunk raises the max, rescale the old sum by
    # exp(m_old - m_new) so it stays consistent. One read of the row.
    m = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    s = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + col
        x = tl.load(row_in_ptr + offs, mask=offs < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # Lanes that have only ever seen -inf would compute exp(-inf - -inf) = nan;
        # guard by treating them as contributing zero.
        alpha = tl.where(m_new == -float("inf"), 0.0, tl.exp(m - m_new))
        beta = tl.where(m_new == -float("inf"), 0.0, tl.exp(x - m_new))
        s = s * alpha + beta
        m = m_new
    # Combine the per-lane (m, s) pairs into one row max and row sum.
    row_max = tl.max(m, axis=0)
    row_sum = tl.sum(s * tl.exp(m - row_max), axis=0)

    # Pass 2: read the row again, normalise, and write. Second read + one write.
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + col
        msk = offs < n_cols
        x = tl.load(row_in_ptr + offs, mask=msk, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(row_out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=msk)


def _num_warps(block_size: int) -> int:
    # More warps for wider rows, so each thread holds fewer elements.
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim of a 2-D CUDA tensor (fp16 / bf16 / fp32)."""
    assert x.ndim == 2 and x.is_cuda
    if x.stride(1) != 1:
        x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_cols <= MAX_FUSED_COLS:
        block = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](
            out, x, x.stride(0), out.stride(0), n_rows, n_cols,
            BLOCK_SIZE=block, num_warps=_num_warps(block),
        )
    else:
        block = 4096
        softmax_online_kernel[(n_rows,)](
            out, x, x.stride(0), out.stride(0), n_cols,
            BLOCK_SIZE=block, num_warps=8,
        )
    return out


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Unfused PyTorch reference: each line is a separate kernel launch and a
    full read/write of an intermediate tensor."""
    x_max = x.max(dim=1, keepdim=True).values   # read x, write n_rows
    z = x - x_max                               # read x, write z
    num = torch.exp(z)                          # read z, write num
    den = num.sum(dim=1, keepdim=True)          # read num, write n_rows
    return num / den                            # read num, write out
