"""Element-wise vector addition: out = x + y.

The simplest possible Triton kernel. It shows the three ideas every other
kernel in this repo builds on:
  1. a 1-D grid of program instances, each owning one contiguous block,
  2. pointer arithmetic with tl.arange to address that block,
  3. a mask so the last block does not read or write past the end.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,            # pointer to the first element of x (in GPU memory)
    y_ptr,            # pointer to the first element of y
    out_ptr,          # pointer to the first element of the output
    n_elements,       # total number of elements (runtime value, can be anything)
    BLOCK_SIZE: tl.constexpr,  # elements per program; compile-time constant
):
    # Each program instance ("program") handles one block of BLOCK_SIZE
    # elements. program_id(0) is this program's index along grid axis 0,
    # e.g. 0, 1, 2, ... up to cdiv(n_elements, BLOCK_SIZE) - 1.
    pid = tl.program_id(axis=0)

    # First element this program is responsible for.
    block_start = pid * BLOCK_SIZE

    # offsets is a vector [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1].
    # Triton operates on whole blocks at once: this one line describes
    # BLOCK_SIZE addresses, which the compiler spreads across the threads
    # of the program.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # If n_elements is not a multiple of BLOCK_SIZE, the last program's
    # offsets run past the end of the arrays. The mask is True only for
    # valid positions; masked-off lanes neither load nor store, so we never
    # touch memory we do not own.
    mask = offsets < n_elements

    # x_ptr + offsets is a block of pointers (base address + element index;
    # Triton scales by the element size for us). Each element is read from
    # global memory exactly once.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y

    # Each output element is written exactly once. Total traffic is the
    # minimum possible: 2 reads + 1 write per element.
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Launch add_kernel on two same-shaped contiguous CUDA tensors."""
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    # One program per block; cdiv rounds up so the tail is covered.
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out
