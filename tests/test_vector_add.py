import pytest
import torch

from kernels.vector_add import add


@pytest.mark.parametrize("n", [1, 7, 1000, 1024, 1025, 98_432, 1_000_003])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_add(n, dtype):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    torch.testing.assert_close(add(x, y), x + y)


@pytest.mark.parametrize("block_size", [128, 512, 4096])
def test_add_block_sizes(block_size):
    x = torch.randn(12_345, device="cuda")
    y = torch.randn(12_345, device="cuda")
    torch.testing.assert_close(add(x, y, block_size=block_size), x + y)
