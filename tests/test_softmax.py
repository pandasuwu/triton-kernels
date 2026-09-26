import pytest
import torch

from kernels.softmax import MAX_FUSED_COLS, naive_softmax, softmax

SHAPES = [(1, 1), (3, 7), (128, 128), (1823, 781), (64, 1000), (17, 4099), (8, MAX_FUSED_COLS)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_fused(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))


@pytest.mark.parametrize("n_cols", [MAX_FUSED_COLS + 1, 40_000, 65_537])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_online_long_rows(n_cols, dtype):
    torch.manual_seed(0)
    x = torch.randn(5, n_cols, device="cuda", dtype=dtype)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))


def test_softmax_large_values_stable():
    # Without max-subtraction exp(1000) overflows to inf.
    x = torch.randn(32, 513, device="cuda") * 1000
    out = softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=1))


def test_softmax_non_contiguous_rows():
    big = torch.randn(64, 2048, device="cuda")
    x = big[:, 100:1100]  # row stride 2048, n_cols 1000
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))


def test_naive_matches_torch():
    x = torch.randn(100, 333, device="cuda")
    torch.testing.assert_close(naive_softmax(x), torch.softmax(x, dim=1))
