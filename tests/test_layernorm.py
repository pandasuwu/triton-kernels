import pytest
import torch
import torch.nn.functional as F

from kernels.layernorm import layernorm


@pytest.mark.parametrize("shape", [(1, 1), (4, 7), (128, 768), (1000, 1023), (33, 4097), (8, 16384)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_layernorm(shape, dtype):
    torch.manual_seed(0)
    n_cols = shape[1]
    x = torch.randn(shape, device="cuda", dtype=dtype) * 3 + 1
    w = torch.randn(n_cols, device="cuda", dtype=dtype)
    b = torch.randn(n_cols, device="cuda", dtype=dtype)
    ref = F.layer_norm(x.float(), (n_cols,), w.float(), b.float(), 1e-5).to(dtype)
    torch.testing.assert_close(layernorm(x, w, b), ref, atol=1e-2 if dtype == torch.float16 else 1e-4, rtol=1e-2)
