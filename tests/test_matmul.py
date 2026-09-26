import pytest
import torch

from kernels.matmul import matmul, torch_reference

SHAPES = [(1, 1, 1), (16, 16, 16), (128, 256, 64), (333, 517, 129), (1000, 1000, 1000), (2048, 512, 1536), (1, 4096, 77)]


@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("activation", ["none", "leaky_relu", "gelu"])
def test_matmul(M, N, K, activation):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = matmul(a, b, activation)
    # Reference in fp32 from the same fp16 inputs, then rounded to fp16:
    # both sides accumulate in fp32, but in different orders, so allow a
    # tolerance that grows with K.
    ref = torch_reference(a.float(), b.float(), activation).half()
    tol = 1e-2 * max(1.0, K ** 0.5 / 8)
    torch.testing.assert_close(out, ref, atol=tol, rtol=1e-2)


def test_matmul_transposed_input():
    # Non-contiguous B (column-major) exercises the stride arguments.
    a = torch.randn(300, 200, device="cuda", dtype=torch.float16)
    b = torch.randn(400, 200, device="cuda", dtype=torch.float16).t()
    ref = torch_reference(a.float(), b.float(), "gelu").half()
    torch.testing.assert_close(matmul(a, b, "gelu"), ref, atol=2e-2, rtol=1e-2)
