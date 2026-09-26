"""Shared benchmark helpers."""
import os
import sys

import torch
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# torch.compile recompiles for every new static shape. The default
# recompile limit (8) is smaller than our sweeps, and hitting it silently
# falls back to eager, which would make the "compiled" line a lie.
import torch._dynamo  # noqa: E402

for _name in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 256)

QUANTILES = [0.5, 0.2, 0.8]  # median, p20, p80


def bench(fn):
    """Median / p20 / p80 runtime in ms, via triton.testing.do_bench."""
    return triton.testing.do_bench(fn, quantiles=QUANTILES, warmup=50, rep=200)


def compiled(fn):
    """torch.compile with static shapes (one specialised graph per size)."""
    return torch.compile(fn, dynamic=False)
