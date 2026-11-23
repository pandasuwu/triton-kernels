import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA GPU required")
        for item in items:
            item.add_marker(skip)
