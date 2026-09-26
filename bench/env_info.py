"""Record the real environment the results were produced on."""
import os
import platform
import subprocess

import torch
import triton

from common import RESULTS


def main():
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    lines = [
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"GPU: {torch.cuda.get_device_name(0)}",
        f"Compute capability: {torch.cuda.get_device_capability(0)}",
        f"nvidia-smi (name, driver, memory, max SM clock, max mem clock, power limit): {smi}",
        f"PyTorch: {torch.__version__}",
        f"PyTorch CUDA: {torch.version.cuda}",
        f"cuDNN: {torch.backends.cudnn.version()}",
        f"Triton: {triton.__version__}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "environment.txt"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
