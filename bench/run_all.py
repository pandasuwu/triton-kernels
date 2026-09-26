"""Regenerate every CSV, plot and profile under results/.

Usage:  python bench/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "env_info.py",
    "bench_vector_add.py",
    "bench_softmax.py",
    "bench_matmul.py",
    "bench_layernorm.py",
    "profile_softmax.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n===== {script} =====", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)


if __name__ == "__main__":
    main()
