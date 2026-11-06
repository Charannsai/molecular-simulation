"""Experiment E0b: measure this GPU's matrix-multiply throughput (the compute roof).

    uv run --extra bench python bench/compute.py

FP16 matrix multiplies (8192 x 8192) on the tensor cores; 30 timed repetitions, median reported.
Used by the digital model to check whether decode is compute-bound or memory-bound.
Label: measured.
"""

from __future__ import annotations

import torch

from bandwidth import time_ms
from common import context, save, summary


def main(n: int = 8192, reps: int = 30) -> None:
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    ms = time_ms(lambda: a @ b, reps)
    tflops = [2 * n**3 / (t / 1e3) / 1e12 for t in ms]
    result = {
        "experiment": "E0b",
        "title": "GPU FP16 matrix-multiply throughput on the author's PC",
        "label": "measured",
        "method": f"PyTorch {torch.__version__}, {n}x{n} FP16 matmul, {reps} timed reps after 5 warm-ups, median reported",
        "context": context(),
        "fp16_TFLOPS": summary(tflops),
    }
    path = save("E0b_compute", result)
    print(f"FP16 matmul: {result['fp16_TFLOPS']['median']:.1f} TFLOP/s median; saved {path}")


if __name__ == "__main__":
    main()
