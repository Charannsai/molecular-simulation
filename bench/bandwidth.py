"""Experiment E0: measure this GPU's real memory bandwidth.

    uv run --extra bench python bench/bandwidth.py

Two tests on large buffers (far bigger than any cache):
- copy: read one buffer and write another (counts read + write bytes);
- read: sum a buffer (reads only), the closest pattern to LLM decode, which streams weights.

Each test runs 30 timed repetitions after a warm-up; we report the median.
Label: measured.
"""

from __future__ import annotations

import sys

import torch

from common import context, save, summary

GB = 1e9


def time_ms(fn, reps: int) -> list[float]:
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main(size_gb: float = 1.5, reps: int = 30) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch")
    n = int(size_gb * GB / 4)  # float32 elements
    src = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    dst = torch.empty_like(src)
    nbytes = src.numel() * 4

    for _ in range(5):  # warm-up
        dst.copy_(src)
        src.sum()
    torch.cuda.synchronize()

    copy_ms = time_ms(lambda: dst.copy_(src), reps)
    read_ms = time_ms(lambda: src.sum(), reps)

    copy_gbs = [2 * nbytes / (t / 1e3) / GB for t in copy_ms]  # read + write
    read_gbs = [nbytes / (t / 1e3) / GB for t in read_ms]

    result = {
        "experiment": "E0",
        "title": "GPU memory bandwidth on the author's PC",
        "label": "measured",
        "method": f"PyTorch {torch.__version__}, CUDA events, {size_gb} GB float32 buffers, {reps} timed reps after 5 warm-ups, median reported",
        "context": context(),
        "copy_bandwidth_GBps": summary(copy_gbs),
        "read_bandwidth_GBps": summary(read_gbs),
    }
    path = save("E0_bandwidth", result)
    print(f"GPU: {result['context']['gpu']['name']}")
    print(f"copy (read+write): {result['copy_bandwidth_GBps']['median']:.1f} GB/s median")
    print(f"read only:         {result['read_bandwidth_GBps']['median']:.1f} GB/s median")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
