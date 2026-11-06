"""Experiment E1: measure real LLM inference on this PC through Ollama.

    uv run python bench/measure_llm.py

For each model: load it, confirm it runs fully on the GPU, do one warm-up, then 5 measured runs.
Each run uses a slightly different prompt (so Ollama's prompt cache can't skip the prefill) and asks
for 128 generated tokens with temperature 0.

Timing comes from Ollama's own counters (nanosecond durations for prompt processing and generation).
GPU board power comes from nvidia-smi, sampled every 100 ms, and is averaged over the matching window.
Energy per token = average power during generation x generation time / tokens generated.

Label: measured (GPU board power only; whole-PC power would need a wall meter).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
import urllib.request

from common import context, save, summary

OLLAMA = "http://localhost:11434"
MODELS = [
    ("llama3.2:3b-instruct-q8_0", "Llama-3.2-3B", "Q8_0 (8-bit)"),
    ("llama3.1:8b-instruct-q4_K_M", "Llama-3.1-8B", "Q4_K_M (about 4.5-bit)"),
]
RUNS = 5
NUM_PREDICT = 128
NUM_CTX = 4096

PASSAGE = (
    "Large language models generate text one token at a time. For every new token the processor reads "
    "most of the model's weights from memory, so generation speed is usually limited by memory bandwidth "
    "rather than arithmetic. Compute-in-memory designs try to avoid that data movement by doing the "
    "multiplication inside the memory array itself. "
)


def post(path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(OLLAMA + path, timeout=30) as r:
        return json.loads(r.read())


class PowerSampler:
    """Background nvidia-smi sampling: (wall time, watts, MiB used)."""

    def __init__(self, interval_ms: int = 100):
        self.samples: list[tuple[float, float, float]] = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
            stdout=subprocess.PIPE, text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                w, mem = (float(x) for x in line.split(","))
                self.samples.append((time.time(), w, mem))
            except ValueError:
                pass

    def window(self, t0: float, t1: float) -> list[tuple[float, float, float]]:
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def stop(self):
        self.proc.terminate()


def prompt(i: int) -> str:
    return f"Note {i}. " + PASSAGE * 6 + "\nWrite a long, detailed explanation of why memory bandwidth limits LLM decoding, with examples."


def unload_all():
    for m in get("/api/ps").get("models", []):
        post("/api/generate", {"model": m["name"], "keep_alive": 0})


def measure(sampler: PowerSampler, tag: str) -> dict:
    # Load (and time) the model, then check it sits fully in GPU memory.
    t = time.time()
    first = post("/api/generate", {"model": tag, "prompt": "Hello", "stream": False, "keep_alive": "15m", "options": {"num_predict": 8, "num_ctx": NUM_CTX}})
    load_s = first.get("load_duration", 0) / 1e9
    ps = next((m for m in get("/api/ps")["models"] if m["name"] == tag), {})
    gpu_share = ps.get("size_vram", 0) / ps["size"] if ps.get("size") else None

    post("/api/generate", {"model": tag, "prompt": prompt(999), "stream": False, "options": {"num_predict": NUM_PREDICT, "temperature": 0, "seed": 42, "num_ctx": NUM_CTX}})  # warm-up

    runs = []
    for i in range(RUNS):
        t_start = time.time()
        r = post("/api/generate", {"model": tag, "prompt": prompt(i), "stream": False, "options": {"num_predict": NUM_PREDICT, "temperature": 0, "seed": 42, "num_ctx": NUM_CTX}})
        t_end = time.time()
        gen_s = r["eval_duration"] / 1e9
        pre_s = r["prompt_eval_duration"] / 1e9
        gen = sampler.window(t_end - gen_s, t_end)
        pre = sampler.window(t_end - gen_s - pre_s, t_end - gen_s)
        gen_w = statistics.mean(w for _, w, _ in gen) if gen else None
        runs.append({
            "prompt_tokens": r["prompt_eval_count"],
            "generated_tokens": r["eval_count"],
            "prefill_tok_s": r["prompt_eval_count"] / pre_s if pre_s else None,
            "decode_tok_s": r["eval_count"] / gen_s,
            "decode_power_W": gen_w,
            "prefill_power_W": statistics.mean(w for _, w, _ in pre) if pre else None,
            "decode_J_per_token": gen_w * gen_s / r["eval_count"] if gen_w else None,
            "power_samples_in_decode": len(gen),
            "vram_MiB": max((m for _, _, m in sampler.window(t_start, t_end)), default=None),
            "wall_s": t_end - t_start,
        })
        time.sleep(2)

    def med(key):
        xs = [x[key] for x in runs if x[key] is not None]
        return summary(xs) if xs else None

    return {
        "load_s": load_s,
        "gpu_share": gpu_share,
        "decode_tok_s": med("decode_tok_s"),
        "prefill_tok_s": med("prefill_tok_s"),
        "decode_power_W": med("decode_power_W"),
        "decode_J_per_token": med("decode_J_per_token"),
        "vram_MiB": med("vram_MiB"),
        "prompt_tokens": med("prompt_tokens"),
        "generated_tokens": med("generated_tokens"),
        "runs": runs,
    }


def main():
    unload_all()
    sampler = PowerSampler()
    time.sleep(6)
    idle = [w for _, w, _ in sampler.samples]
    results = {}
    try:
        for tag, name, quant in MODELS:
            print(f"measuring {name} {quant} ...", flush=True)
            results[tag] = {"model": name, "quantisation": quant, **measure(sampler, tag)}
            post("/api/generate", {"model": tag, "keep_alive": 0})
            time.sleep(3)
    finally:
        sampler.stop()

    payload = {
        "experiment": "E1",
        "title": "Baseline LLM inference on the author's PC (Ollama, GPU)",
        "label": "measured",
        "method": f"Ollama {get('/api/version')['version']}, {RUNS} runs after 1 warm-up, {NUM_PREDICT} generated tokens, temperature 0, num_ctx {NUM_CTX}, "
        "nvidia-smi board power every 100 ms aligned to Ollama's eval timings; medians reported",
        "context": context(),
        "idle_power_W": summary(idle) if idle else None,
        "models": results,
    }
    path = save("E1_baseline", payload)
    print(f"\nidle GPU power: {payload['idle_power_W']['median']:.1f} W")
    for r in results.values():
        print(
            f"{r['model']:14} {r['quantisation']:24} decode {r['decode_tok_s']['median']:6.1f} tok/s"
            f" | prefill {r['prefill_tok_s']['median']:7.0f} tok/s"
            f" | {r['decode_power_W']['median']:5.1f} W | {r['decode_J_per_token']['median']:.3f} J/tok"
            f" | VRAM {r['vram_MiB']['median']:.0f} MiB | on GPU {100 * (r['gpu_share'] or 0):.0f}%"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
