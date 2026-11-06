"""Experiment E2: can the digital model predict the measured GPU speed (E1)?

    uv run python experiments/e2_digital.py

Inputs (all measured on the author's PC): read bandwidth (E0), FP16 throughput (E0b),
model file sizes (Ollama), decode speed, power and context (E1).

Validation is cross-validated so it can't be circular:
- the utilisation (achieved / ceiling) is calibrated on ONE model,
- then used to predict the OTHER model's speed, and the reverse.
Pass if both predictions are within 30% (docs/simulator-plan.md). We also report the error of the
raw ceiling (utilisation 1), which shows how much a pure spec-sheet calculation would be off.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import RESULTS, context, save  # noqa: E402
from molsim import Value  # noqa: E402
from molsim.digital import predict, roofline  # noqa: E402
from molsim.llm import Model  # noqa: E402

TOLERANCE = 0.30
OLLAMA_TO_MODEL = {"llama3.2:3b-instruct-q8_0": "llama-3.2-3b", "llama3.1:8b-instruct-q4_K_M": "llama-3.1-8b"}


def ollama_sizes() -> dict[str, int]:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=30) as r:
        return {m["name"]: m["size"] for m in json.loads(r.read())["models"]}


def main() -> None:
    e0 = json.loads((RESULTS / "E0_bandwidth.json").read_text())
    e0b = json.loads((RESULTS / "E0b_compute.json").read_text())
    e1 = json.loads((RESULTS / "E1_baseline.json").read_text())
    sizes = ollama_sizes()

    bandwidth = Value.of(e0["read_bandwidth_GBps"]["median"], "GB/s", label="measured", source="E0 read bandwidth")
    throughput = Value.of(e0b["fp16_TFLOPS"]["median"], "Top/s", label="measured", source="E0b FP16 matmul")
    idle_w = e1["idle_power_W"]["median"]

    rows = {}
    for tag, key in OLLAMA_TO_MODEL.items():
        r = e1["models"][tag]
        model = Model.load(key)
        file_bytes = Value.of(sizes[tag], "B", label="measured", source=f"Ollama model file size ({tag})")
        ctx = int(r["prompt_tokens"]["median"] + r["generated_tokens"]["median"] / 2)  # average context while generating
        rl = roofline(model, file_bytes, ctx, bandwidth, throughput)
        measured = Value.of(r["decode_tok_s"]["median"], "tok/s", label="measured", source="E1 median decode speed")
        utilisation = (measured / rl["ceiling"]).to("")
        moved = (rl["total"] * measured).to("GB/s")  # bytes actually streamed per second
        dyn_w = Value.of(r["decode_power_W"]["median"] - idle_w, "W", label="measured", source="E1 decode power minus idle")
        rows[tag] = {
            "model": key, "context": ctx, "roofline": rl, "measured": measured, "utilisation": utilisation,
            "streamed_bandwidth": moved,
            # GPU energy above idle per byte of weights streamed (covers DRAM, caches and arithmetic together)
            "energy_per_byte": (dyn_w / moved).to("pJ/B"),
            "compute_share": (measured / rl["compute_ceiling"]).to(""),
        }

    tags = list(rows)
    checks = []
    for target, other in [(tags[0], tags[1]), (tags[1], tags[0])]:
        t, o = rows[target], rows[other]
        pred = predict(t["roofline"]["ceiling"], o["utilisation"])
        err = (pred.m - t["measured"].m) / t["measured"].m
        raw_err = (t["roofline"]["ceiling"].m - t["measured"].m) / t["measured"].m
        checks.append({
            "target": t["model"], "calibrated_on": o["model"], "utilisation_used": o["utilisation"].m,
            "predicted_tok_s": pred.m, "measured_tok_s": t["measured"].m, "error_pct": 100 * err,
            "raw_ceiling_tok_s": t["roofline"]["ceiling"].m, "raw_ceiling_error_pct": 100 * raw_err,
            "pass": abs(err) <= TOLERANCE,
        })

    passed = all(c["pass"] for c in checks)
    payload = {
        "experiment": "E2",
        "title": "Digital model validation against measured GPU decode",
        "label": "estimated (predictions) vs measured (targets)",
        "method": "Roofline: ceiling = min(bandwidth / bytes per token, compute / ops per token); utilisation calibrated on one model, applied to the other",
        "tolerance_pct": 100 * TOLERANCE,
        "passed": passed,
        "context": context(),
        "inputs": {"read_bandwidth_GBps": bandwidth.m, "fp16_TFLOPS": throughput.m, "file_bytes": {t: sizes[t] for t in tags}},
        "models": {
            t: {
                "model": r["model"], "avg_context_tokens": r["context"],
                "bytes_per_param": r["roofline"]["bytes_per_param"].to("B").m,
                "weight_GB_per_token": r["roofline"]["weights"].m, "kv_GB_per_token": r["roofline"]["kv_cache"].m,
                "memory_ceiling_tok_s": r["roofline"]["memory_ceiling"].m, "compute_ceiling_tok_s": r["roofline"]["compute_ceiling"].m,
                "measured_tok_s": r["measured"].m, "utilisation": r["utilisation"].m,
                "streamed_GBps": r["streamed_bandwidth"].m, "compute_share": r["compute_share"].m,
                "dynamic_energy_pJ_per_byte": r["energy_per_byte"].m,
            }
            for t, r in rows.items()
        },
        "cross_validation": checks,
    }
    path = save("E2_digital", payload)

    print(f"inputs: read bandwidth {bandwidth.m:.1f} GB/s, FP16 {throughput.m:.1f} TFLOP/s (both measured)\n")
    for t, m in payload["models"].items():
        print(
            f"{m['model']:13} {m['bytes_per_param'] * 8:4.2f} bits/weight | {m['weight_GB_per_token']:.2f} GB weights + {m['kv_GB_per_token'] * 1000:.0f} MB KV per token"
            f" | memory ceiling {m['memory_ceiling_tok_s']:5.1f} | compute ceiling {m['compute_ceiling_tok_s']:6.0f} tok/s"
            f" | measured {m['measured_tok_s']:.1f} | utilisation {100 * m['utilisation']:.1f}% | compute used {100 * m['compute_share']:.1f}%"
        )
    print()
    for c in checks:
        print(
            f"predict {c['target']:13} (calibrated on {c['calibrated_on']}): {c['predicted_tok_s']:.1f} vs measured {c['measured_tok_s']:.1f} tok/s"
            f" -> error {c['error_pct']:+.1f}% {'PASS' if c['pass'] else 'FAIL'} | raw ceiling error {c['raw_ceiling_error_pct']:+.1f}%"
        )
    print(f"\nE2 {'PASSED' if passed else 'FAILED'} (tolerance {100 * TOLERANCE:.0f}%). saved {path}")


if __name__ == "__main__":
    main()
