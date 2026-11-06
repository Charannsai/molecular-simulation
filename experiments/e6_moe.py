"""Experiment E6: dense vs mixture-of-experts models on the CIM card (H3, H4; MoE idle capacity).

    uv run python experiments/e6_moe.py

On a CIM card every weight must be installed (it occupies devices and their converters), but an
MoE token uses only a few experts. On a unified-memory machine (Ryzen AI Max+ 395) idle experts
cost nothing but cheap DRAM capacity, and only active weights are read.
Context 4,096 tokens. Card: molecular-scaled scenario, attention on the card. Ryzen: digital model
with published bandwidth, E2 utilisation and the measured 4-bit format (E2 bits per weight).
All values estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import RESULTS, context, save  # noqa: E402
from molsim.cim.system import token_costs  # noqa: E402
from molsim.llm import Model  # noqa: E402
from molsim.params import load  # noqa: E402
from molsim.scenarios import card  # noqa: E402

CTX = 4096
MODELS = ["llama-3.1-8b", "qwen2.5-32b", "qwen3-30b-a3b", "gpt-oss-120b"]


def main() -> None:
    e2 = json.loads((RESULTS / "E2_digital.json").read_text())
    util = sum(x["utilisation"] for x in e2["models"].values()) / len(e2["models"])
    bpp4 = e2["models"]["llama3.1:8b-instruct-q4_K_M"]["bytes_per_param"]  # measured 4-bit format, bytes per weight
    ryzen = load("systems")["ryzen_ai_max_395"]
    r_bw = ryzen["memory_bandwidth"].q.to("GB/s").magnitude
    r_mem = ryzen["memory"].q.to("GB").magnitude
    core_mm2 = load("chips")["hermes"]["core_area"].q.to("mm**2").magnitude

    rows = []
    for key in MODELS:
        m = Model.load(key)
        r = token_costs(m, card("molecular_scaled"), CTX)
        mp = r["mapping"]
        kv_gb = m.kv_bytes_per_token(2) * CTX / 1e9
        r_bytes = m.streamed_params() * bpp4 / 1e9 + kv_gb
        rows.append({
            "model": key, "moe": m.is_moe, "total_B": m.total_params() / 1e9, "active_B": m.streamed_params() / 1e9,
            "tiles_installed": mp["tiles"], "tiles_active": mp["active_tiles"], "idle_fraction": 1 - mp["active_tiles"] / mp["tiles"],
            "area_mm2_hermes_class": mp["tiles"] * core_mm2,
            "card_tok_s": r["tokens_per_s"].m, "card_mJ_tok": r["energy"].m,
            "ryzen_tok_s": r_bw * util / r_bytes, "ryzen_fits": m.total_params() * bpp4 / 1e9 + kv_gb <= r_mem,
            "area_per_active_B": mp["tiles"] * core_mm2 / (m.streamed_params() / 1e9),
        })

    payload = {"experiment": "E6", "title": "Dense vs mixture-of-experts on the CIM card", "label": "estimated", "context_tokens": CTX, "context": context(), "rows": rows}
    path = save("E6_moe", payload)

    print(f"Context {CTX}. Card: molecular-scaled, attention on card. Ryzen AI Max+ 395: digital model, 4-bit.\n")
    print(f"{'model':15} {'total':>7} {'active':>7} {'tiles':>9} {'idle':>6} {'silicon mm2':>12} {'mm2/active B':>13} | {'card tok/s':>10} {'mJ/tok':>7} | {'Ryzen tok/s':>11} {'fits':>5}")
    for r in rows:
        print(
            f"{r['model']:15} {r['total_B']:6.1f}B {r['active_B']:6.2f}B {r['tiles_installed']:9,} {100 * r['idle_fraction']:5.0f}% {r['area_mm2_hermes_class']:12,.0f} {r['area_per_active_B']:13,.0f}"
            f" | {r['card_tok_s']:10.0f} {r['card_mJ_tok']:7.1f} | {r['ryzen_tok_s']:11.1f} {'yes' if r['ryzen_fits'] else 'NO':>5}"
        )
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
