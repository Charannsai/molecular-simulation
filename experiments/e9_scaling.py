"""Experiment E9: scaling 8B, 32B and 70B (H7).

    uv run python experiments/e9_scaling.py      (after e10_breakeven.py)

Two card designs, each with 8-bit and 4-bit weights:
  - nominal: the E4 card (HERMES-class periphery, one converter per column, 256 rows, 10 uS window);
  - aggressive: the smallest passing 8B design from E10 (its rows, sharing, periphery area, pitch, window).
For each model: tiles, silicon, reticle-size dies, energy and speed at 4,096 tokens, against the RTX 4090
and Ryzen AI Max+ 395 digital models (4-bit, E2 utilisation) and whether the model fits their memory.
All values estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import RESULTS, context, save  # noqa: E402
from molsim.cim import precision  # noqa: E402
from molsim.labeled import Value  # noqa: E402
from molsim.llm import Model  # noqa: E402
from molsim.params import load  # noqa: E402
from molsim.scenarios import card  # noqa: E402
from molsim.sweep import evaluate  # noqa: E402

CTX = 4096
MODELS = ["llama-3.1-8b", "qwen2.5-32b", "llama-3.1-70b"]
TAG = "llama3.1:8b-instruct-q4_K_M"


def designs() -> dict:
    e10 = json.loads((RESULTS / "E10_breakeven.json").read_text())
    nom = e10["nominal"]
    out = {}
    for wb in (8, 4):
        best = min((d for d in e10["budgets"][str(wb)].values() if d), key=lambda d: d["reticles"])
        out[f"nominal, {wb}-bit weights"] = {"rows": nom["rows"], "adc_share": nom["adc_share"], "periph_area": nom["periph_area"],
                                             "pitch_nm": nom["pitch_nm"], "g_max_uS": nom["g_max_uS"], "output_bits": nom["output_bits"], "weight_bits": wb}
        # E10's least-demanding design inside the smallest budget it could meet
        out[f"aggressive, {wb}-bit weights"] = {k: best[k] for k in ("rows", "adc_share", "periph_area", "pitch_nm", "g_max_uS", "output_bits")} | {"weight_bits": wb}
    return out


def build(p: dict):
    pitch = Value.of(p["pitch_nm"], "nm", label="assumed", source="E10 design point", range=(p["pitch_nm"], p["pitch_nm"]))
    return card("molecular_scaled", rows=int(p["rows"]), adc_share=int(p["adc_share"]), periph_area=float(p["periph_area"]), device_pitch=pitch,
                g_max=float(p["g_max_uS"]), output_bits=int(p["output_bits"]), weight_slices=precision.molecular_slices(int(p["weight_bits"]), float(p["g_max_uS"])))


def main() -> None:
    e2 = json.loads((RESULTS / "E2_digital.json").read_text())
    util = sum(x["utilisation"] for x in e2["models"].values()) / len(e2["models"])
    bpp4 = e2["models"][TAG]["bytes_per_param"]
    systems = load("systems")
    digital = {
        "RTX 4090": (systems["rtx4090"]["memory_bandwidth"].q.to("GB/s").magnitude, systems["rtx4090"]["vram"].q.to("GB").magnitude),
        "Ryzen AI Max+ 395": (systems["ryzen_ai_max_395"]["memory_bandwidth"].q.to("GB/s").magnitude, systems["ryzen_ai_max_395"]["memory"].q.to("GB").magnitude),
    }
    ds = designs()
    rows = []
    for key in MODELS:
        m = Model.load(key)
        kv_gb = m.kv_bytes_per_token(2) * CTX / 1e9
        w_gb = m.streamed_params() * bpp4 / 1e9
        base = {name: {"tok_s": bw * util / (w_gb + kv_gb), "fits": m.total_params() * bpp4 / 1e9 + kv_gb <= mem} for name, (bw, mem) in digital.items()}
        for dname, p in ds.items():
            x = evaluate(m, build(p), CTX)
            rows.append({"model": key, "params_B": m.total_params() / 1e9, "design": dname, **p, **x, "digital": base})

    payload = {"experiment": "E9", "title": "Scaling 8B, 32B and 70B on the CIM card", "label": "estimated", "context_tokens": CTX,
               "designs": ds, "context": context(), "rows": rows}
    path = save("E9_scaling", payload)

    print(f"Context {CTX}. Molecular card designs from E10. Digital: 4-bit weights, E2 utilisation.\n")
    print(f"{'model':14} {'design':26} {'tiles':>10} {'mm2':>11} {'dies':>7} {'mJ/tok':>7} {'tok/s':>6} | {'4090 tok/s':>10} {'fits':>4} | {'Ryzen tok/s':>11} {'fits':>4}")
    for r in rows:
        g, z = r["digital"]["RTX 4090"], r["digital"]["Ryzen AI Max+ 395"]
        print(f"{r['model']:14} {r['design']:26} {r['tiles']:10,} {r['area_mm2']:11,.0f} {r['reticles']:7.1f} {r['mJ_tok']:7.1f} {r['tok_s']:6.0f} | "
              f"{g['tok_s']:10.1f} {'yes' if g['fits'] else 'no':>4} | {z['tok_s']:11.1f} {'yes' if z['fits'] else 'no':>4}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
