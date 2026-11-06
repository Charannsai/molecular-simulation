"""Experiment E4: main comparison on Llama-3.1-8B (hypotheses H1, H2, H3, H4, H8).

    uv run python experiments/e4_comparison.py

Context = the average context of the E1 measurement, so the GPU numbers are like-for-like.
Baselines:
  - RTX 5050 Laptop: MEASURED (E1, E2).
  - Memory-offload design: weights streamed over PCIe 5.0 x16 every token (the design H1 is defined against).
  - RTX 4090 and Ryzen AI Max+ 395: digital model with published bandwidth and the E2 utilisation (estimated).
CIM cards: four device scenarios (params/cards.yaml), attention on the card; plus molecular-scaled with
attention on the GPU. All CIM numbers are model estimates; the GPU numbers are measurements.
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
from molsim.scenarios import SCENARIOS, card  # noqa: E402

TAG = "llama3.1:8b-instruct-q4_K_M"


def main() -> None:
    e1 = json.loads((RESULTS / "E1_baseline.json").read_text())
    e2 = json.loads((RESULTS / "E2_digital.json").read_text())
    systems, circuits, chips = load("systems"), load("circuits"), load("chips")
    core_mm2 = chips["hermes"]["core_area"].q.to("mm**2").magnitude  # one 256 x 256 tile with its 256 ADCs (published, verify)
    reticle = circuits["fab"]["reticle_area"].q.to("mm**2").magnitude
    m = Model.load("llama-3.1-8b")
    meas = e1["models"][TAG]
    ctx = int(meas["prompt_tokens"]["median"] + meas["generated_tokens"]["median"] / 2)
    util = sum(x["utilisation"] for x in e2["models"].values()) / len(e2["models"])
    gpu_bytes = e2["models"][TAG]["weight_GB_per_token"] + e2["models"][TAG]["kv_GB_per_token"]  # GB per token on the GPU

    # ── baselines ──
    gpu = {
        "name": "RTX 5050 Laptop (measured, 4-bit)", "label": "measured",
        "tok_s": meas["decode_tok_s"]["median"], "mJ_tok": 1000 * meas["decode_J_per_token"]["median"],
        "weight_GB_moved": e2["models"][TAG]["weight_GB_per_token"], "link_bytes": None,
        "gpu_memory_GB": meas["vram_MiB"]["median"] * 1.048576 / 1000, "install_s": None,
    }
    pcie = circuits["link"]["pcie5_x16_bandwidth"].q.to("GB/s").magnitude
    w8_gb = m.streamed_params() / 1e9  # 8-bit weights, 1 byte each
    offload = {
        "name": "Memory offload over PCIe 5.0 x16 (8-bit weights)", "label": "estimated",
        "tok_s": pcie / w8_gb, "mJ_tok": None, "weight_GB_moved": w8_gb, "link_bytes": w8_gb * 1e9,
        "gpu_memory_GB": None, "install_s": None,
    }
    est = []
    for key, name in [("rtx4090", "RTX 4090 (digital model, 4-bit)"), ("ryzen_ai_max_395", "Ryzen AI Max+ 395 (digital model, 4-bit)")]:
        bw = systems[key]["memory_bandwidth"].q.to("GB/s").magnitude
        tok = bw * util / gpu_bytes
        row = {"name": name, "label": "estimated", "tok_s": tok, "mJ_tok": None, "weight_GB_moved": e2["models"][TAG]["weight_GB_per_token"], "link_bytes": None, "gpu_memory_GB": None, "install_s": None}
        if key == "ryzen_ai_max_395":
            row["mJ_tok"] = 1000 * systems[key]["package_power"].q.to("W").magnitude / tok
            row["note"] = "energy is an upper bound: full 120 W package power"
        est.append(row)

    # ── CIM cards ──
    cim = []
    for s, attn in [(x, "card") for x in SCENARIOS] + [("molecular_scaled", "gpu")]:
        c = card(s, attention_on=attn)
        r = token_costs(m, c, ctx)
        kv_gb = m.kv_bytes_per_token(c.kv_bytes_per_value) * ctx / 1e9
        weight_ops = 2 * m.streamed_params()
        attn_ops = 4 * m.num_layers * m.num_heads * m.head_dim * ctx
        cim.append({
            "name": f"{c.tile.name}" + (", attention on GPU" if attn == "gpu" else ""), "scenario": s, "attention_on": attn, "label": "estimated",
            "tok_s": r["tokens_per_s"].m, "mJ_tok": r["energy"].m, "energy_parts_mJ": r["energy_parts_mJ"], "latency_parts_ms": r["latency_parts_ms"],
            "weight_GB_moved": 0.0, "link_bytes": r["link_bytes_per_token"],
            "gpu_memory_GB": kv_gb if attn == "gpu" else 0.0,
            "offload_fraction": weight_ops / (weight_ops + attn_ops) if attn == "gpu" else 1.0,
            "install_s": r["install_time_s"], "tiles": r["mapping"]["tiles"], "devices": r["mapping"]["devices"],
            # area if every tile needs HERMES-class periphery (estimated); E9 refines this
            "area_mm2_hermes_class": r["mapping"]["tiles"] * core_mm2, "reticles": r["mapping"]["tiles"] * core_mm2 / reticle,
        })

    rows = [gpu, offload, *est, *cim]
    for r in rows:
        r["energy_vs_gpu"] = (gpu["mJ_tok"] / r["mJ_tok"]) if r.get("mJ_tok") else None
        r["speed_vs_gpu"] = r["tok_s"] / gpu["tok_s"]

    payload = {
        "experiment": "E4", "title": "Main comparison on Llama-3.1-8B: measured GPU vs molecular, PCM and RRAM CIM cards",
        "label": "GPU measured; everything else estimated", "context_tokens": ctx, "e2_utilisation": util,
        "context": context(), "rows": rows,
    }
    path = save("E4_comparison", payload)

    print(f"Llama-3.1-8B, context {ctx} tokens (E1 average), batch 1\n")
    print(f"{'system':62} {'tok/s':>8} {'mJ/token':>10} {'energy x':>9} {'weights moved':>14} {'link/token':>11} {'GPU mem':>8} {'install':>9}")
    def fmt(x, spec, suffix=""):
        return "-" if x is None else f"{x:{spec}}{suffix}"

    for r in rows:
        lb = r["link_bytes"]
        link = "-" if lb is None else (f"{lb / 1e9:.1f} GB" if lb > 1e8 else f"{lb / 1e3:.0f} KB")
        print(
            f"{r['name'][:62]:62} {r['tok_s']:8.1f} {fmt(r['mJ_tok'], '.2f'):>10} {fmt(r['energy_vs_gpu'], '.3g', 'x'):>9}"
            f" {r['weight_GB_moved']:11.2f} GB {link:>11} {fmt(r['gpu_memory_GB'], '.2f', ' GB'):>8} {fmt(r['install_s'], '.1f', ' s'):>9}"
        )
    c0 = cim[0]
    print(
        f"\nsilicon if every tile needs HERMES-class periphery ({core_mm2} mm2 per 256 x 256 tile, published): "
        f"{c0['tiles']:,} tiles = {c0['area_mm2_hermes_class']:,.0f} mm2 = {c0['reticles']:.0f} reticle-size dies ({reticle:.0f} mm2 each)"
    )
    print("\nenergy breakdown (mJ/token) for the CIM cards:")
    for r in cim:
        parts = ", ".join(f"{k} {v:.2f}" for k, v in r["energy_parts_mJ"].items() if v)
        print(f"  {r['name'][:60]:60} {parts}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
