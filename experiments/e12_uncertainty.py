"""Experiment E12: error bars on the headline numbers (addition 3).

    uv run python experiments/e12_uncertainty.py

Every input that is estimated or assumed is drawn at random from its range in the parameter registers,
all at once, 5,000 times, for two designs on Llama-3.1-8B (the E4 nominal card and an aggressive design) at 512 and 4,096 tokens of context. Ranges wider than 3x are sampled log-uniformly, others uniformly.
Reported: 5th, 50th and 95th percentiles, the share of draws that still pass the proposed criteria, and the
inputs that move each result most (Spearman rank correlation). All values estimated.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, context, save  # noqa: E402
from molsim import plotstyle  # noqa: E402
from molsim.cim import precision  # noqa: E402
from molsim.labeled import Value  # noqa: E402
from molsim.llm import Model  # noqa: E402
from molsim.params import load  # noqa: E402
from molsim.scenarios import card  # noqa: E402
from molsim.sweep import evaluate  # noqa: E402

N_DRAWS = 5000
RNG = np.random.default_rng(2026)

# (name, low, high): ranges from params/cards.yaml and devices.yaml (estimated or assumed values only)
INPUTS = {
    "periph_energy_pJ": (48.4, 73.0),          # E3 calibration vs NeuRRAM
    "g_mean_fraction": (0.10, 0.50),
    "input_duty": (0.10, 0.50),
    "kv_pJ_per_bit": (2.0, 10.0),              # LPDDR5X, no primary source
    "digital_pJ_per_op": (0.1, 2.0),
    "static_W": (0.5, 3.0),
    "overhead_energy": (0.06, 0.23),
    "overhead_latency": (0.03, 0.16),
    "t_conv_ns": (2.0, 20.0),
    "column_area_fraction": (0.10, 0.40),
    "wire_ohm_per_cell": (0.1, 5.0),
}
DESIGNS = {
    "nominal (E4 card, 5-bit weights)": {"rows": 256, "share": 1, "periph_um2": (5438.0, 5438.0), "pitch_nm": (50, 300), "g": 10.0},
    "aggressive design": {"rows": 512, "share": 16, "periph_um2": (500.0, 1500.0), "pitch_nm": (50, 300), "g": 10.0},
}
CONTEXTS = [512, 4096]


def draw(lo: float, hi: float) -> float:
    if lo == hi:
        return lo
    if hi / lo > 3:
        return float(math.exp(RNG.uniform(math.log(lo), math.log(hi))))
    return float(RNG.uniform(lo, hi))


def v(x: float, unit: str) -> Value:
    return Value.of(x, unit, label="assumed", source="E12 draw", range=(x, x))


def one(model: Model, d: dict, ctx: int) -> dict:
    s = {k: draw(*r) for k, r in INPUTS.items()}
    periph = draw(*d["periph_um2"])
    pitch = draw(*d["pitch_nm"])
    c = card("molecular_scaled", rows=d["rows"], adc_share=d["share"], periph_area=periph, device_pitch=v(pitch, "nm"), g_max=d["g"],
             weight_slices=precision.molecular_slices(5, d["g"]), kv_bytes_per_value=1,
             g_mean=v(d["g"] * s["g_mean_fraction"], "uS"), input_duty=v(s["input_duty"], ""),
             periph_energy_per_output=v(s["periph_energy_pJ"], "pJ"),
             t_conv=v(s["t_conv_ns"], "ns") if d["share"] > 1 else None,
             adc_area=v(periph * (1 - s["column_area_fraction"]), "um**2"), column_area=v(periph * s["column_area_fraction"], "um**2"),
             wire_resistance=v(s["wire_ohm_per_cell"], "ohm"),
             kv_energy_per_bit=v(s["kv_pJ_per_bit"], "pJ/bit"), digital_energy_per_op=v(s["digital_pJ_per_op"], "pJ/op"),
             static_power=v(s["static_W"], "W"), overhead_energy_fraction=s["overhead_energy"], overhead_latency_fraction=s["overhead_latency"])
    x = evaluate(model, c, ctx)
    return {**s, "periph_um2": periph, "pitch_nm": pitch, "mJ_tok": x["mJ_tok"], "tok_s": x["tok_s"], "reticles": x["reticles"],
            "power_W": x["mJ_tok"] / 1000 * x["tok_s"], "ir_drop": x["ir_drop"]}


def main() -> None:
    m = Model.load("llama-3.1-8b")
    e1 = json.loads((RESULTS / "E1_baseline.json").read_text())
    gpu_mj = 1000 * e1["models"]["llama3.1:8b-instruct-q4_K_M"]["decode_J_per_token"]["median"]
    e10 = json.loads((RESULTS / "E10_breakeven.json").read_text())
    speed_limit = e10["limits"]["rtx4090_tok_s"]
    ir_limit = load("cards")["common"]["ir_drop_limit"].m
    rows = []
    for dname, d in DESIGNS.items():
        for ctx in CONTEXTS:
            for _ in range(N_DRAWS):
                rows.append({"design": dname, "context": ctx, **one(m, d, ctx)})
    df = pd.DataFrame(rows)
    df["energy_x_vs_gpu"] = gpu_mj / df.mJ_tok
    df["passes"] = (df.mJ_tok <= gpu_mj / 10) & (df.ir_drop <= ir_limit) & ((df.context != 4096) | (df.tok_s >= speed_limit))

    summary = []
    metrics = ["mJ_tok", "tok_s", "power_W", "reticles", "energy_x_vs_gpu", "ir_drop"]
    for (dname, ctx), g in df.groupby(["design", "context"]):
        row = {"design": dname, "context": int(ctx), "pass_share": float(g.passes.mean())}
        for k in metrics:
            row[k] = {p: float(np.percentile(g[k], p)) for p in (5, 50, 95)}
        drivers = {}
        for k in ("mJ_tok", "tok_s", "reticles"):
            rk = g[k].rank()
            drivers[k] = sorted(((inp, float(g[inp].rank().corr(rk))) for inp in list(INPUTS) + ["periph_um2", "pitch_nm"] if g[inp].nunique() > 1),
                                key=lambda t: -abs(t[1]))[:4]
        row["drivers"] = drivers
        summary.append(row)

    payload = {"experiment": "E12", "title": "Error bars on the headline numbers (5,000 random draws of every uncertain input)", "label": "estimated",
               "draws_per_case": N_DRAWS, "inputs": {k: list(r) for k, r in INPUTS.items()}, "designs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()} for k, d in DESIGNS.items()},
               "gpu_mJ_measured": gpu_mj, "rtx4090_tok_s_4k": speed_limit, "summary": summary, "context": context()}
    path = save("E12_uncertainty", payload)

    plotstyle.apply()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (k, lab, log) in enumerate([("mJ_tok", "energy per token (mJ)", True), ("tok_s", "tokens/s", True), ("reticles", "reticle-size dies", True)]):
        a = ax[j]
        labels, data = [], []
        for dname in DESIGNS:
            for ctx in CONTEXTS:
                if k == "reticles" and ctx != 4096:
                    continue
                labels.append(f"{'best' if 'best' in dname else 'nominal'}\n{ctx:,} tok")
                data.append(df[(df.design == dname) & (df.context == ctx)][k].to_numpy())
        parts = a.violinplot(data, showmedians=True, showextrema=False)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(plotstyle.SERIES[1] if "best" in labels[i] else plotstyle.SERIES[0])
            b.set_alpha(0.6)
        a.set_xticks(range(1, len(labels) + 1), labels)
        if log:
            a.set_yscale("log")
        if k == "mJ_tok":
            a.axhline(gpu_mj / 10, color=plotstyle.INK2, linestyle="--", linewidth=1, label="pass limit (10x below measured GPU)")
            a.legend(fontsize=7)
        if k == "tok_s":
            a.axhline(speed_limit, color=plotstyle.INK2, linestyle="--", linewidth=1, label="RTX 4090 model, 4k context")
            a.legend(fontsize=7)
        a.set(title=lab, ylabel=lab)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}" if val >= 10 else f"{val:g}"))
    fig.suptitle("E12: spread of the headline numbers over 5,000 random draws of every uncertain input (Llama-3.1-8B, estimated)", fontsize=9, color=plotstyle.INK2)
    fig.tight_layout()
    fig_path = plotstyle.save(fig, "E12_uncertainty")

    for r in summary:
        print(f"{r['design']:36} ctx {r['context']:>5}: pass {100 * r['pass_share']:5.1f}%  "
              + "  ".join(f"{k} {r[k][5]:,.3g}/{r[k][50]:,.3g}/{r[k][95]:,.3g}" for k in ("mJ_tok", "tok_s", "reticles", "energy_x_vs_gpu", "power_W")))
        for k, dr in r["drivers"].items():
            print(f"      {k:9} driven by " + ", ".join(f"{n} ({c:+.2f})" for n, c in dr))
    print(f"saved {path}\nsaved {fig_path}")


if __name__ == "__main__":
    main()
