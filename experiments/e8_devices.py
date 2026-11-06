"""Experiment E8: molecular vs PCM vs RRAM at equal converter cost (H6).

    uv run python experiments/e8_devices.py

Every card shares the same converters, digital core, KV memory and link (params/cards.yaml), so the
device decides only three things: read energy (window x voltage), how many column slices a weight
needs (levels, molecular/cim/precision.py) and array area (pitch, stacked or 1T1R).
The molecular conductance window is swept from 1 uS to the demonstrated 6 mS under both level rules,
for 4-bit and 8-bit weights. Design points whose IR drop exceeds the limit are marked infeasible.
Llama-3.1-8B, context 4,096 tokens. Win / tie / lose: tie within 10%. All values estimated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from common import context, save  # noqa: E402
from molsim import plotstyle  # noqa: E402
from molsim.cim import precision  # noqa: E402
from molsim.llm import Model  # noqa: E402
from molsim.params import load  # noqa: E402
from molsim.scenarios import card  # noqa: E402
from molsim.sweep import evaluate  # noqa: E402

CTX = 4096
WINDOWS_US = [1, 3, 10, 30, 50, 100, 300, 1000, 3000, 6000]
TIE = 0.10


def verdict(mol: float, best_other: float) -> str:
    r = mol / best_other
    return "win" if r < 1 - TIE else ("lose" if r > 1 + TIE else "tie")


def main() -> None:
    m = Model.load("llama-3.1-8b")
    ir_limit = load("cards")["common"]["ir_drop_limit"].m
    rows = []
    for wb in (4, 8):
        for dev, label in [("pcm", "PCM"), ("rram", "RRAM")]:
            for opt in (False, True):
                s = precision.slices(dev, wb, opt)
                x = evaluate(m, card(dev, weight_slices=s), CTX)
                rows.append({"weight_bits": wb, "device": label, "variant": "optimistic" if opt else "nominal",
                             "g_max_uS": load("devices")[dev]["g_max"].q.to("uS").magnitude, "slices": s, **x})
        for rule in ("constant_spacing", "proportional"):
            for g in WINDOWS_US:
                s = precision.molecular_slices(wb, g, rule)
                x = evaluate(m, card("molecular_scaled", g_max=g, weight_slices=s), CTX)
                rows.append({"weight_bits": wb, "device": "Molecular", "variant": rule, "g_max_uS": g,
                             "levels": precision.molecular_levels(g, rule), "slices": s, **x})
    df = pd.DataFrame(rows)
    df["feasible"] = df["ir_drop"] <= ir_limit

    # verdicts: the best feasible molecular design (under each level rule) vs the best nominal PCM or RRAM design
    verdicts = []
    for wb in (4, 8):
        other = df[(df.weight_bits == wb) & (df.device != "Molecular") & (df.variant == "nominal")]
        other_opt = df[(df.weight_bits == wb) & (df.device != "Molecular")]
        for rule in ("constant_spacing", "proportional"):
            mol = df[(df.weight_bits == wb) & (df.device == "Molecular") & (df.variant == rule) & df.feasible]
            best = mol.loc[mol.mJ_tok.idxmin()]
            v = {"weight_bits": wb, "rule": rule, "molecular_window_uS": best.g_max_uS, "molecular_slices": int(best.slices)}
            for metric in ("mJ_tok", "area_mm2", "install_s"):
                v[metric] = {
                    "molecular": float(best[metric]), "best_other": float(other[metric].min()),
                    "best_other_device": other.loc[other[metric].idxmin(), "device"],
                    "vs_nominal": verdict(best[metric], other[metric].min()),
                    "vs_optimistic": verdict(best[metric], other_opt[metric].min()),
                }
            verdicts.append(v)

    payload = {"experiment": "E8", "title": "Molecular vs PCM vs RRAM at equal converter cost", "label": "estimated",
               "model": "llama-3.1-8b", "context_tokens": CTX, "ir_drop_limit": ir_limit, "tie_band": TIE,
               "context": context(), "rows": df.to_dict("records"), "verdicts": verdicts}
    path = save("E8_devices", payload)

    # ── figure: energy and area against the molecular window, PCM and RRAM as reference lines ──
    plotstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, wb in enumerate((4, 8)):
        for row, (metric, ylab) in enumerate([("mJ_tok", "energy per token (mJ)"), ("area_mm2", "silicon area (mm2)")]):
            ax = axes[row][col]
            for i, rule in enumerate(("constant_spacing", "proportional")):
                d = df[(df.weight_bits == wb) & (df.device == "Molecular") & (df.variant == rule)]
                ax.plot(d.g_max_uS, d[metric], marker="o", color=plotstyle.SERIES[i], label=f"molecular, {rule.replace('_', ' ')}")
                bad = d[~d.feasible]
                ax.scatter(bad.g_max_uS, bad[metric], s=60, facecolors="white", edgecolors=plotstyle.SERIES[i], zorder=3)
            for j, dev in enumerate(("PCM", "RRAM")):
                y = df[(df.weight_bits == wb) & (df.device == dev) & (df.variant == "nominal")][metric].iloc[0]
                ax.axhline(y, color=plotstyle.SERIES[2 + j], linewidth=1.5, linestyle="--", label=f"{dev} (nominal)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if str(v)[0] in "235" else ""))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{wb}-bit weights")
            if row == 1:
                ax.set_xlabel("molecular conductance window, g_max (uS)")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("E8: device comparison at equal converter cost (Llama-3.1-8B, 4k context, estimated). Open markers: IR drop above limit", fontsize=9, color=plotstyle.INK2)
    fig_path = plotstyle.save(fig, "E8_devices")

    # ── console ──
    print(f"Llama-3.1-8B, context {CTX}, shared periphery. IR-drop limit {ir_limit:.0%}.\n")
    show = ["weight_bits", "device", "variant", "g_max_uS", "slices", "mJ_tok", "array_mJ", "area_mm2", "reticles", "ir_drop", "install_s", "feasible"]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:,.3g}".format):
        print(df[show].to_string(index=False))
    print("\nverdicts (best feasible molecular vs best nominal PCM/RRAM; vs optimistic precision in brackets):")
    for v in verdicts:
        parts = ", ".join(f"{k} {v[k]['vs_nominal']} ({v[k]['vs_optimistic']}) {v[k]['molecular']:.3g} vs {v[k]['best_other']:.3g}" for k in ("mJ_tok", "area_mm2", "install_s"))
        print(f"  {v['weight_bits']}-bit, {v['rule']}: window {v['molecular_window_uS']} uS, {v['molecular_slices']} slice(s): {parts}")
    print(f"\nsaved {path}\nsaved {fig_path}")


if __name__ == "__main__":
    main()
