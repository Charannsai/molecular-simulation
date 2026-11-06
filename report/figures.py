"""Figures for the experiments that only had tables (E1 to E6, E9, E7 training), in the shared style.

    uv run python report/figures.py      -> results/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from molsim import plotstyle as ps  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
FMT = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}")


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))


def label_bars(ax, bars, fmt="{:,.1f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8, color=ps.INK2, xytext=(0, 2), textcoords="offset points")


def e1_e2() -> None:
    e1, e2 = load("E1_baseline"), load("E2_digital")
    ms = list(e1["models"].values())
    names = [f"{m['model']}\n{m['quantisation'].split(' ')[0]}" for m in ms]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    b = ax[0].bar(names, [m["decode_tok_s"]["median"] for m in ms], color=ps.SERIES[0], width=0.5)
    label_bars(ax[0], b)
    ax[0].set(title="Decode speed (measured)", ylabel="tokens/s")
    b = ax[1].bar(names, [m["decode_J_per_token"]["median"] for m in ms], color=ps.SERIES[1], width=0.5)
    label_bars(ax[1], b, "{:.3f}")
    ax[1].set(title="Energy per token (measured, board)", ylabel="J per token")
    cv = e2["cross_validation"]
    x = np.arange(len(cv))
    w = 0.25
    for i, (k, lab) in enumerate([("raw_ceiling_tok_s", "roofline ceiling"), ("predicted_tok_s", "model (cross-calibrated)"), ("measured_tok_s", "measured")]):
        b = ax[2].bar(x + (i - 1) * w, [c[k] for c in cv], w, color=ps.SERIES[i], label=lab)
        label_bars(ax[2], b)
    ax[2].set_xticks(x, [c["target"] for c in cv])
    ax[2].set(title="E2: digital model vs measurement", ylabel="tokens/s")
    ax[2].legend(fontsize=7, loc="upper right")
    fig.suptitle("E1 and E2: RTX 5050 Laptop GPU, Ollama, batch 1 (medians of 5 runs)", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    ps.save(fig, "E1_E2_baseline")


def e3() -> None:
    d = load("E3_cim_validation")
    c = d["checks"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = ["V1 HERMES\n1-phase latency", "V2 HERMES\n4-phase latency", "V3 HERMES\n4-phase energy", "V4 NeuRRAM\n8-bit MVM energy"]
    err = [k["error_pct"] for k in c]
    bars = ax.bar(names, err, color=[ps.SERIES[2] if k["pass"] else ps.SERIES[7] for k in c], width=0.5)
    for b, k in zip(bars, c):
        ax.annotate(f"{k['error_pct']:+.1f}%\n{k['pred']:.3g} vs {k['actual']:.3g} {k['unit']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=7, color=ps.INK2, xytext=(0, 3 if b.get_height() >= 0 else -3), textcoords="offset points")
    ax.axhspan(-30, 30, color=ps.GRID, zorder=0)
    ax.axhline(0, color=ps.INK2, linewidth=0.8)
    ax.set(ylabel="model error vs published (%)", ylim=(-50, 25), title="E3: CIM model vs IBM HERMES and NeuRRAM (grey: 30% gate; red: failed)")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    ps.save(fig, "E3_validation")


def e4() -> None:
    d = load("E4_comparison")
    rows = d["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    sys_rows = [r for r in rows if r.get("mJ_tok")]
    names = [r["name"].replace(" (", "\n(").replace(", attention", "\nattention")[:70] for r in sys_rows]
    b = ax[0].barh(names, [r["mJ_tok"] for r in sys_rows], color=[ps.SERIES[0] if r["label"] == "measured" else ps.SERIES[1] for r in sys_rows])
    for bar, r in zip(b, sys_rows):
        ax[0].annotate(f"{r['mJ_tok']:,.1f} mJ", (bar.get_width(), bar.get_y() + bar.get_height() / 2), va="center", fontsize=7, color=ps.INK2, xytext=(3, 0), textcoords="offset points")
    ax[0].set_xscale("log")
    ax[0].xaxis.set_major_formatter(FMT)
    ax[0].invert_yaxis()
    ax[0].tick_params(axis="y", labelsize=7)
    ax[0].set(xlabel="mJ per token (log)", title="Energy per token, Llama-3.1-8B, 457-token context")
    cim = [r for r in rows if r.get("energy_parts_mJ")]
    parts = ["tile_array", "tile_periphery", "onchip_overhead", "digital", "kv_cache", "link", "static", "host"]
    left = np.zeros(len(cim))
    short = ["Molecular, demonstrated (6 mS)", "Molecular, scaled (10 uS)", "PCM (HERMES-like)", "RRAM (NeuRRAM-like)", "Molecular scaled, attention on GPU"]
    for i, p in enumerate(parts):
        vals = np.array([r["energy_parts_mJ"][p] for r in cim])
        if vals.sum() == 0:
            continue
        ax[1].barh(short, vals, left=left, color=ps.SERIES[i % 8], label=p.replace("_", " "), edgecolor="white", linewidth=1)
        left += vals
    ax[1].invert_yaxis()
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set(xlabel="mJ per token", title="Where the card's energy goes (estimated)")
    ax[1].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("E4: blue = measured GPU, orange = estimated (model). Ryzen bar is an upper bound at full 120 W package power.", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    ps.save(fig, "E4_comparison")


def e5() -> None:
    d = load("E5_context")["rows"]
    ctx = [r["context"] for r in d]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, (k, lab) in enumerate([("cim_fp16", "card, FP16 KV cache"), ("cim_int8", "card, 8-bit KV cache")]):
        ax[0].plot(ctx, [r[k]["tok_s"] for r in d], marker="o", color=ps.SERIES[i], label=lab)
        ax[1].plot(ctx, [r[k]["mJ_tok"] for r in d], marker="o", color=ps.SERIES[i], label=lab)
    ax[0].plot(ctx, [r["gpu_fp16"]["tok_s"] for r in d], marker="o", color=ps.SERIES[2], label="RTX 5050 Laptop (model)")
    for r in d:
        if not r["gpu_fp16"]["fits_in_vram"]:
            ax[0].scatter([r["context"]], [r["gpu_fp16"]["tok_s"]], s=70, facecolors="white", edgecolors=ps.SERIES[2], zorder=3)
    ax[1].axhline(1052, color=ps.INK2, linestyle="--", linewidth=1, label="GPU measured at 457 tokens")
    ax[2].plot(ctx, [100 * r["offload_fraction_gpu_attention"] for r in d], marker="o", color=ps.SERIES[3])
    for a, t, y in [(ax[0], "Speed vs context", "tokens/s"), (ax[1], "Energy vs context", "mJ per token"), (ax[2], "H3: share of work on the card (attention on GPU)", "% of per-token operations")]:
        a.set_xscale("log", base=2)
        a.set(xlabel="context (tokens)", ylabel=y, title=t)
        a.set_xticks(ctx, [f"{c:,}" for c in ctx])
        a.xaxis.set_minor_locator(plt.NullLocator())
        if a is not ax[2]:
            a.set_yscale("log")
            a.yaxis.set_major_formatter(FMT)
            a.legend(fontsize=7)
    fig.suptitle("E5: Llama-3.1-8B, molecular-scaled card, KV cache on 136 GB/s LPDDR5X (estimated). Open markers: does not fit in 8 GB.", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    ps.save(fig, "E5_context")


def e6() -> None:
    d = load("E6_moe")["rows"]
    names = [r["model"] for r in d]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    b = ax[0].bar(names, [100 * r["idle_fraction"] for r in d], color=ps.SERIES[1], width=0.55)
    label_bars(ax[0], b, "{:.0f}%")
    ax[0].set(title="Installed tiles idle per token", ylabel="%")
    b = ax[1].bar(names, [r["area_mm2_hermes_class"] / 858 for r in d], color=ps.SERIES[0], width=0.55)
    label_bars(ax[1], b, "{:,.0f}")
    ax[1].set(title="Silicon (HERMES-class periphery)", ylabel="reticle-size dies")
    x = np.arange(len(d))
    for i, (k, lab) in enumerate([("card_tok_s", "CIM card"), ("ryzen_tok_s", "Ryzen AI Max+ 395 (model)")]):
        b = ax[2].bar(x + (i - 0.5) * 0.35, [r[k] for r in d], 0.35, color=ps.SERIES[i * 2], label=lab)
        label_bars(ax[2], b, "{:,.0f}")
    ax[2].set_xticks(x, names)
    ax[2].set(title="Speed at 4k context", ylabel="tokens/s")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.tick_params(axis="x", labelsize=7)
    fig.suptitle("E6: dense vs mixture-of-experts models on the CIM card (estimated)", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    ps.save(fig, "E6_moe")


def e9() -> None:
    d = load("E9_scaling")["rows"]
    models = list(dict.fromkeys(r["model"] for r in d))
    designs = list(dict.fromkeys(r["design"] for r in d))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(models))
    w = 0.2
    for i, ds in enumerate(designs):
        vals = [next(r["reticles"] for r in d if r["model"] == m and r["design"] == ds) for m in models]
        b = ax[0].bar(x + (i - 1.5) * w, vals, w, color=ps.SERIES[i], label=ds)
        label_bars(ax[0], b, "{:,.0f}")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(FMT)
    ax[0].set_xticks(x, models)
    ax[0].set(ylabel="reticle-size dies (log)", title="Silicon needed")
    ax[0].legend(fontsize=7)
    ag = [next(r for r in d if r["model"] == m and r["design"] == "aggressive, 4-bit weights") for m in models]
    b1 = ax[1].bar(x - 0.27, [r["tok_s"] for r in ag], 0.27, color=ps.SERIES[0], label="CIM card (aggressive, 4-bit)")
    b2 = ax[1].bar(x, [r["digital"]["RTX 4090"]["tok_s"] for r in ag], 0.27, color=ps.SERIES[1], label="RTX 4090 (model; 70B does not fit)")
    b3 = ax[1].bar(x + 0.27, [r["digital"]["Ryzen AI Max+ 395"]["tok_s"] for r in ag], 0.27, color=ps.SERIES[2], label="Ryzen AI Max+ 395 (model)")
    for b in (b1, b2, b3):
        label_bars(ax[1], b)
    ax[1].set_xticks(x, models)
    ax[1].set(ylabel="tokens/s", title="Speed at 4k context")
    ax[1].legend(fontsize=7)
    fig.suptitle("E9: scaling 8B, 32B, 70B (estimated)", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    ps.save(fig, "E9_scaling")


def e7_training() -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, m in enumerate(("digital", "hwa")):
        d = load(f"E7_train_{m}")
        ax.plot([x["step"] for x in d["log"]], [x["test_ppl_quick"] for x in d["log"]], marker="o", color=ps.SERIES[i],
                label="digital fine-tune (no noise)" if m == "digital" else "noise-aware fine-tune (measured on noisy analog model)")
    ax.set(xlabel="training step (8 x 512 tokens)", ylabel="perplexity (16 test windows)", title="E7 fine-tuning of GPT-2 small on WikiText-2")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FMT)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (25, 30, 35) else ""))
    ax.legend(fontsize=7)
    fig.tight_layout()
    ps.save(fig, "E7_training")


if __name__ == "__main__":
    ps.apply()
    for f in (e1_e2, e3, e4, e5, e6, e9, e7_training):
        f()
        print("made", f.__name__)
