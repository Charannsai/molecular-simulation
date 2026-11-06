"""Figures for E11 (task accuracy on the emulated chip) and E13 (drift tolerance).

    uv run python report/figures_e11.py   -> results/figures/E11_accuracy.png, E11_bits.png, E13_drift.png

All values are emulated or simulated (estimated); they are read from results/E11_tasks/*.json and results/E13_drift.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from molsim import plotstyle as ps  # noqa: E402

T = Path(__file__).resolve().parent.parent / "results" / "E11_tasks"


def tasks(name: str) -> dict:
    return json.loads((T / name).read_text(encoding="utf-8"))["tasks"]


def a(t: dict, k: str) -> float:
    return 100 * t[k]["accuracy"]


def p(t: dict) -> float:
    return t["wikitext"]["perplexity"]


def fig_accuracy() -> None:
    ps.apply()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.2, 1.25, 1]})
    # (a) diagnostics, perplexity
    names = ["digital_d200", "chip_digitalhead_s0_R_full", "chip_digitalhead_s0_A_circuits_only", "chip_digitalhead_s0_B_molecules_only", "chip_digitalhead_s0_C_tile_scaling"]
    labels = ["Digital", "Original\ndesign", "Ideal\ndevices", "Ideal\nconverters", "Group\nscaling"]
    vals = [p(tasks(f"Qwen2.5-0.5B-Instruct_{n}.json")) for n in names]
    cols = [ps.MUTED, ps.SERIES[1], ps.SERIES[1], ps.SERIES[0], ps.SERIES[2]]
    b = ax[0].bar(labels, vals, color=cols, width=0.62)
    ax[0].bar_label(b, fmt="%.1f", fontsize=8, padding=2)
    ax[0].set_ylabel("WikiText-2 perplexity (lower is better)")
    ax[0].set_ylim(0, 28)
    ax[0].set_title("(a) What causes the loss (Qwen2.5-0.5B)")
    # (b) model size: accuracy relative to digital
    sizes = ["0.5B", "1.5B", "3B"]
    runs = {"0.5B": ("digital_d200", "chip_digitalhead_s0_R_full", "chip_digitalhead_s0_C_tile_scaling"),
            "1.5B": ("digital_Q15_digital", "chip_digitalhead_s0_Q15_R", "chip_digitalhead_s0_Q15_C"),
            "3B": ("digital_colab", "chip_digitalhead_s0_colab_R", "chip_digitalhead_s0_colab_C")}
    x = np.arange(len(sizes))
    for i, (lab, col) in enumerate([("Original design", ps.SERIES[1]), ("Group scaling", ps.SERIES[2])]):
        gaps = []
        for s in sizes:
            d, r, c = (tasks(f"Qwen2.5-{s}-Instruct_{n}.json") for n in runs[s])
            t = (r, c)[i]
            gaps.append(np.mean([a(t, k) - a(d, k) for k in ("arc_easy", "mmlu")]))
        bb = ax[1].bar(x + (i - 0.5) * 0.36, gaps, width=0.36, color=col, label=lab)
        ax[1].bar_label(bb, fmt="%.1f", fontsize=8, padding=2)
    ax[1].axhline(0, color=ps.INK2, lw=0.8)
    ax[1].set_xticks(x, [f"Qwen2.5-{s}" for s in sizes])
    ax[1].set_ylabel("accuracy change vs digital (points)\nmean of ARC-Easy and MMLU")
    ax[1].set_ylim(-36, 6)
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    ax[1].set_title("(b) Model size (200 questions per task)")
    # (c) large exam, four tasks
    run = [("Digital", "digital_X2000_digital", ps.MUTED), ("Group scaling", "chip_digitalhead_s0_X2000_C", ps.SERIES[2]),
           ("+ correction, chip 1", "chip_digitalhead_s0_X2000_CD", ps.SERIES[0]), ("+ correction, chip 2", "chip_digitalhead_s1_X2000_CD_chip2", ps.SERIES[4])]
    ks = [("arc_easy", "ARC-E"), ("arc_challenge", "ARC-C"), ("hellaswag", "HellaSwag"), ("mmlu", "MMLU")]
    x = np.arange(len(ks))
    for i, (lab, n, col) in enumerate(run):
        t = tasks(f"Qwen2.5-0.5B-Instruct_{n}.json")
        ax[2].bar(x + (i - 1.5) * 0.2, [a(t, k) for k, _ in ks], width=0.2, color=col, label=lab)
    ax[2].set_xticks(x, [n for _, n in ks])
    ax[2].set_ylabel("accuracy (%)")
    ax[2].set_ylim(0, 72)
    ax[2].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax[2].set_title("(c) Up to 2,000 questions (0.5B)")
    fig.suptitle("Emulated task accuracy on molecular crossbar tiles (estimated)", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    print(ps.save(fig, "E11_accuracy"))


def fig_bits() -> None:
    ps.apply()
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    base = p(tasks("Qwen2.5-0.5B-Instruct_chip_digitalhead_s0_C_tile_scaling.json"))
    dig = p(tasks("Qwen2.5-0.5B-Instruct_digital_d200.json"))
    ins = {6: "in6_out12", 7: "in7_out12", 8: None, 10: "in10_out12"}
    outs = {8: "in8_out8", 9: "in8_out9", 10: "in8_out10", 12: None, 14: "in8_out14"}
    for axx, data, xl in [(ax[0], ins, "input converter bits (outputs 12)"), (ax[1], outs, "output converter bits (inputs 8)")]:
        xs = sorted(data)
        ys = [base if data[k] is None else p(tasks(f"Qwen2.5-0.5B-Instruct_chip_digitalhead_s0_BITS_{data[k]}.json")) for k in xs]
        axx.plot(xs, ys, "o-", color=ps.SERIES[0])
        axx.axhline(dig, color=ps.MUTED, ls="--", lw=1)
        axx.text(xs[0], dig - 0.35, "digital", fontsize=8, color=ps.MUTED)
        axx.set_xticks(xs)
        axx.set_xlabel(xl)
    ax[0].set_ylabel("WikiText-2 perplexity")
    ax[0].set_ylim(17, 21)
    fig.suptitle("Converter resolution with group scaling, Qwen2.5-0.5B (estimated)", fontsize=9, color=ps.INK2)
    fig.tight_layout()
    print(ps.save(fig, "E11_bits"))


def fig_drift() -> None:
    ps.apply()
    d = json.loads((Path(__file__).resolve().parent.parent / "results" / "E13_drift.json").read_text())
    times = ["1 day", "1 month", "1 year"]
    days = [1, 30, 365]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    nus = sorted({r["nu"] for r in d["rows"]})
    for i, nu in enumerate(nus):
        rows = {r["time"]: r["loss_pct"] for r in d["rows"] if r["nu"] == nu}
        ys = [rows.get(t, rows.get("1 day")) for t in times]
        ax.plot(days, ys, "o-", color=ps.SERIES[i % len(ps.SERIES)], label=f"nu = {nu:g}" + (" (PCM-like)" if nu == 0.06 else ""))
    ax.axhline(5, color=ps.MUTED, ls="--", lw=1)
    ax.text(1.1, 5.6, "5% budget", fontsize=8, color=ps.MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(days, ["1 day", "1 month", "1 year"])
    ax.set_ylabel("perplexity loss vs digital (%)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Drift tolerance, GPT-2 small (estimated)")
    fig.tight_layout()
    print(ps.save(fig, "E13_drift"))


if __name__ == "__main__":
    fig_accuracy()
    fig_bits()
    fig_drift()
