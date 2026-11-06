"""Experiment E7 report: summary table, tolerances and figure from the WSL runs (H5, feeds E8 and E10).

    uv run python experiments/e7_report.py      (after accuracy/run_e7.sh)

Reads results/E7_runs.jsonl, E7_train_*.json and E7_validation.json. Loss is perplexity relative
to the digitally fine-tuned model without analog hardware. Tolerances use 2%, 5% and 10% loss.
The level tolerance is turned into the smallest molecular window that provides it (constant
0.35 uS spacing, published) and the weight slices E8/E10 assumed. All values simulated (estimated).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from common import RESULTS, save  # noqa: E402
from molsim import plotstyle  # noqa: E402
from molsim.params import load  # noqa: E402

LOSSES = (2, 5, 10)
TIME_S = {"1 s": 1, "1 h": 3600, "1 day": 86400, "1 year": 3.15e7}
MODEL_LABEL = {"digital": "digital fine-tune", "hwa": "noise-aware fine-tune"}


def tolerance(d: pd.DataFrame, x: str, base: float, larger_is_harder: bool) -> dict:
    """Largest noise (or fewest levels) whose perplexity stays within each loss budget."""
    out = {}
    for loss in LOSSES:
        ok = d[d.ppl_mean <= base * (1 + loss / 100)]
        out[f"{loss}%"] = None if ok.empty else float(ok[x].max() if larger_is_harder else ok[x].min())
    return out


def window_for_levels(levels: float | None) -> float | None:
    if levels is None:
        return None
    m = load("devices")["molecular"]
    return (levels - 1) * m["level_spacing"].q.to("uS").magnitude + m["g_min"].q.to("uS").magnitude


def main() -> None:
    df = pd.DataFrame([json.loads(x) for x in (RESULTS / "E7_runs.jsonl").read_text().splitlines()])
    df["prog_rel"] = df.params.map(lambda p: p.get("prog_rel"))
    df["levels"] = df.params.map(lambda p: p.get("levels"))
    train = {m: json.loads((RESULTS / f"E7_train_{m}.json").read_text()) for m in ("digital", "hwa")}
    validation = json.loads((RESULTS / "E7_validation.json").read_text())
    base = float(df[(df.model == "digital") & (df.group == "digital")].ppl_mean.iloc[0])
    df["loss_pct"] = 100 * (df.ppl_mean / base - 1)

    tol = {}
    for m in ("digital", "hwa"):
        d = df[df.model == m]
        pn = tolerance(d[d.group == "prog_noise"], "prog_rel", base, True)
        lv = tolerance(d[d.group == "levels"], "levels", base, False)
        ro = tolerance(d[d.group == "readout"].assign(out_noise=lambda x: x.out_noise.astype(float)), "out_noise", base, True)
        tol[m] = {
            "programming_noise_share_of_gmax": pn, "readout_noise": ro, "levels_per_device": lv,
            "molecular_window_uS_for_levels": {k: window_for_levels(v) for k, v in lv.items()},
            "weight_bits_equivalent": {k: (None if v is None else 1 + math.log2(v)) for k, v in lv.items()},
        }
    dev = df[df.group == "device"].copy()
    dev_table = dev.pivot_table(index=["model", "name"], columns="t", values="loss_pct").reset_index()

    payload = {
        "experiment": "E7", "title": "GPT-2 small on simulated analog hardware: devices, time and tolerances", "label": "estimated (simulated perplexity)",
        "model": "GPT-2 small (124M), fine-tuned on WikiText-2 train; output head digital", "test_tokens": int(df.windows.iloc[0]) * 512,
        "baseline_ppl_digital_ft": base, "baseline_ppl_hwa_digital": float(df[(df.model == "hwa") & (df.group == "digital")].ppl_mean.iloc[0]),
        "training": {m: {k: train[m][k] for k in ("steps", "tokens_seen", "seconds", "hwa_out_noise", "hwa_weight_noise")} for m in train},
        "validation": validation, "tolerances": tol, "nominal_readout_noise": 0.02,
        "devices": dev_table.to_dict("records"), "runs": df.drop(columns=["params"]).to_dict("records"),
    }
    path = save("E7_accuracy", payload)

    # ── figure ──
    plotstyle.apply()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    names = list(dict.fromkeys(dev[dev.model == "hwa"].name))
    for i, n in enumerate(names):
        d = dev[(dev.model == "hwa") & (dev.name == n)].assign(ts=lambda x: x.t.map(TIME_S)).sort_values("ts")
        ax[0][0].plot(d.ts, d.ppl_mean, marker="o", color=plotstyle.SERIES[i], label=n)
    ax[0][0].axhline(base, color=plotstyle.INK2, linestyle="--", linewidth=1.2, label="digital fine-tune, no analog")
    ax[0][0].set(xscale="log", yscale="log", xlabel="time after programming (s)", ylabel="perplexity", title="Devices over time (noise-aware model, readout noise 0.02)")
    ax[0][0].legend(fontsize=7)
    for j, m in enumerate(("digital", "hwa")):
        d = df[(df.model == m) & (df.group == "readout")].assign(x=lambda x: x.out_noise.astype(float)).sort_values("x")
        ax[0][1].plot(d.x, d.ppl_mean, marker="o", color=plotstyle.SERIES[j], label=MODEL_LABEL[m])
        d = df[(df.model == m) & (df.group == "prog_noise")].sort_values("prog_rel")
        ax[1][0].plot(100 * d.prog_rel, d.ppl_mean, marker="o", color=plotstyle.SERIES[j], label=MODEL_LABEL[m])
        d = df[(df.model == m) & (df.group == "levels")].sort_values("levels")
        ax[1][1].plot(d.levels, d.ppl_mean, marker="o", color=plotstyle.SERIES[j], label=MODEL_LABEL[m])
    for a, title, xl, xlog in [(ax[0][1], "Readout noise (ideal devices)", "readout noise (share of converter range)", False),
                               (ax[1][0], "Programming noise tolerance (no readout noise)", "programming noise (% of g_max)", True),
                               (ax[1][1], "Levels per device (no noise)", "conductance levels per device", True)]:
        linear_y = a is not ax[1][0]  # narrow ranges read better on a linear axis
        a.axhline(base, color=plotstyle.INK2, linestyle="--", linewidth=1.2, label="digital fine-tune, no analog")
        a.axhspan(base, base * 1.05, color=plotstyle.GRID, zorder=0)
        a.set(yscale="linear" if linear_y else "log", xlabel=xl, ylabel="perplexity", title=title)
        if xlog:
            a.set_xscale("log", base=2 if "levels" in xl else 10)
        a.legend(fontsize=7)
    for a in ax.flat:
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 100 else f"{v:g}"))
        if a.get_yscale() == "log":
            a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle("E7: GPT-2 small on simulated analog hardware (WikiText-2 perplexity, lower is better; grey band: within 5% of digital). Estimated.", fontsize=9, color=plotstyle.INK2)
    fig.tight_layout()
    fig_path = plotstyle.save(fig, "E7_accuracy")

    print(f"digital fine-tune perplexity {base:.2f}; noise-aware model without analog {payload['baseline_ppl_hwa_digital']:.2f}\n")
    with pd.option_context("display.width", 200, "display.float_format", "{:+.1f}%".format):
        print("loss vs digital fine-tune, devices at readout noise 0.02:")
        print(dev_table[["model", "name"] + [t for t in TIME_S if t in dev_table.columns]].to_string(index=False))
    print("\nreadout-noise sweep (ideal devices):")
    for m in ("digital", "hwa"):
        d = df[(df.model == m) & (df.group == "readout")]
        print(f"  {MODEL_LABEL[m]:24} " + "  ".join(f"{float(r.out_noise):g}: {r.ppl_mean:.2f}" for r in d.itertuples()))
    print("\ntolerances (loss budget -> limit):")
    for m, t in tol.items():
        print(f"  {MODEL_LABEL[m]}")
        for k, v in t.items():
            print(f"    {k:34} {v}")
    print(f"\nsaved {path}\nsaved {fig_path}")


if __name__ == "__main__":
    main()
