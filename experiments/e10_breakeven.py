"""Experiment E10: the break-even map (all hypotheses; the paper's main figure).

    uv run python experiments/e10_breakeven.py

Question: what device and circuit parameters must a CIM card reach to beat the best local digital
baseline on Llama-3.1-8B decode at batch 1 (4,096-token context)?

Pass criteria (thresholds proposed for this study):
  - energy per token at least 10x below the measured RTX 5050 board (E1);
  - speed at least the RTX 4090's (digital model, E2 utilisation, same context);
  - IR drop at most the limit in params/cards.yaml (analog result trustworthy).
Silicon area is then the deciding output, read against budgets of 1, 4, 16 and 64 reticle-size dies
(budgets are for the reader to choose; none is assumed to be "the" answer).

Levers swept (ranges from params/cards.yaml and devices.yaml):
  tile rows, columns per converter (sharing), periphery area per output, device pitch,
  converter output bits, molecular conductance window (levels, constant spacing rule), weight bits.
PCM and RRAM run the same grid with their own pitch and published precision.
All values estimated.
"""

from __future__ import annotations

import json
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
from molsim.sweep import evaluate, grid  # noqa: E402

CTX = 4096
TAG = "llama3.1:8b-instruct-q4_K_M"
BUDGETS = [1, 4, 16, 64]
AXES = {
    "rows": [128, 256, 512, 1024],
    "adc_share": [1, 2, 4, 8, 16],
    "periph_area": [5438, 2500, 1000, 500],  # um2 per output: HERMES-class down to aggressive
    "output_bits": [8, 10, 12],
    "weight_bits": [4, 8],
}
PITCH_NM = [50, 100, 300, 1100]
MOL_WINDOWS_US = [10, 50]
NOMINAL = {"rows": 256, "adc_share": 1, "periph_area": 5438, "output_bits": 12, "weight_bits": 8, "pitch_nm": 100, "g_max_uS": 10}


def baselines(m: Model) -> dict:
    e1 = json.loads((RESULTS / "E1_baseline.json").read_text())
    e2 = json.loads((RESULTS / "E2_digital.json").read_text())
    util = sum(x["utilisation"] for x in e2["models"].values()) / len(e2["models"])
    w_gb = e2["models"][TAG]["weight_GB_per_token"]
    kv_gb = m.kv_bytes_per_token(2) * CTX / 1e9
    bw4090 = load("systems")["rtx4090"]["memory_bandwidth"].q.to("GB/s").magnitude
    gpu_mj = 1000 * e1["models"][TAG]["decode_J_per_token"]["median"]
    return {"gpu_mJ_tok_measured": gpu_mj, "energy_limit_mJ": gpu_mj / 10, "rtx4090_tok_s": bw4090 * util / (w_gb + kv_gb)}


def design(device: str, p: dict) -> dict:
    """One design point -> card overrides plus the precision bookkeeping."""
    o = {"rows": p["rows"], "adc_share": p["adc_share"], "periph_area": p["periph_area"], "output_bits": p["output_bits"]}
    if device == "molecular":
        o["g_max"] = p["g_max_uS"]
        o["device_pitch"] = Value.of(p["pitch_nm"], "nm", label="assumed", source="sweep", range=(p["pitch_nm"], p["pitch_nm"]))
        o["weight_slices"] = precision.molecular_slices(p["weight_bits"], p["g_max_uS"])
        return {"scenario": "molecular_scaled", "overrides": o}
    o["weight_slices"] = precision.slices(device, p["weight_bits"])
    return {"scenario": device, "overrides": o}


def run_grid(m: Model, limits: dict, ir_limit: float) -> pd.DataFrame:
    rows = []
    points = [("molecular", p | {"pitch_nm": pn, "g_max_uS": g}) for p in grid(AXES) for pn in PITCH_NM for g in MOL_WINDOWS_US]
    points += [(d, p) for d in ("pcm", "rram") for p in grid(AXES)]
    for device, p in points:
        d = design(device, p)
        o = d["overrides"]
        x = evaluate(m, card(d["scenario"], **o), CTX)
        rows.append({"device": device, **p, "slices": o["weight_slices"], **x})
    df = pd.DataFrame(rows)
    df["ok_energy"] = df.mJ_tok <= limits["energy_limit_mJ"]
    df["ok_speed"] = df.tok_s >= limits["rtx4090_tok_s"]
    df["ok_ir"] = df.ir_drop <= ir_limit
    df["passes"] = df.ok_energy & df.ok_speed & df.ok_ir
    return df


def least_aggressive(df: pd.DataFrame, budget: float) -> dict | None:
    """The passing design inside the budget that asks the least of the technology."""
    d = df[df.passes & (df.reticles <= budget)]
    if d.empty:
        return None
    d = d.assign(_p=-d.periph_area, _s=d.adc_share, _r=d.rows, _pitch=-d.pitch_nm.fillna(1e9), _b=-d.output_bits)
    best = d.sort_values(["_p", "_s", "_r", "_pitch", "_b"]).iloc[0]
    keys = ["device", "weight_bits", "g_max_uS", "pitch_nm", "rows", "adc_share", "periph_area", "output_bits", "slices", "reticles", "area_mm2", "mJ_tok", "tok_s", "ir_drop"]
    return {k: best[k] for k in keys}


def tornado(df: pd.DataFrame) -> list[dict]:
    """Area at the nominal design, and with one lever moved to its most aggressive passing value."""
    mol = df[(df.device == "molecular") & (df.weight_bits == NOMINAL["weight_bits"])]

    def at(**kw):
        q = mol
        for k, v in (NOMINAL | kw).items():
            if k != "weight_bits":
                q = q[q[k] == v]
        return q.iloc[0]

    base = at()
    out = [{"lever": "nominal (E4 card, 8-bit weights)", "reticles": base.reticles, "passes": bool(base.passes)}]
    for lever, values in [("rows", [512, 1024]), ("adc_share", [16]), ("periph_area", [500]), ("pitch_nm", [50]), ("output_bits", [8]), ("g_max_uS", [50])]:
        cand = [at(**{lever: v}) for v in values]
        cand = [c for c in cand if c.passes] or cand
        c = min(cand, key=lambda r: r.reticles)
        out.append({"lever": f"{lever} = {c[lever]:g}", "reticles": c.reticles, "passes": bool(c.passes), "factor": base.reticles / c.reticles})
    return out


def figure(df: pd.DataFrame, limits: dict) -> Path:
    plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (dev, label) in zip(axes, [("molecular", "Molecular (stacked)"), ("pcm", "PCM (1T1R)"), ("rram", "RRAM (1T1R)")]):
        d = df[(df.device == dev) & (df.weight_bits == 8) & df.passes]
        piv = d.groupby(["periph_area", "adc_share"]).reticles.min().unstack("adc_share").sort_index(ascending=False)
        im = ax.imshow(np.log10(piv.to_numpy()), cmap=plotstyle.SEQUENTIAL + "_r", vmin=0, vmax=np.log10(600), aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if not np.isnan(v) and v < 4 else plotstyle.INK)
        ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{int(r):,}" for r in piv.index])
        ax.set_xlabel("columns sharing one converter")
        ax.set_title(label)
        ax.grid(False)
    axes[0].set_ylabel("periphery area per output (um2)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("reticle-size dies (log scale)")
    cb.set_ticks(np.log10([1, 4, 16, 64, 256]), labels=["1", "4", "16", "64", "256"])
    fig.suptitle(
        f"E10: fewest reticle-size dies for Llama-3.1-8B, 8-bit weights, passing energy (<= {limits['energy_limit_mJ']:.0f} mJ/token), "
        f"speed (>= {limits['rtx4090_tok_s']:.0f} tok/s, RTX 4090) and IR drop. Best over rows, pitch, output bits and window. Estimated.",
        fontsize=8.5, color=plotstyle.INK2,
    )
    return plotstyle.save(fig, "E10_breakeven")


def main() -> None:
    m = Model.load("llama-3.1-8b")
    limits = baselines(m)
    ir_limit = load("cards")["common"]["ir_drop_limit"].m
    df = run_grid(m, limits, ir_limit)

    budgets = {wb: {b: least_aggressive(df[df.weight_bits == wb], b) for b in BUDGETS} for wb in (4, 8)}
    by_device = {dev: {wb: least_aggressive(df[(df.device == dev) & (df.weight_bits == wb)], 1e9) for wb in (4, 8)} for dev in ("molecular", "pcm", "rram")}
    min_area = {dev: {wb: float(df[(df.device == dev) & (df.weight_bits == wb) & df.passes].reticles.min()) for wb in (4, 8)} for dev in ("molecular", "pcm", "rram")}
    torn = tornado(df)
    fails = {k: int((~df[k]).sum()) for k in ("ok_energy", "ok_speed", "ok_ir")}
    fig = figure(df, limits)

    payload = {"experiment": "E10", "title": "Break-even map: what a CIM card must reach to beat digital on Llama-3.1-8B", "label": "estimated",
               "context_tokens": CTX, "criteria": "thresholds proposed for this study", "limits": limits, "ir_drop_limit": ir_limit,
               "axes": AXES | {"pitch_nm": PITCH_NM, "g_max_uS": MOL_WINDOWS_US}, "nominal": NOMINAL, "designs": len(df), "passing": int(df.passes.sum()),
               "fail_counts": fails, "budgets": budgets, "least_aggressive_by_device": by_device, "min_reticles": min_area, "tornado": torn,
               "context": context()}
    path = save("E10_breakeven", payload)
    df.to_csv(RESULTS / "E10_breakeven_grid.csv", index=False)

    print(f"Llama-3.1-8B, context {CTX}. Limits: energy <= {limits['energy_limit_mJ']:.1f} mJ/token (10x below measured {limits['gpu_mJ_tok_measured']:.0f}), "
          f"speed >= {limits['rtx4090_tok_s']:.1f} tok/s (RTX 4090 model), IR drop <= {ir_limit:.0%}")
    print(f"{len(df):,} designs, {int(df.passes.sum()):,} pass. Failures: {fails}\n")
    print("fewest reticle-size dies among passing designs:")
    for dev, v in min_area.items():
        print(f"  {dev:9} 4-bit {v[4]:7.2f}   8-bit {v[8]:7.2f}")
    for wb in (8, 4):
        print(f"\nleast demanding passing design per budget, {wb}-bit weights:")
        for b, d in budgets[wb].items():
            if d is None:
                print(f"  <= {b:>3} dies: none")
            else:
                print(f"  <= {b:>3} dies: {d['device']}, window {d['g_max_uS']} uS, pitch {d['pitch_nm']} nm, rows {d['rows']}, share {d['adc_share']}, "
                      f"periphery {d['periph_area']} um2, {d['output_bits']}-bit out, {d['slices']} slice(s) -> {d['reticles']:.2f} dies, {d['mJ_tok']:.1f} mJ, {d['tok_s']:.0f} tok/s, IR {d['ir_drop']:.1%}")
    print("\none lever at a time from the nominal card (8-bit weights):")
    for t in torn:
        print(f"  {t['lever']:38} {t['reticles']:8.1f} dies  {'pass' if t['passes'] else 'FAIL'}  {'' if 'factor' not in t else f'{t['factor']:.1f}x smaller'}")
    print(f"\nsaved {path}\nsaved {fig}")


if __name__ == "__main__":
    main()
