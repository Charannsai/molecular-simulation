"""E7 step 2: perplexity of the fine-tuned GPT-2 models on simulated analog hardware (H5).

    python accuracy/e7_eval.py              # every condition, resumes where it stopped
    python accuracy/e7_eval.py --smoke      # a few conditions on the base model

Models: GPT-2 small fine-tuned on WikiText-2 digitally ("digital FT", the fair baseline) and
noise-aware ("noise-aware FT"), both from e7_train.py. Every attention and MLP layer runs on
FastAnalogLinear (validated against AIHWKit, e7_validate.py); the output head stays digital.

Conditions per model:
  readout noise sweep        converters on, ideal devices, readout noise 0 to 0.06
  devices over time          readout noise 0.02 (nominal, assumed), 1 s / 1 h / 1 day / 1 year:
                             PCM (AIHWKit PCMLikeNoiseModel), HERMES (HermesNoiseModel),
                             RRAM (ReRamWan2022NoiseModel), molecular at 10, 50 and 6,000 uS
  tolerance sweeps           readout noise 0: programming noise as a share of g_max; levels per device
Molecular: levels by constant 0.35 uS spacing (published), programming noise 42 nS (published single-pulse
RMSE, an upper bound) on the chosen window, read noise 1% (assumed), no drift (published retention).
Each finished condition is appended to results/E7_runs.jsonl; e7_report.py (Windows) builds the summary.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import yaml
from aihwkit.inference import HermesNoiseModel, PCMLikeNoiseModel, ReRamWan2022NoiseModel

import gpt2_common as g

N_WINDOWS = 128  # 65,536 test tokens
SEEDS = 3
NOMINAL_READOUT = 0.02
TIMES = {"1 s": 1.0, "1 h": 3600.0, "1 day": 86400.0, "1 year": 3.15e7}
OUT = g.SIM / "results" / "E7_runs.jsonl"


def molecular(window_uS: float) -> tuple:
    mol = yaml.safe_load((g.SIM / "params" / "devices.yaml").read_text(encoding="utf-8"))["molecular"]
    spacing, g_min, top = mol["level_spacing"]["value"], mol["g_min"]["value"], mol["conductance_levels"]["value"]
    levels = int(max(2, min(top, (window_uS - g_min) / spacing + 1)))
    prog_rel = mol["write_rmse_single_pulse"]["value"] / 1000 / window_uS
    read_rel = mol["relative_read_noise"]["value"] / 100
    return (lambda: g.MolecularNoiseModel(window_uS, levels, prog_rel, read_rel)), {"g_max_uS": window_uS, "levels": levels, "prog_rel": prog_rel, "read_rel": read_rel}


def conditions(smoke: bool) -> list[dict]:
    if smoke:
        return [{"model": "base", "group": "readout", "name": "readout noise 0.02", "out_noise": 0.02, "seeds": 1},
                {"model": "base", "group": "device", "name": "HERMES", "noise": lambda: HermesNoiseModel(), "t": "1 day", "out_noise": 0.0, "seeds": 1},
                {"model": "base", "group": "device", "name": "RRAM", "noise": lambda: ReRamWan2022NoiseModel(), "t": "1 day", "out_noise": 0.0, "seeds": 1},
                {"model": "base", "group": "device", "name": "Molecular 10 uS", "noise": molecular(10)[0], "t": "1 day", "out_noise": 0.0, "seeds": 1}]
    out = []
    for model in ("digital", "hwa"):
        out.append({"model": model, "group": "digital", "name": "digital (no analog)", "seeds": 1})
        for on in (0.0, 0.01, 0.02, 0.04, 0.06):
            out.append({"model": model, "group": "readout", "name": f"readout noise {on:g}", "out_noise": on, "seeds": 1 if on == 0 else SEEDS})
        devices = [("PCM (PCMLikeNoiseModel)", lambda: PCMLikeNoiseModel(g_max=25.0), list(TIMES), {}),
                   ("PCM (HermesNoiseModel)", lambda: HermesNoiseModel(), list(TIMES), {}),
                   # fitted only at 1 s, 1 day and 2 days (Wan 2022 data): no 1 h or 1 year point exists
                   ("RRAM (ReRamWan2022NoiseModel)", lambda: ReRamWan2022NoiseModel(), ["1 s", "1 day"], {})]
        for w in (10, 50, 6000):
            make, params = molecular(w)
            devices.append((f"Molecular {w:g} uS ({params['levels']} levels)", make, ["1 s", "1 year"], params))
        for name, make, times, params in devices:
            for t in times:
                out.append({"model": model, "group": "device", "name": name, "noise": make, "t": t, "out_noise": NOMINAL_READOUT, "seeds": SEEDS, "params": params})
        for p in (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
            out.append({"model": model, "group": "prog_noise", "name": f"programming noise {100 * p:g}%", "noise": (lambda p=p: g.MolecularNoiseModel(10.0, None, p, 0.0)),
                        "t": "1 s", "out_noise": 0.0, "seeds": SEEDS, "params": {"prog_rel": p}})
        for lv in (8, 16, 32, 64, 128, 256):
            out.append({"model": model, "group": "levels", "name": f"{lv} levels", "noise": (lambda lv=lv: g.MolecularNoiseModel(10.0, lv, 0.0, 0.0)),
                        "t": "1 s", "out_noise": 0.0, "seeds": 1, "params": {"levels": lv, "weight_bits": 1 + math.log2(lv)}})
    return out


def key(c: dict) -> str:
    return f"{c['model']}|{c['group']}|{c['name']}|{c.get('t')}|{c.get('out_noise')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    test = g.tokens("test")
    out = OUT.with_name("E7_runs_smoke.jsonl") if args.smoke else OUT
    done = {json.loads(line)["key"] for line in out.read_text().splitlines()} if out.exists() else set()
    weights = {}
    for c in conditions(args.smoke):
        k = key(c)
        if k in done:
            continue
        ckpt = None if c["model"] == "base" else c["model"]
        if c["model"] not in weights:
            weights[c["model"]] = g.load_model(ckpt).state_dict()
        t0, runs = time.time(), []
        for seed in range(c["seeds"]):
            torch.manual_seed(seed)
            m = g.load_model()
            m.load_state_dict(weights[c["model"]])
            if c["group"] != "digital":
                m = g.analogize(m, g.AnalogSettings(out_noise=c.get("out_noise", 0.0)))
            m = m.to(g.DEV).eval()
            if c.get("noise"):
                g.program(m, c["noise"](), TIMES[c["t"]])
            runs.append(g.perplexity(m, test, 16 if args.smoke else N_WINDOWS))
            del m
            torch.cuda.empty_cache()
        row = {"key": k, **{x: c[x] for x in ("model", "group", "name") if x in c}, "t": c.get("t"), "out_noise": c.get("out_noise"),
               "params": c.get("params", {}), "runs": runs, "ppl_mean": sum(runs) / len(runs), "ppl_min": min(runs), "ppl_max": max(runs),
               "windows": 16 if args.smoke else N_WINDOWS, "seconds": time.time() - t0}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{c['model']:8} {c['group']:10} {c['name']:42} t={c.get('t')!s:7} readout={c.get('out_noise')!s:5} ppl {row['ppl_mean']:9.3f} [{row['ppl_min']:.2f}, {row['ppl_max']:.2f}]  {row['seconds']:4.0f} s", flush=True)


if __name__ == "__main__":
    main()
