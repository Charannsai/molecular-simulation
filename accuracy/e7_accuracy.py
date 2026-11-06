"""Experiment E7: does an LLM keep its quality on noisy analog devices? (H5)

Runs inside WSL2 Ubuntu (AIHWKit has no Windows build); see setup_wsl.sh.
    ~/molsim-acc/.venv/bin/python accuracy/e7_accuracy.py            # full run
    ~/molsim-acc/.venv/bin/python accuracy/e7_accuracy.py --smoke    # timing check

Method
- Model: Qwen2.5-0.5B (open weights; a 3B model does not fit in 8 GB once simulated as analog).
  Every nn.Linear, including the output head, becomes an AIHWKit analog layer on 256 x 256 tiles
  (as params/cards.yaml), 8-bit inputs and 12-bit outputs, AIHWKit's default output noise.
  Embeddings, norms and attention stay digital.
- Quality: perplexity on the WikiText-2 test set (first N windows of 512 tokens).
- Devices:
    PCM     AIHWKit PCMLikeNoiseModel (fitted on IBM PCM; programming, read noise and drift)
    HERMES  AIHWKit HermesNoiseModel (fitted on the IBM HERMES chip)
    RRAM    AIHWKit ReRamWan2022NoiseModel (fitted on NeuRRAM data, Wan 2022)
    Molecular  our model (below): a limited number of levels, programming and read noise, no drift
  each at 1 s, 1 h, 1 day and 1 year after programming, with global drift compensation.
- Tolerance sweeps (break-even inputs for M7): programming noise as a share of g_max with ideal
  levels, and the number of levels with no noise.
- No noise-aware fine-tuning in this run (models are converted as trained). That only helps analog,
  so these are pessimistic for every device alike.

Molecular parameters come from params/devices.yaml (Sharma 2024 published values, plus labelled
assumptions). Results: results/E7_accuracy.json. All perplexities are simulated (estimated).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from aihwkit.inference import (
    GlobalDriftCompensation,
    HermesNoiseModel,
    PCMLikeNoiseModel,
    ReRamWan2022NoiseModel,
    StateIndependentNoiseModel,
)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SIM = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-0.5B"
WINDOW = 512
TIMES = {"1 s": 1.0, "1 h": 3600.0, "1 day": 86400.0, "1 year": 3.15e7}
DEV = "cuda"


class MolecularNoiseModel(StateIndependentNoiseModel):
    """Levels, then Gaussian programming noise (share of g_max), then frozen read noise (share of g). No drift.

    Sharma 2024 shows no drift over 11 days and under 500 nS error after 7 months unpowered on a 6 mS window,
    so drift is set to zero; read noise is drawn once per inference time, as AIHWKit does for PCM.
    """

    def __init__(self, g_max: float, levels: int | None, prog_rel: float, read_rel: float):
        super().__init__(g_max=g_max, prog_coeff=[prog_rel * g_max, 0.0, 0.0], drift_nu_mean=0.0, drift_nu_std=0.0)
        self.levels, self.prog_rel, self.read_rel = levels, prog_rel, read_rel

    @torch.no_grad()
    def apply_programming_noise_to_conductance(self, g_target):
        g = g_target
        if self.levels:
            step = self.g_max / (self.levels - 1)
            g = torch.round(g / step) * step
        if self.prog_rel:
            g = g + self.prog_rel * self.g_max * torch.randn_like(g)
        return g.clamp(min=0.0, max=self.g_max)

    @torch.no_grad()
    def apply_drift_noise_to_conductance(self, g_prog, drift_noise_param, t_inference):
        if not self.read_rel:
            return g_prog
        return (g_prog + self.read_rel * g_prog.abs() * torch.randn_like(g_prog)).clamp(min=0.0)


def rpu(noise_model) -> TorchInferenceRPUConfig:
    c = TorchInferenceRPUConfig()
    c.mapping.max_input_size = 256  # tile rows (params/cards.yaml)
    c.mapping.max_output_size = 256  # tile cols
    c.forward.inp_res = 1 / (2**7 - 2)  # 8-bit signed inputs
    c.forward.out_res = 1 / (2**11 - 2)  # 12-bit signed outputs
    c.noise_model = noise_model
    c.drift_compensation = GlobalDriftCompensation()
    return c


def devices_yaml() -> dict:
    return yaml.safe_load((SIM / "params" / "devices.yaml").read_text(encoding="utf-8"))


def molecular_levels(g_max_uS: float, mol: dict) -> int:
    """Constant level spacing (as molsim/cim/precision.py)."""
    spacing, g_min, top = mol["level_spacing"]["value"], mol["g_min"]["value"], mol["conductance_levels"]["value"]
    return int(max(2, min(top, (g_max_uS - g_min) / spacing + 1)))


def conditions(smoke: bool) -> list[dict]:
    mol = devices_yaml()["molecular"]
    read_rel = mol["relative_read_noise"]["value"] / 100  # assumed 1%
    prog_abs_uS = mol["write_rmse_single_pulse"]["value"] / 1000  # published 42 nS (upper bound, one pulse)
    out = [{"group": "reference", "name": "digital FP32", "noise": None, "t": None, "seeds": 1},
           {"group": "reference", "name": "analog, ideal devices (converters only)", "noise": lambda: MolecularNoiseModel(10.0, None, 0.0, 0.0), "t": "1 s", "seeds": 1}]
    if smoke:
        return out + [{"group": "device", "name": "PCM", "noise": lambda: PCMLikeNoiseModel(g_max=25.0), "t": "1 day", "seeds": 1}]
    seeds = 3
    for t in TIMES:
        out += [
            {"group": "device", "name": "PCM (PCMLikeNoiseModel)", "noise": lambda: PCMLikeNoiseModel(g_max=25.0), "t": t, "seeds": seeds},
            {"group": "device", "name": "PCM (HermesNoiseModel)", "noise": lambda: HermesNoiseModel(), "t": t, "seeds": seeds},
            {"group": "device", "name": "RRAM (ReRamWan2022NoiseModel)", "noise": lambda: ReRamWan2022NoiseModel(), "t": t, "seeds": seeds},
        ]
    for t in ("1 s", "1 year"):
        for g in (10.0, 50.0, 6000.0):  # scaled nominal, a window giving about 8-bit weights, as demonstrated
            lv = molecular_levels(g, mol)
            prog_rel = prog_abs_uS / g
            out.append({"group": "device", "name": f"Molecular, {g:g} uS window ({lv} levels)", "noise": (lambda g=g, lv=lv, p=prog_rel: MolecularNoiseModel(g, lv, p, read_rel)),
                        "t": t, "seeds": seeds, "params": {"g_max_uS": g, "levels": lv, "prog_rel": prog_rel, "read_rel": read_rel}})
    for p in (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
        out.append({"group": "noise_sweep", "name": f"programming noise {100 * p:g}% of g_max", "noise": (lambda p=p: MolecularNoiseModel(10.0, None, p, 0.0)), "t": "1 s", "seeds": seeds, "params": {"prog_rel": p}})
    for lv in (8, 16, 32, 64, 128, 256, 1024):
        out.append({"group": "level_sweep", "name": f"{lv} levels per device", "noise": (lambda lv=lv: MolecularNoiseModel(10.0, lv, 0.0, 0.0)), "t": "1 s", "seeds": 1, "params": {"levels": lv, "weight_bits": 1 + math.log2(lv)}})
    return out


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, n_windows: int, batch: int = 4) -> float:
    windows = ids[: n_windows * WINDOW].view(n_windows, WINDOW)
    nll, count = 0.0, 0
    for i in range(0, n_windows, batch):
        x = windows[i : i + batch].to(DEV)
        out = model(input_ids=x, labels=x)
        n = x.numel() - x.shape[0]
        nll += out.loss.float().item() * n
        count += n
    return math.exp(nll / count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--windows", type=int, default=64)
    args = ap.parse_args()
    n_windows = 8 if args.smoke else args.windows

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]

    def load_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
        m.config.tie_word_embeddings = False
        return m

    results = []
    for c in conditions(args.smoke):
        runs = []
        for seed in range(c["seeds"]):
            torch.manual_seed(seed)
            t0 = time.time()
            m = load_model()
            if c["noise"] is not None:
                m = convert_to_analog(m, rpu(c["noise"]()), inplace=True).eval()
                m.program_analog_weights()
                m.drift_analog_weights(TIMES[c["t"]])
            m.eval()
            t1 = time.time()
            ppl = perplexity(m, ids, n_windows)
            runs.append({"seed": seed, "ppl": ppl, "seconds": time.time() - t0, "eval_seconds": time.time() - t1})
            del m
            torch.cuda.empty_cache()
            print(f"{c['group']:12} {c['name']:48} t={c['t']!s:7} seed {seed}: ppl {ppl:8.3f}  ({runs[-1]['seconds']:.0f} s)", flush=True)
        p = [r["ppl"] for r in runs]
        results.append({k: v for k, v in c.items() if k != "noise"} | {"runs": runs, "ppl_mean": sum(p) / len(p), "ppl_min": min(p), "ppl_max": max(p)})

    digital = results[0]["ppl_mean"]
    for r in results:
        r["ppl_increase_pct"] = 100 * (r["ppl_mean"] / digital - 1)
    payload = {
        "experiment": "E7", "title": "LLM quality on noisy analog devices (AIHWKit)", "label": "estimated (simulated perplexity)",
        "model": MODEL, "dataset": "WikiText-2 raw test", "tokens": n_windows * WINDOW, "window": WINDOW, "smoke": args.smoke,
        "setup": {"tile": "256 x 256", "inputs": "8-bit", "outputs": "12-bit", "out_noise": TorchInferenceRPUConfig().forward.out_noise,
                  "drift_compensation": "global", "fine_tuning": "none"},
        "context": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "aihwkit": __import__("aihwkit").__version__, "os": platform.platform()},
        "results": results,
    }
    out = SIM / "results" / ("E7_accuracy_smoke.json" if args.smoke else "E7_accuracy.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\ndigital perplexity {digital:.3f}; saved {out}")


if __name__ == "__main__":
    main()
