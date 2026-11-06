"""E13: how much drift can molecular devices tolerate? (addition 7)

    python accuracy/e13_drift.py      (inside WSL; run by run_e13.sh)

E7 modelled molecular devices with no drift, as the device paper reports (11 days, 7 months unpowered, at or
below 290 K, 64 x 64 array). Here we give them PCM-like power-law drift, g(t) = g0 (t / t0)^-nu with a
device-to-device spread of nu (std = nu / 2), and ask how large nu can be before quality drops. Global drift
compensation per tile is applied, as for PCM in E7. For reference, PCM drift is nu of about 0.06 (Joshi 2020).
Model: the noise-aware GPT-2 small of E7; molecular 10 uS window (28 levels, 42 nS write error, 1% read noise);
readout noise 0.02; 3 seeds; 65,536 WikiText-2 test tokens. Loss is relative to the digital fine-tune.
Output: results/E13_drift.json. All values simulated (estimated).
"""

from __future__ import annotations

import json
import time

import torch

import gpt2_common as g

NUS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.06]
TIMES = {"1 day": 86400.0, "1 month": 2.6e6, "1 year": 3.15e7}
SEEDS = 3
WINDOWS = 128
G_MAX, LEVELS, PROG_REL, READ_REL = 10.0, 28, 0.042 / 10.0, 0.01


class DriftingMolecular(g.MolecularNoiseModel):
    def __init__(self, nu: float):
        super().__init__(G_MAX, LEVELS, PROG_REL, READ_REL)
        self.nu = nu

    @torch.no_grad()
    def generate_drift_coefficients(self, g_target):
        if not self.nu:
            return torch.zeros_like(g_target)
        return (self.nu + 0.5 * self.nu * torch.randn_like(g_target)).clamp(min=0.0)

    @torch.no_grad()
    def apply_drift_noise_to_conductance(self, g_prog, nu, t_inference):
        t0 = 20.0
        drifted = g_prog * ((t_inference + t0) / t0) ** (-nu) if self.nu else g_prog
        return (drifted + self.read_rel * drifted.abs() * torch.randn_like(drifted)).clamp(min=0.0)


def main() -> None:
    test = g.tokens("test")
    digital = g.load_model("digital").to(g.DEV).eval()
    base = g.perplexity(digital, test, WINDOWS)
    del digital
    torch.cuda.empty_cache()
    weights = g.load_model("hwa").state_dict()
    rows = []
    for nu in NUS:
        for tname, t in TIMES.items():
            if nu == 0.0 and tname != "1 day":
                continue  # no drift: identical at every time
            t1, runs = time.time(), []
            for seed in range(SEEDS):
                torch.manual_seed(seed)
                m = g.load_model()
                m.load_state_dict(weights)
                m = g.analogize(m, g.AnalogSettings(out_noise=0.02)).to(g.DEV).eval()
                g.program(m, DriftingMolecular(nu), t)
                runs.append(g.perplexity(m, test, WINDOWS))
                del m
                torch.cuda.empty_cache()
            mean = sum(runs) / len(runs)
            rows.append({"nu": nu, "time": tname, "ppl_mean": mean, "ppl_min": min(runs), "ppl_max": max(runs), "loss_pct": 100 * (mean / base - 1), "seconds": time.time() - t1})
            print(f"nu {nu:<6} {tname:8} ppl {mean:8.3f} ({100 * (mean / base - 1):+.1f}%)", flush=True)
    budget = {}
    for loss in (2, 5, 10):
        ok = [r["nu"] for r in rows if r["time"] == "1 year" and r["loss_pct"] <= loss] + ([0.0] if any(r["nu"] == 0 and r["loss_pct"] <= loss for r in rows) else [])
        budget[f"{loss}%"] = max(ok) if ok else None
    out = {"experiment": "E13", "title": "Molecular drift tolerance (power-law drift, noise-aware GPT-2 small)", "label": "estimated (simulated perplexity)",
           "baseline_ppl_digital_ft": base, "pcm_reference_nu": 0.06, "max_nu_for_loss_budget_at_1_year": budget, "rows": rows, "seeds": SEEDS, "tokens": WINDOWS * g.WINDOW}
    (g.SIM / "results" / "E13_drift.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("tolerable drift at 1 year:", budget)


if __name__ == "__main__":
    main()
