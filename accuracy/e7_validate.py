"""E7 method check: does FastAnalogLinear reproduce AIHWKit on GPT-2 small? (run inside WSL)

Same model, same test windows, same settings; perplexity from both simulators.
    python accuracy/e7_validate.py      -> results/E7_validation.json
"""

from __future__ import annotations

import gc
import json
import time

import torch
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog

import gpt2_common as g

N_WINDOWS = 16
SEEDS = 2


def fast(out_noise: float, noise_model=None, t: float = 0.0) -> float:
    m = g.analogize(g.load_model(), g.AnalogSettings(out_noise=out_noise)).to(g.DEV).eval()
    if noise_model is not None:
        g.program(m, noise_model, t)
    return g.perplexity(m, TEST, N_WINDOWS)


def aihwkit(out_noise: float, noise_model=None, t: float = 0.0, columnwise: bool = True) -> float:
    m = g.load_model().to(g.DEV)
    m = convert_to_analog(m, g.aihwkit_rpu(noise_model, out_noise, columnwise), inplace=True, exclude_modules=g.DIGITAL).eval()
    if noise_model is not None:
        m.program_analog_weights()
        m.drift_analog_weights(t)
    ppl = g.perplexity(m, TEST, N_WINDOWS)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return ppl


TEST = g.tokens("test")


def main() -> None:
    digital = g.perplexity(g.load_model().to(g.DEV).eval(), TEST, N_WINDOWS)
    print(f"digital {digital:.3f}", flush=True)
    cases = [("converters only (8-bit in, 12-bit out)", 0.0, None, 0.0),
             ("readout noise 0.02", 0.02, None, 0.0),
             ("PCM (PCMLikeNoiseModel), 1 day, no readout noise", 0.0, lambda: PCMLikeNoiseModel(g_max=25.0), 86400.0)]
    rows = []
    for name, on, nm, t in cases:
        seeds = 1 if (on == 0 and nm is None) else SEEDS
        res = {}
        for sim, fn in (("fast", fast), ("aihwkit", aihwkit)):
            t0, vals = time.time(), []
            for s in range(seeds):
                torch.manual_seed(s)
                vals.append(fn(on, nm() if nm else None, t))
            res[sim] = {"ppl": sum(vals) / len(vals), "runs": vals, "seconds": time.time() - t0}
        res["difference_pct"] = 100 * (res["fast"]["ppl"] / res["aihwkit"]["ppl"] - 1)
        if nm is not None:  # the same device without per-column weight scaling (one scale per tile)
            torch.manual_seed(0)
            res["aihwkit_per_tile_scaling"] = {"ppl": aihwkit(on, nm(), t, columnwise=False)}
        rows.append({"case": name, **res})
        print(f"{name:52} fast {res['fast']['ppl']:8.3f} ({res['fast']['seconds']:5.0f} s)  aihwkit {res['aihwkit']['ppl']:8.3f} ({res['aihwkit']['seconds']:5.0f} s)  diff {res['difference_pct']:+.1f}%", flush=True)
    out = {"check": "FastAnalogLinear vs AIHWKit 1.2.0 on GPT-2 small", "windows": N_WINDOWS, "tokens": N_WINDOWS * g.WINDOW, "digital_ppl": digital, "cases": rows,
           "gpu": torch.cuda.get_device_name(0)}
    (g.SIM / "results" / "E7_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
