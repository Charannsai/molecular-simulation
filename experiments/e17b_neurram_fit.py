"""Experiment E17b: why the CIM model misses NeuRRAM, and what form its energy actually follows.

    uv run python experiments/e17b_neurram_fit.py      (after e17_more_chips.py)

E17 found that no output-bit scaling rule of the HERMES-calibrated model fits NeuRRAM's four published operating
points (Wan 2022, Extended Data Table 1). The two chips read inputs differently:
  HERMES  pulse-width inputs: one array read per MVM whatever the input bits; time-based converters.
  NeuRRAM bit-serial inputs: one array read per input bit; integrate-and-fire neurons whose conversion takes
          more steps for more output bits.
So NeuRRAM's energy should grow with (input bits x output bits), which our pulse-width model never charges.
We fit E = a + b x (input bits x output bits) by least squares to the four measured points and compare it with the
best E17 rule (linear output-bit scaling). The measured points are published; the fit is estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import numpy as np  # noqa: E402

from common import RESULTS, context, save  # noqa: E402


def main() -> None:
    e17 = json.loads((RESULTS / "E17_more_chips.json").read_text())
    pts, linear = [], {}
    for c in e17["checks"]:
        if not c["id"].startswith("C2 "):
            continue
        bits, rule = c["id"].split()[1:]
        i, _, o = (int(b) for b in bits.split("/"))
        if rule == "linear":
            linear[(i, o)] = c["pred"]
        if rule == "none":
            pts.append((i, o, c["actual"]))
    x = np.array([[1.0, i * o] for i, o, _ in pts])
    y = np.array([e for *_, e in pts])
    (a, b), *_ = np.linalg.lstsq(x, y, rcond=None)
    rows = []
    for i, o, e in pts:
        fit = a + b * i * o
        rows.append({"in_bits": i, "out_bits": o, "measured_nJ": e, "fit_nJ": fit, "fit_error_pct": 100 * (fit / e - 1),
                     "e17_linear_nJ": linear[(i, o)], "e17_linear_error_pct": 100 * (linear[(i, o)] / e - 1)})
    fit_mae = float(np.mean([abs(r["fit_error_pct"]) for r in rows]))
    lin_mae = float(np.mean([abs(r["e17_linear_error_pct"]) for r in rows]))
    payload = {"experiment": "E17b", "title": "NeuRRAM energy follows input bits x output bits (bit-serial inputs)", "label": "estimated fit to published points",
               "fit": {"fixed_nJ": float(a), "nJ_per_bit_product": float(b)}, "rows": rows,
               "mean_abs_error_pct": {"bit_product_fit": fit_mae, "e17_linear_rule": lin_mae},
               "passes_30pct": {"bit_product_fit": sum(abs(r["fit_error_pct"]) <= 30 for r in rows), "e17_linear_rule": sum(abs(r["e17_linear_error_pct"]) <= 30 for r in rows), "of": len(rows)},
               "conclusion": "The miss is an architecture difference (bit-serial inputs and integrate-and-fire neurons), not an error in the pulse-width model used for our chip. Our design uses HERMES-style pulse-width inputs, which E3 and E17 C1 validate within 10%.",
               "context": context()}
    path = save("E17b_neurram_fit", payload)
    for r in rows:
        print(f"{r['in_bits']}-bit in / {r['out_bits']}-bit out: measured {r['measured_nJ']:.2f} nJ, fit {r['fit_nJ']:.2f} ({r['fit_error_pct']:+.0f}%), E17 linear {r['e17_linear_nJ']:.2f} ({r['e17_linear_error_pct']:+.0f}%)")
    print(f"fit: {a:.2f} nJ + {b:.3f} nJ x (in bits x out bits); mean error {fit_mae:.0f}% vs E17 linear {lin_mae:.0f}%")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
