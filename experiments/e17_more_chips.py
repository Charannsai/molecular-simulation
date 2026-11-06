"""Experiment E17: more chip checks for the CIM model (addition 1).

    uv run python experiments/e17_more_chips.py      (after e3_cim_validation.py)

E3 calibrated the energy per output on IBM HERMES and missed NeuRRAM's 8-bit point by 34%. Here the same
HERMES-calibrated model, with nothing re-fitted, predicts:
  C1   the analog-tile efficiency of IBM's 34-tile PCM chip (Ambrogio et al., Nature 2023): 20 TOPS/W published
       for a tile of 512 x 512 weights at 4 PCM devices per weight. Inputs taken as HERMES-like 8-bit pulse widths,
       12-bit outputs, HERMES read voltage and PCM conductance (assumptions, stated);
  C2   NeuRRAM's energy per MVM at all four published precision points (1/4/3 to 8/4/10 bits in/weight/out), under
       each output-bit scaling rule (none, linear, Walden). This tests which rule, if any, carries across precisions.
Pass gate: within 30%, as E3. Failures are reported, not re-fitted. All model values estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import RESULTS, context, save  # noqa: E402
from molsim.cim.tile import TileSpec, mvm_energy, ops_per_mvm  # noqa: E402
from molsim.labeled import Value  # noqa: E402
from molsim.params import load  # noqa: E402

TOL = 0.30


def main() -> None:
    chips = load("chips")
    h, n, ibm = chips["hermes"], chips["neurram"], chips["ibm_34tile"]
    e3 = json.loads((RESULTS / "E3_cim_validation.json").read_text())
    per_output_pj = e3["calibration"]["periph_energy_per_output_pJ"]
    per_output = Value.of(per_output_pj, "pJ", label="estimated", source="E3 calibration on IBM HERMES", range=(per_output_pj, per_output_pj))
    g_mean = Value.of(2, "uS", label="assumed", source="E3 assumption: mean PCM conductance during reads", range=(0.5, 5))
    duty = Value.of(0.25, "", label="assumed", source="E3 assumption", range=(0.1, 0.5))

    # ── C1: IBM 34-tile chip, analog tile efficiency ──
    rows = int(round(ibm["weights_per_tile"].m ** 0.5))
    tile = TileSpec(name="IBM 34-tile chip tile (modelled)", rows=rows, cols=rows, devices_per_weight=4, input_bits=8, output_bits=12,
                    read_voltage=h["read_voltage"], g_mean=g_mean, input_duty=duty, periph_energy_per_output=per_output, periph_ref_bits=12,
                    adcs=rows, input_mode="pwm", signed_inputs=True, t_lsb=Value.of(1, "ns", label="assumed", source="HERMES-like inputs", range=(1, 1)))
    e = mvm_energy(tile)
    tops_w = ops_per_mvm(tile) / e["total"].q.to("J").magnitude / 1e12
    actual = ibm["tile_efficiency"].q.to("Top/s/W").magnitude
    checks = [{"id": "C1", "what": f"IBM 34-tile chip, analog tile efficiency ({rows}x{rows} weights, 4 PCM per weight)", "pred": tops_w, "actual": actual, "unit": "TOPS/W",
               "array_share": e["array"].q.to("J").magnitude / e["total"].q.to("J").magnitude}]

    # ── C2: NeuRRAM, all four precision points, three bit-scaling rules ──
    ops = 2 * int(n["rows"].m) * int(n["cols"].m)
    cols = int(n["cols"].m)
    for p in n["operating_points"]:
        actual_nj = 1e9 * ops / p["efficiency"].q.to("op/s/W").magnitude
        ob = int(p["output_bits"])
        for rule, scale in (("none", 1.0), ("linear", ob / 12), ("walden", 2.0 ** (ob - 12))):
            pred_nj = cols * per_output_pj * scale / 1000
            checks.append({"id": f"C2 {p['input_bits']}/{p['weight_bits']}/{ob} {rule}", "what": f"NeuRRAM MVM energy, {p['input_bits']}-bit in, {ob}-bit out, rule {rule}",
                           "pred": pred_nj, "actual": actual_nj, "unit": "nJ"})
    for c in checks:
        c["error_pct"] = 100 * (c["pred"] / c["actual"] - 1)
        c["pass"] = abs(c["error_pct"]) <= 100 * TOL
    per_rule = {}
    for rule in ("none", "linear", "walden"):
        cs = [c for c in checks if c["id"].endswith(rule)]
        per_rule[rule] = {"passes": sum(c["pass"] for c in cs), "of": len(cs), "mean_abs_error_pct": sum(abs(c["error_pct"]) for c in cs) / len(cs)}
    payload = {"experiment": "E17", "title": "More chip checks: IBM 34-tile chip and NeuRRAM across precisions", "label": "estimated (model) vs published (chip data)",
               "tolerance_pct": 100 * TOL, "per_output_pJ_from_E3": per_output_pj, "checks": checks, "neurram_rules": per_rule, "context": context()}
    path = save("E17_more_chips", payload)
    for c in checks:
        print(f"{c['id']:22} {c['pred']:9.3g} vs {c['actual']:9.3g} {c['unit']:7} {c['error_pct']:+7.1f}%  {'pass' if c['pass'] else 'FAIL'}")
    print("NeuRRAM by rule:", {k: f"{v['passes']}/{v['of']} pass, mean |error| {v['mean_abs_error_pct']:.0f}%" for k, v in per_rule.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
