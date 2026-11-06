"""Experiment E3: does the compute-in-memory tile model reproduce real published chips?

    uv run python experiments/e3_cim_validation.py

Checks (each must be within 30%, docs/simulator-plan.md):
  V1  HERMES 1-phase MVM latency from first principles (no fitted constants):
      signed 8-bit pulse-width inputs at 1 ns per step, conversion overlapped with integration.
  V2  HERMES 4-phase MVM latency, same model with 4 phases.
  V3  HERMES 4-phase MVM energy, predicted from the 1-phase calibration.
  V4  Cross-chip: the per-output energy calibrated on HERMES (PCM, 14 nm) predicts NeuRRAM's
      (RRAM, 130 nm) energy per 256 x 256 MVM at 8-bit inputs. Reported with three output-bit
      scaling rules; only "none" (matched precision) is the pass/fail check, the others show how
      much the unsettled bit-scaling rule matters.

Calibration (not validation): HERMES measured MVM energy minus the computed array energy gives the
peripheral energy per output. The array energy needs two numbers HERMES doesn't publish (mean device
conductance, mean input level); both are assumptions with ranges, and the conclusion is checked
across the whole range.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import context, save  # noqa: E402
from molsim import Value  # noqa: E402
from molsim.cim.tile import TileSpec, mvm_energy, mvm_latency, ops_per_mvm  # noqa: E402
from molsim.params import load  # noqa: E402

TOL = 0.30


def err(pred: float, actual: float) -> float:
    return (pred - actual) / actual


def main() -> None:
    chips = load("chips")
    h, n = chips["hermes"], chips["neurram"]

    g_mean = Value.of(2, "uS", label="assumed", source="E3 assumption: mean PCM conductance during reads", range=(0.5, 5))
    duty = Value.of(0.25, "", label="assumed", source="E3 assumption: mean input magnitude / full scale", range=(0.1, 0.5))

    def hermes_tile(per_output: Value, phases: int = 1, g=g_mean, d=duty) -> TileSpec:
        return TileSpec(
            name="HERMES core", rows=int(h["rows"].m), cols=int(h["cols"].m), devices_per_weight=int(h["devices_per_weight"].m),
            input_bits=int(h["input_bits"].m), output_bits=int(h["adc_bits"].m), read_voltage=h["read_voltage"], g_mean=g, input_duty=d,
            periph_energy_per_output=per_output, periph_ref_bits=int(h["adc_bits"].m), adcs=int(h["adcs_per_core"].m),
            input_mode="pwm", signed_inputs=True, t_lsb=Value.of(1, "ns", label="published", source="Le Gallo 2023, Ext. Data Table I (1 ns per LSB)"),
            t_conv=None, t_overhead=None, phases=phases,
        )

    cores = int(h["cores"].m)
    measured_core_j = h["mvm_energy_1phase"].q.to("J").magnitude / cores

    # ── calibration: peripheral energy per output ──
    zero = Value.of(0, "pJ", label="assumed", source="placeholder before calibration", range=(0, 0))
    array_j = mvm_energy(hermes_tile(zero))["array"].q.to("J").magnitude
    per_output_j = (measured_core_j - array_j) / int(h["cols"].m)
    per_output = Value.derive(per_output_j * 1e12, "pJ", [h["mvm_energy_1phase"], g_mean, duty, h["read_voltage"]], "E3 calibration: HERMES peripheral energy per output (ADC, drivers, local digital)")

    # robustness of "periphery dominates" across the assumption ranges
    shares = []
    for g, d, v in itertools.product(g_mean.range, duty.range, h["read_voltage"].range):
        t = hermes_tile(zero, g=Value.of(g, "uS", label="assumed", source="sweep"), d=Value.of(d, "", label="assumed", source="sweep")).with_(
            read_voltage=Value.of(v, "V", label="published", source="sweep"))
        shares.append(mvm_energy(t)["array"].q.to("J").magnitude / measured_core_j)

    # ── V1, V2 latency (first principles) ──
    lat1 = mvm_latency(hermes_tile(per_output)).q.to("ns").magnitude
    lat4 = mvm_latency(hermes_tile(per_output, phases=4)).q.to("ns").magnitude
    e4 = mvm_energy(hermes_tile(per_output, phases=4))["total"].q.to("J").magnitude * cores

    # ── V4 cross-chip energy (NeuRRAM, 8-bit inputs, 10-bit outputs) ──
    pt = next(p for p in n["operating_points"] if p["input_bits"] == 8)
    neurram_ops = 2 * int(n["rows"].m) * int(n["cols"].m)
    neurram_mvm_j = neurram_ops / pt["efficiency"].q.to("op/s/W").magnitude
    rules = {"none": 1.0, "linear": pt["output_bits"] / int(h["adc_bits"].m), "walden": 2.0 ** (pt["output_bits"] - int(h["adc_bits"].m))}
    v4 = {name: int(n["cols"].m) * per_output_j * s for name, s in rules.items()}

    checks = [
        {"id": "V1", "what": "HERMES 1-phase MVM latency (first principles)", "pred": lat1, "actual": h["mvm_latency_1phase"].m, "unit": "ns"},
        {"id": "V2", "what": "HERMES 4-phase MVM latency (first principles)", "pred": lat4, "actual": h["mvm_latency_4phase"].m, "unit": "ns"},
        {"id": "V3", "what": "HERMES 4-phase MVM energy (from 1-phase calibration)", "pred": e4 * 1e6, "actual": h["mvm_energy_4phase"].m, "unit": "uJ"},
        {"id": "V4", "what": "NeuRRAM 256x256 MVM energy at 8-bit inputs (HERMES per-output energy, no bit scaling)", "pred": v4["none"] * 1e9, "actual": neurram_mvm_j * 1e9, "unit": "nJ"},
    ]
    for c in checks:
        c["error_pct"] = 100 * err(c["pred"], c["actual"])
        c["pass"] = abs(c["error_pct"]) <= 100 * TOL
    passed = all(c["pass"] for c in checks)
    hermes_passed = all(c["pass"] for c in checks if c["id"] != "V4")
    cross_chip_passed = next(c["pass"] for c in checks if c["id"] == "V4")
    # Shown for information only, NOT the pre-defined check: transferring HERMES's whole per-output
    # energy (array included) instead of the periphery alone.
    v4_alt_whole_per_output_nJ = measured_core_j / int(h["cols"].m) * int(n["cols"].m) * 1e9

    neurram_points = [
        {"bits_in_w_out": f"{p['input_bits']}/{p['weight_bits']}/{p['output_bits']}", "energy_per_mvm_nJ": 1e9 * neurram_ops / p["efficiency"].q.to("op/s/W").magnitude, "latency_us": p["mvm_latency"].m}
        for p in n["operating_points"]
    ]

    payload = {
        "experiment": "E3",
        "title": "CIM tile model validation against published chips (IBM HERMES, NeuRRAM)",
        "label": "estimated (model) vs published (chip data)",
        "tolerance_pct": 100 * TOL,
        "passed": passed,
        "hermes_passed": hermes_passed,
        "cross_chip_passed": cross_chip_passed,
        "v4_alternative_not_predefined_nJ": v4_alt_whole_per_output_nJ,
        "context": context(),
        "assumptions": {"g_mean_uS": [g_mean.m, *g_mean.range], "input_duty": [duty.m, *duty.range]},
        "calibration": {
            "hermes_core_mvm_energy_nJ": measured_core_j * 1e9,
            "array_energy_nJ": array_j * 1e9,
            "array_share_pct": 100 * array_j / measured_core_j,
            "array_share_range_pct": [100 * min(shares), 100 * max(shares)],
            "periph_energy_per_output_pJ": per_output_j * 1e12,
        },
        "checks": checks,
        "v4_bit_scaling_variants_nJ": {k: v * 1e9 for k, v in v4.items()},
        "neurram_operating_points": neurram_points,
        "hermes_ops_per_core_mvm": ops_per_mvm(hermes_tile(per_output)),
    }
    path = save("E3_cim_validation", payload)

    cal = payload["calibration"]
    print(f"HERMES core MVM energy {cal['hermes_core_mvm_energy_nJ']:.2f} nJ = array {cal['array_energy_nJ']:.2f} nJ ({cal['array_share_pct']:.1f}%)"
          f" + periphery -> {cal['periph_energy_per_output_pJ']:.1f} pJ per output")
    print(f"array share across the assumption corners: {cal['array_share_range_pct'][0]:.1f}% to {cal['array_share_range_pct'][1]:.1f}% (not pinned down by published data)\n")
    for c in checks:
        print(f"{c['id']} {c['what']}: predicted {c['pred']:.3g} vs {c['actual']:.3g} {c['unit']} -> {c['error_pct']:+.1f}% {'PASS' if c['pass'] else 'FAIL'}")
    print("\nV4 with output-bit scaling rules (12-bit HERMES -> 10-bit NeuRRAM): " + ", ".join(f"{k} {v * 1e9:.1f} nJ ({100 * err(v, neurram_mvm_j):+.0f}%)" for k, v in v4.items()))
    print("NeuRRAM measured energy per MVM: " + ", ".join(f"{p['bits_in_w_out']}: {p['energy_per_mvm_nJ']:.1f} nJ" for p in neurram_points))
    print(f"(information only, not the pre-defined check: transferring HERMES's whole per-output energy gives {v4_alt_whole_per_output_nJ:.1f} nJ, {100 * err(v4_alt_whole_per_output_nJ * 1e-9, neurram_mvm_j):+.0f}%)")
    print(f"\nHERMES checks {'PASSED' if hermes_passed else 'FAILED'}; cross-chip check {'PASSED' if cross_chip_passed else 'FAILED'} (tolerance {100 * TOL:.0f}%). saved {path}")


if __name__ == "__main__":
    main()
