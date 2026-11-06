"""Build accelerator cards (TileSpec + CardSpec) from params/cards.yaml.

`{ref: "devices.molecular.g_max"}` entries point into the other parameter files, so each number
has exactly one home and keeps its label and source.
"""

from __future__ import annotations

from .cim.system import CardSpec
from .cim.tile import TileSpec
from .labeled import Value
from .params import load

SCENARIOS = ["molecular_demonstrated", "molecular_scaled", "pcm", "rram"]


def _resolve(entry):
    if isinstance(entry, dict) and "ref" in entry:
        file, *path = entry["ref"].split(".")
        node = load(file)
        for p in path:
            node = node[p]
        return node
    return entry


def card(name: str, attention_on: str = "card", **overrides) -> CardSpec:
    """Overrides: any TileSpec or CardSpec field, plus
    g_max (Value, or a number in uS), adc_share (columns per converter) and periph_area (um2 per output)."""
    cfg = load("cards")
    c = cfg["common"]
    s = {k: _resolve(v) for k, v in cfg["scenarios"][name].items()}
    stacked = bool(s.pop("stacked", True))
    g_max: Value = s["g_max"]
    if "g_max" in overrides:
        g = overrides.pop("g_max")
        g_max = g if isinstance(g, Value) else Value.of(g, "uS", label="assumed", source="sweep", range=(g, g))
    share = int(overrides.pop("adc_share", 1))
    periph_area = c["periph_area_per_output"]
    if "periph_area" in overrides:
        a = overrides.pop("periph_area")
        periph_area = a if isinstance(a, Value) else Value.of(a, "um**2", label="assumed", source="sweep", range=(a, a))
    col_frac = c["column_area_fraction"].m
    cols = int(c["tile_cols"].m)
    g_mean = (g_max * c["g_mean_fraction"]).to("uS")
    tile = TileSpec(
        name=s["label_text"],
        rows=int(c["tile_rows"].m), cols=int(c["tile_cols"].m),
        devices_per_weight=int(s["devices_per_weight"].m),
        input_bits=int(c["input_bits"].m), output_bits=int(c["output_bits"].m),
        read_voltage=s["read_voltage"], g_mean=g_mean, input_duty=c["input_duty"],
        periph_energy_per_output=c["periph_energy_per_output"], periph_ref_bits=12,
        adcs=max(1, cols // share), input_mode="pwm", signed_inputs=True, t_lsb=c["t_lsb"],
        t_conv=c["t_conv_shared"] if share > 1 else None,
        device_pitch=s.get("device_pitch"), stacked=stacked,
        adc_area=(periph_area * (1 - col_frac)).to("um**2"), column_area=(periph_area * col_frac).to("um**2"),
        wire_resistance=c["wire_resistance_per_cell"],
        program_iterations=s["program_iterations"], t_program_pulse=s["t_program_pulse"],
        t_verify=s.get("t_verify"), parallel_writes=int(s["parallel_writes"].m),
    )
    tile_over = {k: v for k, v in overrides.items() if hasattr(tile, k)}
    if tile_over:
        tile = tile.with_(**tile_over)
    spec = CardSpec(
        tile=tile, attention_on=attention_on,
        kv_bandwidth=c["kv_bandwidth"], kv_energy_per_bit=c["kv_energy_per_bit"], kv_bytes_per_value=c["kv_bytes_per_value"].m,
        digital_energy_per_op=c["digital_energy_per_op"], digital_throughput=c["digital_throughput"],
        static_power=c["static_power"], link_latency=c["link_latency"], link_energy_per_bit=c["link_energy_per_bit"],
        host_power=c["host_power_gpu_attention"], parallel_tile_writes=int(c["parallel_tile_writes"].m),
        overhead_energy_fraction=c["overhead_energy_fraction"].m, overhead_latency_fraction=c["overhead_latency_fraction"].m,
    )
    card_over = {k: v for k, v in overrides.items() if not hasattr(tile, k)}
    if card_over:
        from dataclasses import replace

        spec = replace(spec, **card_over)
    return spec


def device_area_mm2(name: str, devices: int) -> float | None:
    """Array area from the scenario's device pitch (None when the pitch isn't known)."""
    s = load("cards")["scenarios"][name]
    if "device_pitch" not in s:
        return None
    pitch = _resolve(s["device_pitch"]).q.to("mm").magnitude
    return devices * pitch * pitch
