"""One compute-in-memory tile: a crossbar plus its input drivers and output converters.

A tile holds `rows x cols` weights (rows = inputs, cols = outputs), each stored on
`devices_per_weight` devices. One matrix-vector multiply (MVM):
- drives every input row (time-encoded pulse width, or one bit at a time),
- lets current flow through every device (array energy = V^2 x G x time on, per device),
- converts every output column (peripheral energy per output: ADC + drivers + local digital).
Multi-phase reads (HERMES-style, for accuracy) repeat the whole thing.

Low-precision devices store a weight in `weight_slices` column slices (bit slicing), recombined digitally.

Peripheral energy per output is anchored to a published chip (see experiments/e3_cim_validation.py)
and scaled with output bits by an explicit rule, because the literature doesn't settle one:
- "walden": energy doubles per extra bit (ADC figure-of-merit scaling),
- "linear": energy grows in proportion to bits (charge-decrement style converters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..labeled import Value


@dataclass(frozen=True)
class TileSpec:
    name: str
    rows: int
    cols: int
    devices_per_weight: int
    input_bits: int
    output_bits: int
    read_voltage: Value
    g_mean: Value  # average device conductance during reads
    input_duty: Value  # average input magnitude as a fraction of full scale (fraction of max on-time)
    periph_energy_per_output: Value  # at periph_ref_bits; includes ADC, drivers, local digital
    periph_ref_bits: int
    adcs: int  # converters per tile (cols / adcs outputs share one converter)
    input_mode: str = "pwm"  # "pwm" (pulse width) or "bit_serial"
    signed_inputs: bool = True
    t_lsb: Value | None = None  # pwm: time per input step
    t_step: Value | None = None  # bit_serial: time per input bit
    t_conv: Value | None = None  # per conversion round; None when integration and conversion overlap
    t_overhead: Value | None = None  # fixed per-phase overhead (precharge, latch)
    phases: int = 1
    bit_scaling: str = "walden"
    # area
    device_pitch: Value | None = None
    adc_area: Value | None = None  # one converter (shared by cols / adcs columns)
    column_area: Value | None = None  # per column, not shared: integrator, sample-and-hold, mux input
    stacked: bool = True  # devices on top of CMOS: tile area = max(array, periphery), else the sum
    # programming (model install)
    program_iterations: Value | None = None
    t_program_pulse: Value | None = None
    t_verify: Value | None = None
    parallel_writes: int = 1  # cells written at the same time in one tile
    weight_slices: int = 1  # columns per weight (bit slicing for devices with too few levels)
    wire_resistance: Value | None = None  # crossbar wire resistance per cell (IR drop)

    def with_(self, **kw) -> "TileSpec":
        return replace(self, **kw)

    @property
    def weights(self) -> int:
        return self.rows * self.cols

    @property
    def devices(self) -> int:
        return self.weights * self.devices_per_weight


def _s(v: Value | None) -> float:
    return 0.0 if v is None else v.q.to("s").magnitude


def input_time(t: TileSpec) -> float:
    """Seconds the inputs take to apply, per phase."""
    if t.input_mode == "pwm":
        levels = 2 ** (t.input_bits - 1) if t.signed_inputs else 2**t.input_bits
        return levels * _s(t.t_lsb)
    if t.input_mode == "bit_serial":
        return t.input_bits * _s(t.t_step)
    raise ValueError(t.input_mode)


def mvm_latency(t: TileSpec) -> Value:
    rounds = math.ceil(t.cols / t.adcs)
    per_phase = input_time(t) + rounds * _s(t.t_conv) + _s(t.t_overhead)
    return Value.derive(t.phases * per_phase * 1e9, "ns", [t.t_lsb, t.t_step, t.t_conv, t.t_overhead], f"tile model ({t.name}): MVM latency")


def periph_scale(t: TileSpec) -> float:
    d = t.output_bits - t.periph_ref_bits
    if t.bit_scaling == "walden":
        return 2.0**d
    if t.bit_scaling == "linear":
        return t.output_bits / t.periph_ref_bits
    raise ValueError(t.bit_scaling)


def mvm_energy(t: TileSpec) -> dict[str, Value]:
    """Energy of one MVM on one tile, split into array and periphery."""
    v = t.read_voltage.q.to("V").magnitude
    g = t.g_mean.q.to("S").magnitude
    duty = t.input_duty.q.to("").magnitude
    on_time = duty * input_time(t)
    array_j = t.phases * t.devices * v * v * g * on_time
    periph_j = t.phases * t.cols * t.periph_energy_per_output.q.to("J").magnitude * periph_scale(t)
    src = f"tile model ({t.name})"
    array = Value.derive(array_j * 1e9, "nJ", [t.read_voltage, t.g_mean, t.input_duty, t.t_lsb, t.t_step], src + ": array V^2 G t")
    periph = Value.derive(periph_j * 1e9, "nJ", [t.periph_energy_per_output], src + ": converters and drivers")
    return {"array": array, "periphery": periph, "total": Value.derive((array_j + periph_j) * 1e9, "nJ", [array, periph], src + ": MVM energy")}


def tile_area(t: TileSpec) -> dict[str, Value]:
    pitch = t.device_pitch.q.to("um").magnitude
    array_um2 = t.devices * pitch * pitch
    periph_um2 = t.adcs * t.adc_area.q.to("um**2").magnitude
    if t.column_area is not None:
        periph_um2 += t.cols * t.column_area.q.to("um**2").magnitude
    total = max(array_um2, periph_um2) if t.stacked else array_um2 + periph_um2
    src = f"tile model ({t.name}): area"
    return {
        "array": Value.derive(array_um2 / 1e6, "mm**2", [t.device_pitch], src),
        "periphery": Value.derive(periph_um2 / 1e6, "mm**2", [t.adc_area, t.column_area], src),
        "total": Value.derive(total / 1e6, "mm**2", [t.device_pitch, t.adc_area, t.column_area], src + (" (stacked: max)" if t.stacked else " (side by side: sum)")),
    }


def ir_drop_fraction(t: TileSpec) -> float | None:
    """First-order IR drop along a column, as a fraction of the read voltage (estimated).

    Every row injects about V x G x duty x devices-per-polarity of current; the current through the
    k-th wire segment is k times that, so the summed drop is about rows^2 / 2 x G x duty x r_wire x V.
    Above about 10% the analog result is no longer trustworthy without compensation.
    """
    if t.wire_resistance is None:
        return None
    g = t.g_mean.q.to("S").magnitude * max(1, t.devices_per_weight // 2)
    duty = t.input_duty.q.to("").magnitude
    return t.rows**2 / 2 * g * duty * t.wire_resistance.q.to("ohm").magnitude


def program_time(t: TileSpec) -> Value:
    """Seconds to write every device in one tile (model install), at the tile's write parallelism."""
    per_cell = t.program_iterations.m * (_s(t.t_program_pulse) + _s(t.t_verify))
    seconds = t.devices * per_cell / t.parallel_writes
    return Value.derive(seconds * 1e3, "ms", [t.program_iterations, t.t_program_pulse, t.t_verify], f"tile model ({t.name}): program time")


def ops_per_mvm(t: TileSpec) -> int:
    return 2 * t.weights  # one multiply and one add per weight
