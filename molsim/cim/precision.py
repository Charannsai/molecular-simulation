"""How many conductance levels a device offers, and how many column slices a weight then needs.

A weight of `weight_bits` (signed) is stored on a differential pair. When one device holds fewer
levels than the weight needs, the weight is split over several columns (bit slicing) whose outputs
are recombined digitally; every extra slice costs tiles, converters and area.

Molecular devices (Sharma 2024) show 16,520 levels across a 6 mS window with a roughly constant
0.35 uS spacing. Shrinking the window for low read energy then also shrinks the level count, unless
the spacing shrinks with it. Both rules are offered because no source settles it:
- "constant_spacing": levels = window / 0.35 uS (published spacing), at most 16,520;
- "proportional": 16,520 levels at any window (assumed, optimistic).
PCM and RRAM use the effective weight precision their chips report (per signed weight).
These are precision proxies; whether a model keeps its accuracy is tested in E7.
"""

from __future__ import annotations

import math

from ..params import load


def molecular_levels(g_max_uS: float, rule: str = "constant_spacing") -> int:
    m = load("devices")["molecular"]
    top = m["conductance_levels"].m
    if rule == "proportional":
        return int(top)
    if rule == "constant_spacing":
        spacing = m["level_spacing"].q.to("uS").magnitude
        g_min = m["g_min"].q.to("uS").magnitude
        return int(max(2, min(top, (g_max_uS - g_min) / spacing + 1)))
    raise ValueError(rule)


def molecular_slices(weight_bits: int, g_max_uS: float, rule: str = "constant_spacing") -> int:
    """Differential pair: each device holds the magnitude, so it needs weight_bits - 1 bits."""
    bits = max(1, math.floor(math.log2(molecular_levels(g_max_uS, rule))))
    return math.ceil((weight_bits - 1) / bits)


def effective_bits(device: str, optimistic: bool = False) -> float:
    """Signed weight precision one weight cell of the chip delivers (published)."""
    d = load("devices")[device]
    if device == "pcm":
        v = d["effective_weight_bits"]
        return v.range[1] if optimistic else v.m
    if device == "rram":
        # NeuRRAM differential pair: about 4-bit equivalent (Wan 2022); Rao 2023 shows 2,048 levels per device
        return 1 + math.log2(d["max_levels"].m) if optimistic else 4.0
    raise ValueError(device)


def slices(device: str, weight_bits: int, optimistic: bool = False) -> int:
    return math.ceil(weight_bits / effective_bits(device, optimistic))
