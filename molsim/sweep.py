"""Parameter sweeps and break-even search.

    evaluate(model, card, context) -> one flat row of metrics for a design point
    grid({"rows": [...], "adc_share": [...]}) -> every combination as a dict
    pareto(df, ["area_mm2", "mJ_tok"]) -> the rows no other row beats on every column

Everything here is an estimate built on the labelled parameters; the rows carry the numbers only.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .cim.system import CardSpec, token_costs
from .cim.tile import ir_drop_fraction, mvm_latency, tile_area
from .llm import Model
from .params import load


def evaluate(model: Model, card: CardSpec, context: int) -> dict:
    r = token_costs(model, card, context)
    t = card.tile
    area = tile_area(t)
    tiles = r["mapping"]["tiles"]
    reticle = load("circuits")["fab"]["reticle_area"].q.to("mm**2").magnitude
    area_mm2 = tiles * area["total"].q.to("mm**2").magnitude
    e = r["energy_parts_mJ"]
    return {
        "tiles": tiles,
        "area_mm2": area_mm2,
        "reticles": area_mm2 / reticle,
        "array_area_share": tiles * area["array"].q.to("mm**2").magnitude / area_mm2,
        "mJ_tok": r["energy"].m,
        "tok_s": r["tokens_per_s"].m,
        "mvm_ns": mvm_latency(t).q.to("ns").magnitude,
        "ir_drop": ir_drop_fraction(t),
        "periphery_mJ": e["tile_periphery"],
        "array_mJ": e["tile_array"],
        "kv_mJ": e["kv_cache"],
        "install_s": r["install_time_s"],
    }


def grid(axes: dict[str, list]) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def pareto(df: pd.DataFrame, minimise: list[str]) -> pd.DataFrame:
    """Rows not dominated on every column in `minimise` (lower is better)."""
    vals = df[minimise].to_numpy()
    keep = []
    for i, v in enumerate(vals):
        dominated = ((vals <= v).all(axis=1) & (vals < v).any(axis=1)).any()
        keep.append(not dominated)
    return df[keep].sort_values(minimise[0])
