"""The whole accelerator card, per generated token (batch 1 decode).

Per layer: the four analog stages run in sequence (mapping.STAGES); between the first two, attention
reads the KV cache. Attention runs either on the card's digital core ("card") or on the host GPU
("gpu", which costs two link round trips per layer and keeps the GPU powered).

Energy per token is reported as a breakdown so no single term can hide:
    tiles (array + periphery) + partial-sum additions + attention arithmetic + KV-cache reads
    + link transfers + static power x time (+ host power x time when the host does attention)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labeled import Value
from ..llm import Model
from .mapping import Mapping, map_model
from .tile import TileSpec, mvm_energy, mvm_latency, program_time, tile_area


@dataclass(frozen=True)
class CardSpec:
    tile: TileSpec
    attention_on: str  # "card" or "gpu"
    kv_bandwidth: Value  # on-card LPDDR bandwidth for the KV cache
    kv_energy_per_bit: Value
    kv_bytes_per_value: float  # 2 = FP16 cache, 1 = 8-bit cache
    digital_energy_per_op: Value  # adds and attention arithmetic on the card's digital core
    digital_throughput: Value  # op/s available to the digital core
    static_power: Value  # card power when idle (controller, leakage, LPDDR refresh)
    link_latency: Value  # one host round trip (used when attention_on == "gpu")
    link_energy_per_bit: Value
    host_power: Value  # extra host power while it does attention (used when attention_on == "gpu")
    parallel_tile_writes: int  # tiles programmed at the same time during model install
    # On-chip data movement and digital units beyond the bare MVM (calibrated on HERMES workloads, E5 notes)
    overhead_energy_fraction: float = 0.0
    overhead_latency_fraction: float = 0.0
    # Optional on-chip SRAM holding the most recent kv_sram_tokens of the KV cache; 0 = none
    kv_sram_tokens: int = 0
    kv_sram_bandwidth: Value | None = None
    kv_sram_energy_per_bit: Value | None = None


def _f(v: Value, unit: str) -> float:
    return v.q.to(unit).magnitude


def token_costs(model: Model, card: CardSpec, context: int) -> dict:
    mp: Mapping = map_model(model, card.tile)
    t_mvm = _f(mvm_latency(card.tile), "s")
    e_tile = mvm_energy(card.tile)
    L = model.num_layers

    # ── latency ──
    analog_s = (L * mp.stages_per_layer * t_mvm + t_mvm) * (1 + card.overhead_latency_fraction)  # sequential stages, plus the output head
    kv_bytes = model.kv_bytes_per_token(card.kv_bytes_per_value) * context  # read once per token (all layers)
    attn_ops = 4 * L * model.num_heads * model.head_dim * context
    adds = mp.partial_sum_adds_per_token
    digital_s = (adds + attn_ops) / _f(card.digital_throughput, "op/s")
    sram_tokens = min(context, card.kv_sram_tokens)
    sram_bytes = model.kv_bytes_per_token(card.kv_bytes_per_value) * sram_tokens
    dram_bytes = kv_bytes - sram_bytes
    kv_s = dram_bytes / _f(card.kv_bandwidth, "B/s")
    if sram_bytes:  # the two memories are read in parallel
        kv_s = max(kv_s, sram_bytes / _f(card.kv_sram_bandwidth, "B/s"))
    link_s = 2 * L * _f(card.link_latency, "s") if card.attention_on == "gpu" else 0.0
    latency_s = analog_s + max(kv_s, digital_s) + link_s  # KV streaming and digital work overlap

    # ── energy ──
    # only the tiles a token uses do work (dense: all of them; MoE: router + chosen experts)
    tiles_j = mp.active_tiles_per_token * _f(e_tile["total"], "J")
    array_j = mp.active_tiles_per_token * _f(e_tile["array"], "J")
    overhead_j = tiles_j * card.overhead_energy_fraction  # on-chip links and digital units around the tiles
    digital_j = (adds + attn_ops) * _f(card.digital_energy_per_op, "J/op")
    kv_j = dram_bytes * 8 * _f(card.kv_energy_per_bit, "J/bit")
    if sram_bytes:
        kv_j += sram_bytes * 8 * _f(card.kv_sram_energy_per_bit, "J/bit")
    link_bytes = 2 * L * 2 * model.hidden_size * 2 if card.attention_on == "gpu" else 2 * model.hidden_size * 2
    link_j = link_bytes * 8 * _f(card.link_energy_per_bit, "J/bit")
    static_j = _f(card.static_power, "W") * latency_s
    host_j = _f(card.host_power, "W") * latency_s if card.attention_on == "gpu" else 0.0
    total_j = tiles_j + overhead_j + digital_j + kv_j + link_j + static_j + host_j

    ins = [card.tile.periph_energy_per_output, card.tile.g_mean, card.kv_energy_per_bit, card.digital_energy_per_op, card.static_power, card.link_energy_per_bit]
    src = f"CIM system model ({card.tile.name}, attention on {card.attention_on}, context {context})"
    mj = lambda j, what: Value.derive(j * 1e3, "mJ/tok", ins, f"{src}: {what}")  # noqa: E731
    area = tile_area(card.tile)["total"] if card.tile.device_pitch is not None and card.tile.adc_area is not None else None
    prog = program_time(card.tile) if card.tile.program_iterations is not None else None

    return {
        "mapping": {"tiles": mp.total_tiles, "active_tiles": mp.active_tiles_per_token, "devices": mp.total_devices, "utilisation": mp.utilisation, "tiles_per_layer": mp.tiles_per_layer},
        "latency": Value.derive(latency_s * 1e3, "ms/tok", [mvm_latency(card.tile), card.kv_bandwidth, card.link_latency], f"{src}: time per token"),
        "tokens_per_s": Value.derive(1 / latency_s, "tok/s", [mvm_latency(card.tile), card.kv_bandwidth, card.link_latency], f"{src}: speed"),
        "latency_parts_ms": {"analog": analog_s * 1e3, "kv_cache": kv_s * 1e3, "digital": digital_s * 1e3, "link": link_s * 1e3},
        "energy": mj(total_j, "energy per token"),
        "energy_parts_mJ": {
            "tile_array": array_j * 1e3, "tile_periphery": (tiles_j - array_j) * 1e3, "onchip_overhead": overhead_j * 1e3, "digital": digital_j * 1e3,
            "kv_cache": kv_j * 1e3, "link": link_j * 1e3, "static": static_j * 1e3, "host": host_j * 1e3,
        },
        "link_bytes_per_token": link_bytes,
        "array_area_mm2": None if area is None else mp.total_tiles * area.q.to("mm**2").magnitude,
        "install_time_s": None if prog is None else prog.q.to("s").magnitude * mp.total_tiles / card.parallel_tile_writes,
    }
