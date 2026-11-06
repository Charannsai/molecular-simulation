"""Digital (GPU-style) inference model: a roofline for batch-1 decode.

For every generated token the processor must read the streamed weights plus the KV cache.
    memory ceiling  = memory bandwidth / bytes per token
    compute ceiling = compute throughput / operations per token
    speed ceiling   = the smaller of the two
Real software reaches only part of the memory ceiling; that fraction is the utilisation.
"""

from __future__ import annotations

from .labeled import Value, smallest
from .llm import Model

PER_TOKEN = Value.of(1, "tok", label="published", source="definition: one decoding step produces one token")


def bytes_per_param(model: Model, file_bytes: Value) -> Value:
    """Average stored bytes per weight, from the model file size (quantisation formats mix bit widths)."""
    return file_bytes / model.total_params()


def bytes_per_token(model: Model, file_bytes: Value, context: int, kv_bytes_per_value: float = 2.0) -> dict[str, Value]:
    bpp = bytes_per_param(model, file_bytes)
    weights = bpp * model.streamed_params() / PER_TOKEN
    kv = Value.of(model.kv_bytes_per_token(kv_bytes_per_value) * context, "B/tok", label="estimated", source="KV cache read over the context (FP16 cache)")
    return {"bytes_per_param": bpp, "weights": weights.to("GB/tok"), "kv_cache": kv.to("GB/tok"), "total": (weights + kv).to("GB/tok")}


def memory_ceiling(bytes_per_tok: Value, bandwidth: Value) -> Value:
    return (bandwidth / bytes_per_tok).to("tok/s")


def compute_ceiling(model: Model, context: int, throughput: Value) -> Value:
    ops = Value.of(model.decode_flops(context), "op/tok", label="estimated", source="2 operations per streamed weight plus attention")
    return (throughput / ops).to("tok/s")


def roofline(model: Model, file_bytes: Value, context: int, bandwidth: Value, throughput: Value) -> dict[str, Value]:
    b = bytes_per_token(model, file_bytes, context)
    mem = memory_ceiling(b["total"], bandwidth)
    comp = compute_ceiling(model, context, throughput)
    return {**b, "memory_ceiling": mem, "compute_ceiling": comp, "ceiling": smallest([mem, comp])}


def predict(ceiling: Value, utilisation: Value | float) -> Value:
    return (ceiling * utilisation).to("tok/s")
