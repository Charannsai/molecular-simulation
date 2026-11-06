import pytest

from molsim import Value
from molsim.digital import bytes_per_token, memory_ceiling, roofline
from molsim.llm import Model


def measured(x, unit):
    return Value.of(x, unit, label="measured", source="test")


def test_ceiling_is_bandwidth_over_bytes():
    # 8.03B weights at exactly 1 byte each, no context, 1000 GB/s -> about 124.5 tok/s (embedding not streamed)
    m = Model.load("llama-3.1-8b")
    b = bytes_per_token(m, measured(m.total_params(), "B"), context=0)
    ceiling = memory_ceiling(b["total"], measured(1000, "GB/s"))
    assert ceiling.q.units == ceiling.q.to("tok/s").units
    assert ceiling.m == pytest.approx(1000e9 / m.streamed_params(), rel=1e-9)


def test_kv_cache_grows_with_context():
    m = Model.load("llama-3.1-8b")
    short = bytes_per_token(m, measured(5e9, "B"), context=0)["total"].m
    long = bytes_per_token(m, measured(5e9, "B"), context=32_000)["total"].m
    # 32k tokens x 131,072 bytes = about 4.2 GB extra per generated token
    assert long - short == pytest.approx(32_000 * 131_072 / 1e9, rel=1e-6)


def test_decode_is_memory_bound_on_a_typical_gpu():
    m = Model.load("llama-3.1-8b")
    r = roofline(m, measured(4.9e9, "B"), 400, measured(367, "GB/s"), measured(24.6, "Top/s"))
    assert r["ceiling"].m == r["memory_ceiling"].m
    assert r["compute_ceiling"].m > 10 * r["memory_ceiling"].m


def test_labels_follow_the_inputs():
    m = Model.load("llama-3.2-3b")
    r = roofline(m, measured(3.4e9, "B"), 400, measured(367, "GB/s"), measured(24.6, "Top/s"))
    # the parameter count and KV size are computed, so the ceiling is an estimate
    assert r["ceiling"].label == "estimated"
