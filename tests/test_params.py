import re

import pytest

from molsim import LABELS, Value
from molsim.params import PARAMS_DIR, all_values, load

FILES = ["models", "devices", "circuits", "systems", "chips"]


@pytest.mark.parametrize("name", FILES)
def test_file_loads_and_every_value_is_labelled(name):
    tree = load(name)
    values = all_values(tree)
    assert values, f"{name}.yaml has no labelled values"
    for path, v in values:
        assert isinstance(v, Value)
        assert v.label in LABELS, path
        assert v.source, f"{path} has no source"


def test_assumed_values_have_a_range_to_sweep():
    # An assumption is only useful in a break-even study if we know how far to sweep it.
    for name in FILES:
        for path, v in all_values(load(name)):
            if v.label == "assumed":
                assert v.range is not None, f"{name}:{path} is assumed but has no range"


def test_models_have_complete_shapes():
    needed = {"hidden_size", "num_layers", "num_heads", "num_kv_heads", "head_dim", "vocab_size", "tied_embeddings"}
    for model, spec in load("models").items():
        assert needed <= spec.keys(), f"{model} is missing {needed - spec.keys()}"
        if spec["kind"] == "dense":
            assert "intermediate_size" in spec, model
            assert spec["hidden_size"] == spec["num_heads"] * spec["head_dim"], model
        else:
            assert {"num_experts", "experts_per_token", "moe_intermediate_size"} <= spec.keys(), model


@pytest.mark.parametrize("path", sorted(PARAMS_DIR.glob("*.yaml")))
def test_no_em_or_en_dashes(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search("[–—]", text), f"{path.name} contains an em or en dash"


def test_hermes_table_is_internally_consistent():
    # 64 cores x 256 x 256 MACs x 2 ops in 133 ns must give the published 63.1 TOPS and 0.86 uJ.
    h = load("chips")["hermes"]
    ops = h["cores"].m * h["rows"].m * h["cols"].m * 2
    assert ops / h["mvm_latency_1phase"].q.to("s").magnitude / 1e12 == pytest.approx(h["peak_throughput_1phase"].m, rel=0.01)
    assert ops / (h["peak_efficiency_1phase"].m * 1e12) * 1e6 == pytest.approx(h["mvm_energy_1phase"].m, rel=0.01)


def test_molecular_published_facts_present():
    mol = load("devices")["molecular"]
    for key in ["conductance_levels", "g_max", "read_noise_sigma", "endurance", "cell_width"]:
        assert mol[key].label == "published", key
