import math

import pytest

from molsim import Value
from molsim.cim.mapping import map_model
from molsim.cim.system import CardSpec, token_costs
from molsim.cim.tile import TileSpec, mvm_energy, mvm_latency, program_time, tile_area
from molsim.llm import Model


def a(x, unit):
    return Value.of(x, unit, label="assumed", source="test", range=(x, x))


def tile(**kw) -> TileSpec:
    base = dict(
        name="test", rows=256, cols=256, devices_per_weight=2, input_bits=8, output_bits=8,
        read_voltage=a(0.2, "V"), g_mean=a(1, "uS"), input_duty=a(0.25, ""),
        periph_energy_per_output=a(10, "pJ"), periph_ref_bits=8, adcs=256,
        input_mode="pwm", t_lsb=a(1, "ns"), device_pitch=a(100, "nm"), adc_area=a(1000, "um**2"),
        program_iterations=a(1, ""), t_program_pulse=a(100, "ns"), t_verify=None, parallel_writes=256,
    )
    base.update(kw)
    return TileSpec(**base)


def test_pwm_latency_signed_inputs_use_half_the_levels():
    assert mvm_latency(tile()).q.to("ns").magnitude == pytest.approx(128)
    assert mvm_latency(tile(signed_inputs=False)).q.to("ns").magnitude == pytest.approx(256)
    assert mvm_latency(tile(phases=4)).q.to("ns").magnitude == pytest.approx(512)


def test_shared_converters_add_conversion_rounds():
    shared = tile(adcs=32, t_conv=a(10, "ns"))
    assert mvm_latency(shared).q.to("ns").magnitude == pytest.approx(128 + 8 * 10)


def test_energy_parts_add_up_and_follow_physics():
    e = mvm_energy(tile())
    devices = 256 * 256 * 2
    array_j = devices * 0.2**2 * 1e-6 * 0.25 * 128e-9
    assert e["array"].q.to("J").magnitude == pytest.approx(array_j)
    assert e["periphery"].q.to("J").magnitude == pytest.approx(256 * 10e-12)
    assert e["total"].m == pytest.approx(e["array"].m + e["periphery"].m)
    assert e["total"].label == "estimated"


def test_bit_scaling_rules():
    walden = mvm_energy(tile(output_bits=10, bit_scaling="walden"))["periphery"].q.to("pJ").magnitude
    linear = mvm_energy(tile(output_bits=10, bit_scaling="linear"))["periphery"].q.to("pJ").magnitude
    assert walden == pytest.approx(256 * 10 * 4)
    assert linear == pytest.approx(256 * 10 * 10 / 8)


def test_stacked_area_is_the_larger_part():
    stacked = tile_area(tile())
    side = tile_area(tile(stacked=False))
    assert stacked["total"].m == pytest.approx(max(stacked["array"].m, stacked["periphery"].m))
    assert side["total"].m == pytest.approx(side["array"].m + side["periphery"].m)


def test_program_time():
    # 131,072 devices, one 100 ns pulse each, 256 at a time -> 51.2 us
    assert program_time(tile()).q.to("us").magnitude == pytest.approx(51.2)


def test_mapping_counts_tiles():
    m = Model.load("llama-3.1-8b")
    mp = map_model(m, tile())
    q = next(p for p in mp.layer if p.matrix.name == "q")
    assert (q.row_tiles, q.col_tiles) == (16, 16)
    down = next(p for p in mp.layer if p.matrix.name == "down")
    assert (down.row_tiles, down.col_tiles) == (56, 16)
    assert mp.total_tiles == 32 * mp.tiles_per_layer + math.ceil(4096 / 256) * math.ceil(128256 / 256)
    assert 0.99 < mp.utilisation <= 1.0


def card(attention_on="card") -> CardSpec:
    return CardSpec(
        tile=tile(), attention_on=attention_on, kv_bandwidth=a(136, "GB/s"), kv_energy_per_bit=a(5, "pJ/bit"),
        kv_bytes_per_value=2, digital_energy_per_op=a(1, "pJ/op"), digital_throughput=a(1, "Top/s"),
        static_power=a(1, "W"), link_latency=a(2, "us"), link_energy_per_bit=a(11.4, "pJ/bit"), host_power=a(30, "W"),
        parallel_tile_writes=1000,
    )


def test_system_energy_breakdown_sums_and_gpu_attention_costs_more():
    m = Model.load("llama-3.1-8b")
    on_card = token_costs(m, card("card"), context=1000)
    on_gpu = token_costs(m, card("gpu"), context=1000)
    parts = sum(on_card["energy_parts_mJ"].values())
    assert on_card["energy"].q.to("mJ/tok").magnitude == pytest.approx(parts)
    assert on_card["energy_parts_mJ"]["host"] == 0
    assert on_gpu["energy_parts_mJ"]["host"] > 0
    assert on_gpu["latency_parts_ms"]["link"] == pytest.approx(2 * 32 * 2e-3)
    assert on_gpu["tokens_per_s"].m < on_card["tokens_per_s"].m
    assert on_card["energy"].label == "estimated"


def test_longer_context_costs_more_kv_energy():
    m = Model.load("llama-3.1-8b")
    short = token_costs(m, card(), context=1000)["energy_parts_mJ"]["kv_cache"]
    long = token_costs(m, card(), context=32000)["energy_parts_mJ"]["kv_cache"]
    assert long == pytest.approx(32 * short)


def test_every_scenario_builds_and_shares_the_same_periphery():
    from molsim.scenarios import SCENARIOS
    from molsim.scenarios import card as scenario_card

    cards = [scenario_card(s) for s in SCENARIOS]
    assert len({c.tile.periph_energy_per_output.m for c in cards}) == 1  # only the device differs
    m = Model.load("llama-3.1-8b")
    for c in cards:
        r = token_costs(m, c, context=512)
        assert r["energy"].m > 0 and r["tokens_per_s"].m > 0
        assert r["energy"].label == "estimated"


def test_moe_charges_only_active_tiles():
    from molsim.scenarios import card as scenario_card

    r = token_costs(Model.load("qwen3-30b-a3b"), scenario_card("molecular_scaled"), context=512)
    assert r["mapping"]["active_tiles"] < 0.2 * r["mapping"]["tiles"]


def test_converter_sharing_cuts_area_and_adds_conversion_rounds():
    from molsim.scenarios import card as scenario_card

    one = scenario_card("molecular_scaled")
    shared = scenario_card("molecular_scaled", adc_share=8)
    assert shared.tile.adcs == 32 and one.tile.t_conv is None
    a1, a8 = tile_area(one.tile)["periphery"].m, tile_area(shared.tile)["periphery"].m
    assert a1 == pytest.approx(256 * 5438e-6)  # HERMES core area per tile, anchored
    assert a8 == pytest.approx(a1 * (0.8 / 8 + 0.2))
    assert mvm_latency(shared.tile).q.to("ns").magnitude == pytest.approx(128 + 8 * 5)


def test_weight_slices_multiply_columns_and_adds():
    m = Model.load("llama-3.1-8b")
    one, two = map_model(m, tile()), map_model(m, tile(weight_slices=2))
    q1 = next(p for p in one.layer if p.matrix.name == "q")
    q2 = next(p for p in two.layer if p.matrix.name == "q")
    assert q2.col_tiles == 2 * q1.col_tiles
    assert q2.partial_sum_adds == (q2.row_tiles * 2 - 1) * 4096


def test_ir_drop_grows_with_rows_squared_and_conductance():
    from molsim.cim.tile import ir_drop_fraction

    base = tile(wire_resistance=a(0.5, "ohm"))
    f = ir_drop_fraction(base)
    assert f == pytest.approx(256**2 / 2 * 1e-6 * 0.25 * 0.5)
    assert ir_drop_fraction(base.with_(rows=512)) == pytest.approx(4 * f)
    assert ir_drop_fraction(tile()) is None


def test_molecular_levels_and_slices():
    from molsim.cim import precision

    assert precision.molecular_levels(6000) == 16520  # capped at the published count
    assert precision.molecular_levels(10) == 28  # (10 - 0.4) / 0.35 + 1
    assert precision.molecular_slices(8, 10) == 2  # 7 magnitude bits on 4-bit devices
    assert precision.molecular_slices(8, 50) == 1
    assert precision.molecular_slices(8, 10, "proportional") == 1
    assert precision.slices("pcm", 8) == 3 and precision.slices("pcm", 4, optimistic=True) == 1


def test_param_cache_returns_independent_copies():
    from molsim.params import load

    a1 = load("cards")
    a1["common"].pop("tile_rows")
    assert "tile_rows" in load("cards")["common"]


def test_pareto_keeps_only_undominated_rows():
    import pandas as pd

    from molsim.sweep import pareto

    df = pd.DataFrame({"area": [1, 2, 3, 2], "energy": [3, 2, 1, 3]})
    assert sorted(pareto(df, ["area", "energy"]).index) == [0, 1, 2]
