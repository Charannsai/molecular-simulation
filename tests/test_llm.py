import pytest

from molsim.llm import Model

MODELS = ["llama-3.2-3b", "llama-3.1-8b", "qwen2.5-32b", "llama-3.1-70b", "qwen3-30b-a3b", "gpt-oss-120b"]


@pytest.mark.parametrize("key", MODELS)
def test_parameter_count_matches_the_model_card(key):
    # If the shapes are wrong, everything downstream is wrong. The model cards round to 3 digits.
    m = Model.load(key)
    reported = m.reported_params.m
    assert abs(m.total_params() - reported) / reported < 0.015, (key, m.total_params(), reported)


def test_llama_8b_known_values():
    m = Model.load("llama-3.1-8b")
    assert m.kv_bytes_per_token(2) == 32 * 2 * 8 * 128 * 2  # 131,072 bytes: grouped-query attention
    assert m.layer_params() == 218_112_000  # 4096x(4096+1024+1024+4096) + 3x4096x14336 + 2x4096
    # The token-embedding table (128256 x 4096) is looked up, not streamed.
    assert m.total_params() - m.streamed_params() == 128_256 * 4096


def test_tied_embeddings_are_streamed_once_as_the_head():
    m = Model.load("llama-3.2-3b")
    assert m.tied_embeddings
    assert m.streamed_params() == m.total_params()


def test_decode_work_is_about_two_operations_per_weight():
    m = Model.load("llama-3.1-8b")
    short = m.decode_flops(context=0)
    assert short == 2 * m.streamed_params()
    assert m.decode_flops(context=4096) > short


@pytest.mark.parametrize("key", ["qwen3-30b-a3b", "gpt-oss-120b"])
def test_moe_active_parameters_match_the_model_card(key):
    # Model cards differ on whether the embedding row counts as active, so accept either convention.
    m = Model.load(key)
    reported = m.reported_active_params.m
    best = min(abs(m.active_params() - reported), abs(m.streamed_params() - reported)) / reported
    assert best < 0.03, (key, m.active_params(), m.streamed_params(), reported)


def test_moe_streams_only_the_chosen_experts():
    m = Model.load("qwen3-30b-a3b")
    assert m.is_moe
    assert m.streamed_params() < 0.15 * m.total_params()
    experts_all = [x for x in m.layer_matrices() if x.expert is not None]
    experts_active = [x for x in m.layer_matrices(active_only=True) if x.expert is not None]
    assert len(experts_all) == 128 * 3 and len(experts_active) == 8 * 3
