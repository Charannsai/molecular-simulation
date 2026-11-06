import pytest
from pint import DimensionalityError

from molsim import Value, smallest


def measured(x, unit):
    return Value.of(x, unit, label="measured", source="test bench")


def published(x, unit):
    return Value.of(x, unit, label="published", source="a paper")


def assumed(x, unit):
    return Value.of(x, unit, label="assumed", source="our assumption")


def test_labels_are_validated():
    with pytest.raises(ValueError):
        Value.of(1, "W", label="guessed", source="x")
    with pytest.raises(ValueError):
        Value.of(1, "W", label="measured", source="")


def test_measured_inputs_stay_measured():
    # joules per token from measured watts and measured tokens per second is a derived measurement
    j_per_token = measured(30, "W") / measured(60, "tok/s")
    assert j_per_token.label == "measured"
    assert j_per_token.derived
    assert j_per_token.q.to("J/tok").magnitude == pytest.approx(0.5)


def test_any_non_measured_input_makes_it_estimated():
    assert (published(8.03e9, "") * 0.5).label == "estimated"
    assert (measured(30, "W") / assumed(60, "tok/s")).label == "estimated"


def test_plain_numbers_do_not_change_the_label():
    doubled = measured(3, "W") * 2
    assert doubled.label == "measured"
    assert doubled.m == 6


def test_basis_counts_original_inputs():
    v = (published(1008, "GB/s") / (published(8.03e9, "") * 0.5 * Value.of(1, "B", label="measured", source="x")))
    basis = v.basis()
    assert basis["published"] == 2
    assert basis["measured"] == 1


def test_units_catch_mistakes():
    with pytest.raises(DimensionalityError):
        _ = measured(1, "J") + measured(1, "W")


def test_conversion_keeps_label_and_range():
    pitch = Value.of(100, "nm", label="assumed", source="x", range=(50, 1000))
    um = pitch.to("um")
    assert um.label == "assumed"
    assert um.m == pytest.approx(0.1)
    assert um.range == pytest.approx((0.05, 1.0))


def test_smallest_keeps_provenance():
    ceiling = smallest([published(100, "tok/s"), assumed(40, "tok/s")])
    assert ceiling.m == 40
    assert ceiling.label == "estimated"
    assert len(ceiling.leaves()) == 2
