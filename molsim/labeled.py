"""Labelled values: every number in the simulator says where it came from.

Labels follow the project rule: measured / published / estimated / assumed.

Combining values follows one conservative rule:
- If every input was measured, the result is still "measured" (a derived measurement,
  e.g. joules per token from measured watts and measured tokens per second).
- Otherwise the result is "estimated". A calculation never becomes "published" (no paper printed it)
  and never stays "assumed" (it is the model's estimate, built on assumptions).

Plain Python numbers (2, 0.5 bytes per weight, ...) are treated as exact configuration choices
and do not change the label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .units import Q, ureg

LABELS = ("measured", "published", "estimated", "assumed")


@dataclass(frozen=True)
class Value:
    q: object  # a pint Quantity
    label: str
    source: str
    note: str = ""
    range: tuple[float, float] | None = None  # plausible low and high, in the same unit as q
    inputs: tuple["Value", ...] = field(default=(), repr=False)

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if not self.source:
            raise ValueError("every value needs a source")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError(f"range low > high: {self.range}")

    # ───────── construction ─────────

    @classmethod
    def of(cls, magnitude, unit: str = "", *, label: str, source: str, note: str = "", range=None) -> "Value":
        return cls(Q(magnitude, unit), label, source, note, tuple(range) if range else None)

    @classmethod
    def derive(cls, magnitude, unit: str, inputs, source: str) -> "Value":
        """A computed value (from a model formula) that keeps the provenance of the Values it used.

        Measured only if every input was measured; otherwise estimated. Use this when a formula is
        clearer in plain floats than in Value arithmetic.
        """
        ins = tuple(v for v in inputs if isinstance(v, Value))
        labels = {v.label for v in ins}
        label = "measured" if ins and labels == {"measured"} else "estimated"
        return cls(Q(magnitude, unit), label, source, inputs=ins)

    # ───────── reading ─────────

    @property
    def m(self) -> float:
        return self.q.magnitude

    @property
    def derived(self) -> bool:
        return bool(self.inputs)

    def to(self, unit: str) -> "Value":
        rng = None
        if self.range is not None:
            lo, hi = (Q(x, self.q.units).to(unit).magnitude for x in self.range)
            rng = (lo, hi)
        return Value(self.q.to(unit), self.label, self.source, self.note, rng, self.inputs)

    def leaves(self) -> list["Value"]:
        """The original (non-derived) values this result was computed from."""
        if not self.inputs:
            return [self]
        out: list[Value] = []
        for v in self.inputs:
            out.extend(v.leaves())
        return out

    def basis(self) -> Counter:
        """How many original inputs of each label went into this value."""
        return Counter(v.label for v in self.leaves())

    def __str__(self) -> str:
        mag = self.m
        text = f"{mag:,.4g} {self.q.units:~P}".strip()
        if self.derived:
            parts = ", ".join(f"{n} {lab}" for lab, n in sorted(self.basis().items(), key=lambda kv: LABELS.index(kv[0])))
            return f"{text} [{self.label}; from {parts}]"
        return f"{text} [{self.label}: {self.source}]"

    # ───────── arithmetic ─────────

    def _combine(self, other, op, symbol: str) -> "Value":
        if isinstance(other, Value):
            q = op(self.q, other.q)
            inputs = (self, other)
            labels = {self.label, other.label}
        else:
            q = op(self.q, other)
            inputs = (self,)
            labels = {self.label}
        label = "measured" if labels == {"measured"} else "estimated"
        return Value(q, label, f"derived ({symbol})", inputs=inputs)

    def __add__(self, o):
        return self._combine(o, lambda a, b: a + b, "+")

    def __sub__(self, o):
        return self._combine(o, lambda a, b: a - b, "-")

    def __mul__(self, o):
        return self._combine(o, lambda a, b: a * b, "*")

    def __truediv__(self, o):
        return self._combine(o, lambda a, b: a / b, "/")

    def __pow__(self, p):
        return self._combine(p, lambda a, b: a**b, "**")

    def __radd__(self, o):
        return self.__add__(o)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __rtruediv__(self, o):
        return Value(o / self.q, self.label if self.label == "measured" else "estimated", "derived (/)", inputs=(self,))


def smallest(values: Iterable[Value]) -> Value:
    """The minimum of several values (e.g. a roofline ceiling), keeping provenance of all of them."""
    vals = list(values)
    best = min(vals, key=lambda v: v.q.to(vals[0].q.units).magnitude)
    labels = {v.label for v in vals}
    return Value(best.q, "measured" if labels == {"measured"} else "estimated", "derived (min)", inputs=tuple(vals))


__all__ = ["LABELS", "Value", "smallest", "Q", "ureg"]
