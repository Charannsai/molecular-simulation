"""Smoke test: one labelled calculation from the parameter files.

    uv run python -m molsim

Bandwidth-bound decode ceiling for Llama-3.1-8B at 4-bit on an RTX 4090. This is a first-order
ceiling, not a benchmark: it ignores the KV cache, compute limits and imperfect bandwidth use.
"""

from .labeled import Value
from .params import load


def main() -> None:
    models = load("models")
    systems = load("systems")

    params = models["llama-3.1-8b"]["reported_params"]
    bytes_per_weight = Value.of(0.5, "B", label="assumed", source="4-bit quantisation choice", range=(0.5, 1.0))
    weight_bytes = (params * bytes_per_weight).to("GB")

    bandwidth = systems["rtx4090"]["memory_bandwidth"]
    ceiling = (bandwidth / weight_bytes * Value.of(1, "tok", label="published", source="one weight pass per token")).to("tok/s")

    print("Llama-3.1-8B, 4-bit, RTX 4090 (first-order, ignores KV cache and compute)")
    print(f"  parameters        {params}")
    print(f"  weight bytes      {weight_bytes}")
    print(f"  memory bandwidth  {bandwidth}")
    print(f"  decode ceiling    {ceiling}")


if __name__ == "__main__":
    main()
