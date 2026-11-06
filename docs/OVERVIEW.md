# Overview

## The problem

At batch size one, generating each token of a large language model (LLM) reads every weight once. For Llama-3.1-8B in 4-bit form that is 4.6 GB per token. On the author's laptop (RTX 5050 Laptop GPU) we measured 52.8 tokens/s and 1.05 J per token, with 67 to 72% of the memory bandwidth in use and only 2 to 3% of the arithmetic. Local LLM inference is limited by moving data, not by computing.

## The idea under test

Compute-in-memory (CIM) stores each weight as the conductance of a device in a crossbar. Applying input voltages to the rows makes each column current equal to a dot product (Ohm's and Kirchhoff's laws), so a matrix-vector product happens where the weights are stored and the weights never move.

Molecular memristors (Sharma et al., Nature 2024) are a candidate device: 16,520 conductance levels, single-pulse programming, a write threshold that allows selector-less crossbars, and months of retention in a 64 x 64 array. They have only been shown in small arrays, at high conductance (up to about 6 mS) and at or below 290 K.

## The approach

1. **Measure** the problem on real hardware (bench/, E0 and E1).
2. **Model** the digital baseline and check it against the measurement (E2).
3. **Model** CIM tiles and whole accelerators with labelled parameters, and validate against published chips: IBM HERMES, IBM's 34-tile chip, NeuRRAM, and the NeuroSim converter model (E3, E17, E17b, E20).
4. **Compare** devices and designs on energy, speed and silicon area, with a sweep of 4,800 designs and a 5,000-draw uncertainty study (E4 to E10, E12).
5. **Emulate** model accuracy on analog tiles: GPT-2 perplexity over time (E7), Qwen2.5 task benchmarks with diagnostics (E11), and drift tolerance (E13).

## How the pieces fit

```
params/*.yaml  ->  molsim.params.load()      labelled values (value, unit, label, source, range)
                    |
                    +-> molsim.digital      GPU decode model (roofline plus calibrated utilisation)
                    +-> molsim.cim.tile     one crossbar tile: latency, energy, area, IR drop
                    +-> molsim.cim.system   a whole accelerator per token: tiles, digital core, KV cache, link
                    +-> molsim.sweep        evaluate many designs, pass/fail, Pareto sets
                    |
experiments/*.py  ->  results/*.json, results/figures/*.png

accuracy/gpt2_common.py     vectorised analog linear layer (validated against AIHWKit)
accuracy/chip_generate.py   ChipLinear: any Hugging Face linear layer on emulated molecular tiles
accuracy/bench_tasks.py     ARC, HellaSwag, MMLU and WikiText-2 on the emulated chip
```

## Main findings

- Removing weight traffic cuts energy per token by one to two orders of magnitude for an 8B model (estimated), and that gain comes from computing in memory, not from the device itself: molecular, PCM and RRAM devices tie at equal conductance and equal circuits.
- Silicon area, set by the peripheral converters, decides feasibility; a single die is out of reach for 8B.
- About 30% of random design draws fail on column IR drop; mitigation is a first-order requirement for tall tiles.
- On task benchmarks, a straightforward emulated design loses accuracy and collapses on a 3B model. The converters cause it; with ideal converters the molecular devices match digital. Scaling each tile's inputs by their own maximum recovers most of the loss across 0.5B, 1.5B and 3B models.
- 10-bit output converters suffice; inputs need about 8 bits.
- The design tolerates a year of drift up to a power-law exponent of 0.003.

## What this work is not

It is not a fabricated or measured molecular chip, and it does not show molecular LLM inference on hardware. All accelerator and accuracy numbers are model estimates.
