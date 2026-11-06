# molecular-simulation

**An open, labelled-parameter simulator for molecular memristor compute-in-memory (CIM) and local LLM inference.**

This repository holds the code, parameters, results and figures of a break-even study. It answers one question: **what must molecular compute-in-memory achieve, at the device and circuit level, to beat optimised digital inference for dense 8B language models at batch size one?**

Everything outside the measured GPU baseline is a model estimate. Nothing here is a fabricated chip.

---

## Headline results

| Result | Value | Label |
|---|---|---|
| Llama-3.1-8B (4-bit) on an RTX 5050 Laptop GPU | 52.8 tokens/s, 1.05 J per token, 2 to 3% of compute used | measured |
| Digital model vs measured GPU | within 7% | estimated |
| CIM model vs IBM HERMES (latency, energy) | within 4% and 2% | estimated vs published |
| CIM model vs IBM 34-tile chip | within 9.5% | estimated vs published |
| Accuracy emulator vs IBM AIHWKit | within 1% | estimated |
| CIM energy per token, Llama-3.1-8B, 4k context | 8.8 to 25 mJ (41 to 119 times below measured) over 5,000 random draws | estimated |
| Cause of task-accuracy loss on emulated tiles | the converters, not the molecular devices | estimated (emulated) |
| Qwen2.5 0.5B to 3B with tile-level input scaling | within about 0.5 to 5 points of digital | estimated (emulated) |
| Output converter resolution needed | 10 bits | estimated (emulated) |
| Drift budget for under 5% loss after one year | power-law exponent up to 0.003 | estimated |

## What is in this repository

```
molsim/          the simulator package: labelled values, digital model, CIM tile and system model, sweeps
params/          parameter registers (YAML): devices, chips, circuits, cards, models, systems
bench/           measurement scripts for a local GPU (bandwidth, compute, LLM decode with Ollama)
experiments/     one script per experiment: E2 to E10, E12, E17 and E17b
accuracy/        analog accuracy emulator, GPT-2 study (E7), Qwen2.5 task benchmarks (E11), drift (E13)
neurosim/        probe that drives the DNN+NeuroSim SAR converter model directly (E20)
report/          scripts that draw the figures
results/         every result file (JSON, CSV) and figure of the study
tests/           unit tests for the simulator
docs/            detailed documentation (start with docs/OVERVIEW.md)
```

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                      # install the simulator
uv run pytest -q                             # 56 unit tests
uv run python -m molsim                      # one labelled calculation from the registers
uv run python experiments/e4_comparison.py   # main comparison on Llama-3.1-8B
uv run python experiments/e12_uncertainty.py # 5,000-draw uncertainty study
```

The accuracy experiments need PyTorch, AIHWKit and a CUDA GPU (Linux or WSL2). See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | The research question, the approach and how the pieces fit |
| [docs/MODEL.md](docs/MODEL.md) | The digital model, the CIM tile and system model, area and IR drop, with equations |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | The labelled parameter registers and the rules for numbers |
| [docs/ACCURACY.md](docs/ACCURACY.md) | The analog emulator, the GPT-2 study, the Qwen2.5 task benchmarks and diagnostics |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Every experiment: question, script, outputs and key result |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Environment setup and step-by-step reproduction, including Colab and NeuroSim |

## Rules for numbers

Every value in `params/` and every result carries a label:

- **measured**: measured on the author's laptop;
- **published**: taken from a cited paper or datasheet (values read from plots are marked "verify");
- **estimated**: computed by this simulator;
- **assumed**: a design choice or a value no source reports, always with a range that the sweeps explore.

A result is only as strong as its weakest input.

## Scope of this release

This release covers the break-even study, the validation and the accuracy study. Accelerator-level design studies (packaging, partitioning and cost) are not part of it.

## Citation

If you use this code or its results, please cite this repository (see [CITATION.cff](CITATION.cff)).

## License

MIT, see [LICENSE](LICENSE). Published device and chip parameters remain the work of their cited authors.

## Contact

Charan Sai Pathuri, independent researcher, Hyderabad, India. pathurisai31@gmail.com
