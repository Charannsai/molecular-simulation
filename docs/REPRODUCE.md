# Reproducing the results

## 1. The simulator (any OS)

```bash
git clone https://github.com/Charannsai/molecular-simulation
cd molecular-simulation
uv sync                      # Python 3.12, numpy, pandas, matplotlib, pint, pyyaml
uv run pytest -q             # unit tests
uv run python experiments/e3_cim_validation.py
uv run python experiments/e4_comparison.py
uv run python experiments/e10_breakeven.py
uv run python experiments/e12_uncertainty.py
uv run python report/figures.py
```

These need no GPU and run in seconds to minutes.

## 2. GPU measurements (optional, your own machine)

```bash
uv sync --extra bench                      # PyTorch with CUDA 12.8
uv run python bench/bandwidth.py           # E0
uv run python bench/compute.py             # E0b
uv run python bench/measure_llm.py         # E1, needs Ollama with llama3.1:8b-instruct-q4_K_M and llama3.2:3b-instruct-q8_0
```

Your numbers describe your GPU; update `params/systems.yaml` before E2 if you want the digital model calibrated on it.

## 3. Accuracy experiments (Linux or WSL2 with an NVIDIA GPU)

AIHWKit has no Windows build, so the accuracy code runs under Linux or WSL2.

```bash
bash accuracy/setup_wsl.sh                 # creates ~/molsim-acc with PyTorch, AIHWKit 1.2.0, transformers, datasets
bash accuracy/run_e7.sh all                # E7: GPT-2 fine-tuning and evaluation (a few hours on a laptop GPU)
bash accuracy/run_e13.sh                   # E13: drift (about 20 minutes)
bash accuracy/run_e11.sh diag              # E11 diagnostics (about 15 minutes)
bash accuracy/run_e11.sh exam              # E11, 2,000 questions per task (about 1.5 hours)
bash accuracy/run_e11.sh bits              # converter resolution (about 30 minutes)
bash accuracy/run_e11.sh size              # 1.5B and 3B (3B needs about 16 GB of GPU memory)
uv run python report/figures_e11.py        # figures for E11 and E13
```

Set `PY` to use another Python, for example `PY="python -u" bash accuracy/run_e11.sh diag`.

### Running the 3B model on Colab

`python accuracy/make_colab_bundle.py` writes `docs/colab/e11_bundle.zip` and a notebook. Upload the notebook to Colab, choose a T4 GPU, run all cells, upload the zip when asked, and download `e11_results.zip` at the end. The notebook uses `--dtype auto`, which falls back to float16 on GPUs without bfloat16.

## 4. NeuroSim converter probe (E20)

```bash
git clone --depth 1 https://github.com/neurosim/DNN_NeuroSim_V1.4 ~/neurosim/DNN_NeuroSim_V1.4
make -C ~/neurosim/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM
bash neurosim/build_probe.sh "$(pwd)/neurosim"   # writes neurosim_adc.csv next to the probe
```

The probe links against NeuroSim's compiled objects and calls its `SarADC` model for 5 to 12 bits at 22, 14 and 7 nm.

## Expected run-to-run variation

Measurements vary by a few percent between runs. Emulated accuracy depends on the seed (each seed is a new chip); with 200 questions per task, differences under about 3.5 points are within sampling error.
