# Experiments

Run each script from the repository root with `uv run python <script>` (accuracy scripts with the WSL or Linux environment, see REPRODUCE.md). Results go to `results/`, figures to `results/figures/`.

| ID | Question | Script | Output | Key result (label) |
|---|---|---|---|---|
| E0 | How fast is the GPU's memory and arithmetic? | `bench/bandwidth.py`, `bench/compute.py` | `E0_bandwidth.json`, `E0b_compute.json` | 367.3 GB/s read, 24.6 TFLOP/s FP16 (measured) |
| E1 | How fast and costly is LLM decode on it? | `bench/measure_llm.py` | `E1_baseline.json` | 8B: 52.8 tok/s, 1.052 J/token (measured) |
| E2 | Can a digital model predict it? | `experiments/e2_digital.py` | `E2_digital.json` | within 7% (estimated) |
| E3 | Does the CIM model reproduce IBM HERMES? | `experiments/e3_cim_validation.py` | `E3_cim_validation.json` | latency within 4%, energy within 2% |
| E4 | Main comparison on Llama-3.1-8B | `experiments/e4_comparison.py` | `E4_comparison.json` | 4.6 to 12.9 mJ/token vs 1,052 measured |
| E5 | How does context length change it? | `experiments/e5_context.py` | `E5_context.json` | KV cache dominates: 54% to 81% of energy |
| E6 | Dense vs mixture-of-experts | `experiments/e6_moe.py` | `E6_moe.json` | 90 to 96% of MoE tiles idle |
| E7 | Does GPT-2 stay accurate on analog devices over time? | `accuracy/run_e7.sh`, `experiments/e7_report.py` | `E7_*.json` | molecular within 2.4 to 4.6% for a year |
| E8 | Molecular vs PCM vs RRAM at equal circuits | `experiments/e8_devices.py` | `E8_devices.json` | energy ties; precision and install time differ |
| E9 | Scaling to 32B and 70B | `experiments/e9_scaling.py` | `E9_scaling.json` | silicon scales linearly |
| E10 | Break-even map over 4,800 designs | `experiments/e10_breakeven.py` | `E10_breakeven.json`, `.csv` | none fail energy or speed; area decides |
| E11 | Task accuracy of Qwen2.5 on emulated tiles | `accuracy/run_e11.sh` | `E11_tasks/*.json` | converters cause the loss; tile scaling fixes it |
| E12 | Uncertainty over 5,000 random draws | `experiments/e12_uncertainty.py` | `E12_uncertainty.json` | 8.8 to 25 mJ/token, 41 to 119 times below measured |
| E13 | How much drift is tolerable? | `accuracy/run_e13.sh` | `E13_drift.json` | nu up to 0.003 for under 5% at 1 year |
| E17 | More chip checks | `experiments/e17_more_chips.py` | `E17_more_chips.json` | IBM 34-tile within 9.5% |
| E17b | Why NeuRRAM differs | `experiments/e17b_neurram_fit.py` | `E17b_neurram_fit.json` | bit-serial energy, 13% mean error |
| E20 | NeuroSim converter cross-check | `neurosim/build_probe.sh` | `E20_neurosim_adc.json`, `.csv` | our converter area and energy are conservative |

Figures: `report/figures.py` (E1 to E10) and `report/figures_e11.py` (E11 and E13).

Experiment numbers E14 to E16, E18 and E19 belong to accelerator-level design studies that are not part of this release.
