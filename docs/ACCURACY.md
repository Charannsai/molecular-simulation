# Accuracy on emulated analog tiles

All results here are emulated (estimated). They run on a CUDA GPU under Linux or WSL2 because IBM AIHWKit has no Windows build.

## The analog layer

`accuracy/gpt2_common.py` (`FastAnalogLinear`) and `accuracy/chip_generate.py` (`ChipLinear`) replace each linear layer with tiles of 256 rows (configurable with `--rows`). Per forward pass:

1. **Input scaling and conversion.** Inputs are divided by a scale and rounded to `input_bits` (8 by default). The scale is either one value per input vector (the straightforward design) or one value per tile, the maximum of the inputs entering that tile (`--tile-scaling`).
2. **Devices.** Weights are normalised per column and programmed once onto the device model: molecular by default (28 levels in a 10 uS window at the published 0.35 uS spacing, 42 nS programming error, 1% read noise, no drift), or AIHWKit's PCM and RRAM models.
3. **Analog sums.** Each tile produces one partial sum per column.
4. **Output conversion.** Readout noise (0.02 of full scale) is added, the sum is clipped to `out_bound` and rounded to `output_bits` (12 by default).
5. **Digital gain.** Per-column gain and the input scale are multiplied back and the tile sums are added.

Diagnostic switches: `--ideal-devices` stores exact weights; `--perfect-converters` removes input and output rounding and readout noise.

Validation: on GPT-2 the analog layer reproduces AIHWKit within 1% (converters only 42.24 vs 42.69 perplexity; readout noise 43.38 vs 43.00; PCM at 1 day 53.77 vs 53.71).

## E7: GPT-2 over time

`accuracy/e7_train.py` fine-tunes GPT-2 small on WikiText-2, digitally and with noise-aware training; `e7_eval.py` evaluates devices from 1 s to 1 year. Molecular devices stay within 2.4 to 4.6% of digital; PCM degrades by 24 to 285% after a year and RRAM by 328% after a day.

## E11: task benchmarks on Qwen2.5

`accuracy/bench_tasks.py` scores ARC-Easy, ARC-Challenge, HellaSwag and MMLU by log-likelihood multiple choice on fixed random subsets (seed 1234), and WikiText-2 perplexity on 32 windows of 512 tokens. Each `--seed` is a newly programmed chip.

```bash
python accuracy/bench_tasks.py --model Qwen/Qwen2.5-0.5B-Instruct --mode digital --n 200
python accuracy/bench_tasks.py --model Qwen/Qwen2.5-0.5B-Instruct --mode chip --head digital --tile-scaling --n 200
bash accuracy/run_e11.sh all      # every E11 run in the paper (skips runs whose results exist)
```

Sampling error is about plus or minus 3.5 points at 200 questions and plus or minus 1 point at 2,000.

### Diagnostics (Qwen2.5-0.5B, 200 questions)

| Run | Switch | ARC-Easy | MMLU | Perplexity |
|---|---|---|---|---|
| Digital | | 52.0% | 46.5% | 17.63 |
| Straightforward design | | 46.5% | 39.0% | 24.68 |
| Ideal devices | `--ideal-devices` | 46.5% | 35.0% | 23.82 |
| Ideal converters | `--perfect-converters` | 55.0% | 46.5% | 18.14 |
| Tile input scaling | `--tile-scaling` | 48.0% | 44.5% | 19.03 |

With ideal converters the molecular devices match digital; with ideal devices the loss remains. The converters cause the loss: a few large activations set the input scale and squeeze other inputs into a few steps. Tile-level scaling recovers most of it.

### Larger question sets and model sizes

| Model | Run | Questions | ARC-Easy | MMLU | Perplexity |
|---|---|---|---|---|---|
| 0.5B | digital | 2,000 | 60.0% | 45.8% | 17.63 |
| 0.5B | tile scaling | 2,000 | 57.7% | 44.5% | 19.07 |
| 0.5B | tile scaling, 512-row tiles | 2,000 | 58.1% | 42.7% | 19.45 |
| 1.5B | digital | 200 | 73.0% | 54.0% | 11.66 |
| 1.5B | straightforward | 200 | 58.0% | 51.0% | 14.87 |
| 1.5B | tile scaling | 200 | 72.5% | 53.5% | 12.30 |
| 3B | digital | 200 | 68.0% | 66.5% | 10.51 |
| 3B | straightforward | 200 | 40.0% | 32.5% | 190.0 |
| 3B | tile scaling (two chips) | 200 | 62.5 / 65.0% | 63.0 / 62.0% | 11.34 / 11.30 |

The 3B runs used a free Colab T4 (`accuracy/make_colab_bundle.py` builds the notebook and bundle).

### Converter resolution (tile scaling, Qwen2.5-0.5B)

| Inputs / outputs | 6/12 | 7/12 | 8/12 | 10/12 | 8/8 | 8/9 | 8/10 | 8/14 |
|---|---|---|---|---|---|---|---|---|
| Perplexity | 20.34 | 19.40 | 19.03 | 18.95 | 19.99 | 19.34 | 19.03 | 19.03 |

## E13: drift tolerance

`accuracy/e13_drift.py` adds power-law drift G(t) = G0 (t/t0)^-nu with device-to-device spread and per-tile global compensation to the noise-aware GPT-2 model. Loss after one year stays under 5% up to nu = 0.003 and under 10% up to nu = 0.01; PCM-like drift (nu about 0.06) gives +477%.

## Result files

`results/E11_tasks/<model>_<mode>[_<head>head_s<seed>]_<tag>.json` holds the task scores, settings and run time. The file names used in the paper: `d200`, `R_full`, `A_circuits_only`, `B_molecules_only`, `C_tile_scaling`, `X2000_*`, `Q15_*` (1.5B), `colab*` (3B), `BITS_*`, `R512_*`. Runs tagged `CD` include a lightweight digital correction whose training code is not part of this release.
