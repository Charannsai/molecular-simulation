# Parameter registers

Every input lives in a YAML file under `params/`, loaded with `molsim.params.load(name)`:

| File | Contents |
|---|---|
| `devices.yaml` | molecular, PCM and RRAM devices: levels, window, programming and read noise, write pulses, pitch, retention |
| `chips.yaml` | published chips used for validation: IBM HERMES, NeuRRAM, IBM 34-tile |
| `circuits.yaml` | links, memory (LPDDR5X, HBM3, SRAM), converter survey points, converter and digital energy |
| `cards.yaml` | accelerator scenarios used in the comparisons, and their common circuit settings |
| `models.yaml` | LLM shapes: layers, hidden size, heads, parameters (Llama, Qwen, GPT-2, MoE models) |
| `systems.yaml` | digital systems: the measured laptop GPU, RTX 4090, H200, Ryzen AI Max+, Apple M4 Max |

## Format

```yaml
g_max:
  {value: 6, unit: "mS", label: published, source: "Sharma 2024, Fig. 1b, 2a, 2c", note: "read from plot (verify)"}
device_pitch:
  {value: 100, unit: "nm", label: assumed, source: "design assumption", range: [50, 1100]}
```

- `label` is one of measured, published, estimated or assumed.
- `source` names where the number comes from; "(verify)" marks values read from a figure or a secondary source.
- `range` is required for assumed and estimated values that the sweeps explore.
- `ref: "devices.molecular.g_max"` points to another entry instead of repeating it.

## Changing a parameter

Edit the YAML and rerun the experiment. The loader caches files by modification time, so a running session picks up changes. To try a value without editing files, pass an override to `molsim.scenarios.card(...)`, for example `card("molecular_scaled", rows=512, g_max=50.0)`.

## Most influential assumptions

| Parameter | Value used | Range | Why it matters |
|---|---|---|---|
| Periphery area per output | 5,438 um2 (HERMES) down to 500 um2 | 250 to 5,438 | sets silicon area |
| Periphery energy per output | 48.4 pJ (HERMES calibration) | 48.4 to 73 | converter energy |
| Molecular window | 10 uS (scaled) | 1 to 6,000 | read energy, IR drop, levels |
| Molecular pitch | 100 nm | 50 to 1,100 | array area |
| Wire resistance per cell | 0.46 ohm | 0.1 to 5 | IR drop |
| KV-cache energy per bit | 5 pJ | 2 to 10 | dominant energy at long context |
| Readout noise | 0.02 | 0 to 0.06 | accuracy |
