# The simulator (molsim)

All quantities are `Value` objects (`molsim/labeled.py`): a number with a unit (via pint), a label (measured, published, estimated, assumed), a source and an optional range. Arithmetic propagates units and keeps the weakest label, so every output says what kind of evidence it rests on.

## 1. Digital model (`molsim/digital.py`)

Decode at batch size one streams the weights once per token:

- memory time = bytes per token / (bandwidth x memory utilisation)
- compute time = operations per token / (peak throughput x compute utilisation)
- time per token = max(memory time, compute time); energy = power x time

Utilisation is calibrated on one measured model and applied to the other (cross-validation). Result (E2): 3B predicted 70.8 tok/s against 75.6 measured (-6.3%), 8B predicted 56.4 against 52.8 (+6.7%). A pure roofline is 40 to 49% optimistic.

## 2. One crossbar tile (`molsim/cim/tile.py`)

A tile has `rows x cols` weights, each stored on `devices_per_weight` devices, read with pulse-width (PWM) or bit-serial inputs and converted by `adcs` converters shared across columns.

**Latency per matrix-vector product (MVM)**

- input time = 2^(input bits - 1) x t_lsb for signed PWM inputs, or input bits x t_step for bit-serial inputs
- MVM latency = phases x (input time + ceil(cols / adcs) x t_conv + overhead)

**Energy per MVM**

- array energy = phases x devices x V_read^2 x G_mean x (duty x input time)
- periphery energy = phases x cols x E_out x bit scaling, where E_out is the energy per output calibrated on HERMES (48.4 pJ at 12 bits) and the bit scaling is "walden" (2^(bits - 12)) or "linear" (bits / 12)

**Area**

- array area = devices x pitch^2
- periphery area = adcs x A_adc + cols x A_column
- tile area = max(array, periphery) when the devices are stacked on CMOS, otherwise array + periphery

**IR drop (first order)**

Each active row injects about V x G x duty of current; the current in the k-th wire segment of a column is k times that, so the summed drop is about

    drop / V = rows^2 / 2 x G_mean x duty x r_wire

Above 10% of the read voltage a design is flagged as failing.

## 3. Weight mapping and precision (`molsim/cim/mapping.py`, `precision.py`)

Each linear layer of a model is split into tiles; weights wider than one device's precision are split into slices (columns). `precision.molecular_slices(bits, window)` returns how many slices a molecular window needs for a given weight precision, from the level spacing.

## 4. A whole accelerator per token (`molsim/cim/system.py`)

For one generated token the system model sums:

- every tile's MVM latency and energy (layers in sequence, tiles within a layer in parallel);
- digital work for attention, normalisation and activation functions;
- KV-cache reads and writes in off-chip DRAM (and optionally an on-chip SRAM tier);
- static power over the token time, on-chip overheads, and host-link traffic.

It reports tokens/s, mJ per token with a breakdown, silicon area in mm2 and reticle-size dies, the share of work offloaded from the GPU, and the time to install a model.

## 5. Sweeps (`molsim/sweep.py`)

`grid()` builds design grids, `evaluate()` runs the system model on one design, and `pareto()` keeps designs not dominated on the chosen columns. E10 sweeps 4,800 designs; E12 draws 5,000 random designs with every estimated or assumed input taken from its range (log-uniform for ranges wider than three times).

## 6. Validation

| Check | Error |
|---|---|
| Digital model vs measured GPU (8B) | +6.7% |
| HERMES one-phase and four-phase MVM latency | -3.8%, -1.5% |
| HERMES four-phase MVM energy | +1.8% |
| IBM 34-tile chip efficiency | -9.5% |
| NeuRRAM energy with the pulse-width form | -33.8% (NeuRRAM is bit-serial; the bit-serial form fits within 13% on average) |
| Analog layer vs AIHWKit on GPT-2 | within 1% |
| NeuroSim SAR converter area and energy | below our assumptions (ours conservative) |
