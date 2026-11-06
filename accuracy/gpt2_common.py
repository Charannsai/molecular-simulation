"""Shared pieces for E7 on GPT-2 small (run inside WSL2 Ubuntu, see setup_wsl.sh).

Why a vectorised analog layer: AIHWKit simulates every 256 x 256 tile with its own Python call, which
leaves the GPU idle (about 33 s per training step for GPT-2 here). `FastAnalogLinear` computes all
tiles of a layer in one batched multiply but applies the same non-idealities per tile:
  1. inputs scaled per token by their absolute maximum (digital, as AIHWKit's ABS_MAX noise
     management) and quantised to 8-bit signed;
  2. weights stored per 256-row tile, normalised per column to the device range [-1, 1];
  3. each tile's partial output gets Gaussian readout noise (`out_noise`, in units of that range),
     is clipped to +-out_bound and quantised by a 12-bit signed converter;
  4. partial outputs are rescaled and summed digitally; bias is digital.
Device physics comes from AIHWKit's own fitted noise models (PCM, HERMES, RRAM) or our molecular
model, applied to the normalised tile weights (programming noise, drift, read noise), followed by
global drift compensation per tile, as AIHWKit does. `e7_validate.py` checks this layer against
AIHWKit itself on the same model.

Every attention and MLP layer is analog; the output head (50,257 x 768) stays digital (stated limitation).
Noise-aware (hardware-aware) training: weight noise and readout noise in the forward pass, with
straight-through gradients through the quantisers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from aihwkit.inference import StateIndependentNoiseModel
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

MODEL = "gpt2"  # GPT-2 small, 124M parameters
WINDOW = 512
SIM = Path(__file__).resolve().parent.parent
HOME = Path.home() / "molsim-acc"
CACHE, CKPT = HOME / "cache", HOME / "ckpt"
DEV = "cuda"
DIGITAL = ["lm_head"]  # exclude_modules for AIHWKit conversion

ROWS = 256  # tile rows (params/cards.yaml)
INPUT_BITS, OUTPUT_BITS, OUT_BOUND = 8, 12, 12.0
HWA_OUT_NOISE = 0.02  # readout noise during noise-aware training (assumed; evaluation sweeps it)
HWA_WEIGHT_NOISE = 0.03  # Gaussian weight noise during training, share of each column's max


@dataclass
class AnalogSettings:
    out_noise: float = 0.0
    weight_noise: float = 0.0  # training only
    input_bits: int = INPUT_BITS
    output_bits: int = OUTPUT_BITS
    out_bound: float = OUT_BOUND


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


class FastAnalogLinear(nn.Module):
    def __init__(self, lin: nn.Linear, cfg: AnalogSettings):
        super().__init__()
        self.weight = lin.weight  # [out, in], digital master copy
        self.bias = lin.bias
        self.cfg = cfg
        self.n_in, self.n_out = lin.in_features, lin.out_features
        self.tiles = math.ceil(self.n_in / ROWS)
        self.register_buffer("device_weight", None, persistent=False)  # programmed normalised tiles, or None

    def tiled(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[out, in] -> normalised tiles [T, out, ROWS] and per-tile column scales [T, out, 1]."""
        pad = self.tiles * ROWS - self.n_in
        wt = nn.functional.pad(w, (0, pad)).view(self.n_out, self.tiles, ROWS).transpose(0, 1)
        scale = wt.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
        return wt / scale, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        shape = x.shape
        x = x.reshape(-1, self.n_in)
        # inputs: per-token abs-max scaling, signed quantisation
        xs = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).detach()
        lv = 2 ** (c.input_bits - 1) - 1
        xq = _ste_round(x / xs * lv) / lv
        xq = nn.functional.pad(xq, (0, self.tiles * ROWS - self.n_in)).view(-1, self.tiles, ROWS)
        # weights: normalised tiles (programmed device copy at inference, if any)
        wn, scale = self.tiled(self.weight)
        if self.device_weight is not None:
            wn = self.device_weight
        elif self.training and c.weight_noise:
            wn = wn + c.weight_noise * torch.randn_like(wn)
        # per-tile analog MVM, readout noise, converter
        y = torch.einsum("nti,toi->nto", xq, wn)
        if c.out_noise:
            y = y + c.out_noise * torch.randn_like(y)
        y = y.clamp(-c.out_bound, c.out_bound)
        step = 2 * c.out_bound / (2**c.output_bits - 2)
        y = _ste_round(y / step) * step
        y = (y * scale.squeeze(-1).unsqueeze(0)).sum(dim=1) * xs
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.n_out)

    @torch.no_grad()
    def program(self, noise_model, t_inference: float) -> None:
        """Write the normalised tiles to devices, let them drift to t_inference, compensate per tile."""
        wn, _ = self.tiled(self.weight)
        flat = wn.reshape(-1, ROWS)
        prog, nu = noise_model.apply_programming_noise(flat)
        drifted = noise_model.apply_drift_noise(prog, nu, t_inference) if t_inference > 0 else prog
        prog, drifted = prog.view_as(wn), drifted.view_as(wn)
        alpha = prog.abs().mean(dim=(1, 2), keepdim=True) / drifted.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        self.device_weight = drifted * alpha

    def unprogram(self) -> None:
        self.device_weight = None


def analogize(model: nn.Module, cfg: AnalogSettings, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    for name, child in list(model.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, FastAnalogLinear(child, cfg))
        else:
            analogize(child, cfg, skip)
    return model


def analog_layers(model: nn.Module) -> list[FastAnalogLinear]:
    return [m for m in model.modules() if isinstance(m, FastAnalogLinear)]


def program(model: nn.Module, noise_model, t_inference: float) -> None:
    for m in analog_layers(model):
        m.program(noise_model, t_inference)


class MolecularNoiseModel(StateIndependentNoiseModel):
    """Levels, then Gaussian programming noise (share of g_max), then frozen read noise (share of g). No drift.

    Sharma 2024 shows no drift over 11 days and under 500 nS error after 7 months unpowered on a 6 mS window.
    """

    def __init__(self, g_max: float, levels: int | None, prog_rel: float, read_rel: float):
        super().__init__(g_max=g_max, prog_coeff=[prog_rel * g_max, 0.0, 0.0], drift_nu_mean=0.0, drift_nu_std=0.0)
        self.levels, self.prog_rel, self.read_rel = levels, prog_rel, read_rel

    @torch.no_grad()
    def apply_programming_noise_to_conductance(self, g_target):
        g = g_target
        if self.levels:
            step = self.g_max / (self.levels - 1)
            g = torch.round(g / step) * step
        if self.prog_rel:
            g = g + self.prog_rel * self.g_max * torch.randn_like(g)
        return g.clamp(min=0.0, max=self.g_max)

    @torch.no_grad()
    def apply_drift_noise_to_conductance(self, g_prog, drift_noise_param, t_inference):
        if not self.read_rel:
            return g_prog
        return (g_prog + self.read_rel * g_prog.abs() * torch.randn_like(g_prog)).clamp(min=0.0)


def to_linear(model: nn.Module) -> nn.Module:
    """Replace every Conv1D (GPT-2's layer type) by an nn.Linear computing the same thing."""
    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D):
            n_in, n_out = child.weight.shape
            lin = nn.Linear(n_in, n_out, bias=True)
            with torch.no_grad():
                lin.weight.copy_(child.weight.T)
                lin.bias.copy_(child.bias)
            setattr(model, name, lin)
        else:
            to_linear(child)
    return model


def load_model(checkpoint: str | None = None) -> GPT2LMHeadModel:
    m = GPT2LMHeadModel.from_pretrained(MODEL, dtype=torch.float32)
    m.config.tie_word_embeddings = False
    m.lm_head.weight = nn.Parameter(m.lm_head.weight.detach().clone())  # untie
    m = to_linear(m)
    if checkpoint:
        m.load_state_dict(torch.load(CKPT / f"{checkpoint}.pt", map_location="cpu"))
    return m


def tokens(split: str) -> torch.Tensor:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"wikitext2_{split}.pt"
    if not path.exists():
        from datasets import load_dataset

        tok = GPT2TokenizerFast.from_pretrained(MODEL)
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
        torch.save(tok(text, return_tensors="pt").input_ids[0], path)
    return torch.load(path)


def aihwkit_rpu(noise_model=None, out_noise: float = 0.0, columnwise: bool = True):
    """AIHWKit configuration matching FastAnalogLinear's inference settings (for validation)."""
    from aihwkit.inference import GlobalDriftCompensation
    from aihwkit.simulator.configs import TorchInferenceRPUConfig
    from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, WeightRemapType

    c = TorchInferenceRPUConfig()
    c.mapping.max_input_size = ROWS
    c.mapping.max_output_size = 256
    c.forward.inp_res = 1 / (2 ** (INPUT_BITS - 1) - 1)
    c.forward.out_res = 1 / (2**OUTPUT_BITS - 2)
    c.forward.out_bound = OUT_BOUND
    c.forward.out_noise = out_noise
    c.forward.noise_management = NoiseManagementType.ABS_MAX
    c.forward.bound_management = BoundManagementType.NONE
    c.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    if columnwise:  # each column uses the full conductance range; a digital scale per column undoes it
        c.mapping.weight_scaling_omega = 1.0
        c.mapping.weight_scaling_columnwise = True
    c.noise_model = noise_model or StateIndependentNoiseModel(g_max=25.0, prog_coeff=[0.0, 0.0, 0.0], drift_nu_mean=0.0, drift_nu_std=0.0, read_noise_scale=0.0)
    c.drift_compensation = GlobalDriftCompensation()
    return c


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, n_windows: int, batch: int = 8) -> float:
    n_windows = min(n_windows, len(ids) // WINDOW)
    windows = ids[: n_windows * WINDOW].view(n_windows, WINDOW)
    nll, count = 0.0, 0
    for i in range(0, n_windows, batch):
        x = windows[i : i + batch].to(DEV)
        out = model(input_ids=x, labels=x)
        n = x.numel() - x.shape[0]
        nll += out.loss.float().item() * n
        count += n
    return math.exp(nll / count)
