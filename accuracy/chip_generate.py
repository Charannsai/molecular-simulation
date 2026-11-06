"""Generate real text through emulated molecular crossbars , or normally (digital) for comparison.

Runs inside WSL2 Ubuntu (see setup_wsl.sh):
    python accuracy/chip_generate.py --mode chip       # every weight on molecular tiles
    python accuracy/chip_generate.py --mode digital    # the same model, normal digital weights

Model: Qwen2.5-1.5B-Instruct (a real chat model; Llama-3.1-8B needs about 32 GB once every weight becomes
a conductance, the laptop GPU has 8 GB). Chip mode programs each linear layer once onto 256-row tiles with
the molecular device model of E7 (28 levels at a 10 uS window at the published 0.35 uS spacing, 42 nS
programming error, 1% read noise, no drift), per-column digital gain, global drift compensation; every
forward pass then goes through 8-bit input conversion, per-tile analog sums, readout noise and 12-bit output
converters (the E7 FastAnalogLinear physics, validated against AIHWKit within 1%). Embeddings, norms,
attention and the tied output head stay digital. Decoding is greedy (temperature 0).
Chip time and energy for the same token counts come from the CIM system model (molsim).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gpt2_common as g  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SIM_KEY = "qwen2.5-1.5b-instruct"
PROMPT = ("In three short paragraphs, explain why running a large language model on a laptop is slow, "
          "and how keeping the model's weights inside the memory chip could make it faster.")
WINDOW_US, READ_REL, PROG_ABS_US = 10.0, 0.01, 0.042
OUT = g.SIM / "results"
B, Y, G, O, D, R = "\033[1m", "\033[93m", "\033[92m", "\033[38;5;208m", "\033[90m", "\033[0m"
WIDTH = 120


class WrapStreamer(TextStreamer):
    """Print tokens as they are generated, wrapping at word boundaries (a console window would split words)."""

    def __init__(self, tok, width: int = WIDTH):
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self.width, self.col = width, 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        for i, part in enumerate(text.split("\n")):
            if i:
                print(flush=True)
                self.col = 0
            for word in part.split(" "):
                if not word:
                    continue
                gap = 1 if self.col else 0
                if self.col + gap + len(word) > self.width:
                    print(flush=True)
                    self.col, gap = 0, 0
                print(" " * gap + word, end="", flush=True)
                self.col += gap + len(word)
        if stream_end:
            print(flush=True)
            self.col = 0


class ChipLinear(nn.Module):
    """A linear layer programmed once onto emulated molecular tiles; analog MVM per tile on every call.

    Diagnostic switches (E11 diagnostics): ideal_devices stores exact weights (no levels, write or read error);
    perfect_converters skips input and output rounding and readout noise; tile_scaling scales each group of
    256 inputs by its own maximum instead of the whole input vector (a small digital change)."""

    def __init__(self, lin: nn.Linear, noise_model, out_noise: float, ideal_devices: bool = False, perfect_converters: bool = False, tile_scaling: bool = False,
                 input_bits: int | None = None, output_bits: int | None = None, rows: int | None = None, out_bound: float | None = None):
        super().__init__()
        self.rows = rows or g.ROWS  # tile rows (256 by default; taller tiles reduce periphery area)
        self.out_bound = out_bound or g.OUT_BOUND  # output converter full scale, in units of one full-scale input times weight
        self.input_bits = input_bits or g.INPUT_BITS
        self.output_bits = output_bits or g.OUTPUT_BITS
        n_out, n_in = lin.weight.shape
        self.n_in, self.n_out, self.out_noise = n_in, n_out, out_noise
        self.perfect_converters, self.tile_scaling = perfect_converters, tile_scaling
        self.tiles = math.ceil(n_in / self.rows)
        w = lin.weight.detach().to(g.DEV, torch.float32)
        wt = nn.functional.pad(w, (0, self.tiles * self.rows - n_in)).view(n_out, self.tiles, self.rows).transpose(0, 1)
        scale = wt.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)  # per-column digital gain
        wn = wt / scale
        if ideal_devices:
            self.register_buffer("tw", wn.to(torch.float16))
        else:
            prog, nu = noise_model.apply_programming_noise(wn.reshape(-1, self.rows))
            read = noise_model.apply_drift_noise(prog, nu, 1.0).view_as(wn)
            alpha = prog.view_as(wn).abs().mean(dim=(1, 2), keepdim=True) / read.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            self.register_buffer("tw", (read * alpha).to(torch.float16))
            del prog, read
        self.register_buffer("scale", scale.squeeze(-1).to(torch.float16))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(g.DEV, torch.float32), requires_grad=False)
        del w, wt, wn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        x = x.reshape(-1, self.n_in).float()
        # chunk the tokens so the per-tile partial sums of a wide layer (e.g. a 151,936-word output head) fit in memory
        rows = max(1, int(1.5e8 // (self.tiles * self.n_out)))
        if x.shape[0] > rows:
            return torch.cat([self._mvm(x[i:i + rows]) for i in range(0, x.shape[0], rows)]).to(dtype).view(*shape[:-1], self.n_out)
        return self._mvm(x).to(dtype).view(*shape[:-1], self.n_out)

    def _mvm(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (0, self.tiles * self.rows - self.n_in)).view(-1, self.tiles, self.rows)
        if self.tile_scaling:  # one scale per tile (per group of 256 inputs)
            xs = x.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
        else:  # one scale for the whole input vector
            xs = x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        xq = x / xs
        if not self.perfect_converters:
            lv = 2 ** (self.input_bits - 1) - 1
            xq = torch.round(xq * lv) / lv  # input converters (8-bit by default)
        y = torch.einsum("nti,toi->nto", xq, self.tw.float())  # analog sums, one per tile and column
        if not self.perfect_converters:
            if self.out_noise:
                y = y + self.out_noise * torch.randn_like(y)
            y = y.clamp(-self.out_bound, self.out_bound)
            step = 2 * self.out_bound / (2**self.output_bits - 2)
            y = torch.round(y / step) * step  # output converters (12-bit by default)
        y = (y * self.scale.float().unsqueeze(0) * xs).sum(dim=1)
        if self.bias is not None:
            y = y + self.bias
        return y


def to_chip(model: nn.Module, out_noise: float, analog_head: bool = False, noise_model=None, **switches) -> tuple[int, int]:
    """Replace every linear layer (and the output head if analog_head) by molecular tiles; returns (layers, devices).

    noise_model: a factory for the device model; default is the molecular model of E7 (10 uS window)."""
    levels = int((WINDOW_US - 0.4) / 0.35 + 1)
    make = noise_model or (lambda: g.MolecularNoiseModel(WINDOW_US, levels, PROG_ABS_US / WINDOW_US, READ_REL))
    layers = devices = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and (analog_head or cname != "lm_head"):
                if child.weight.device.type != "cuda":
                    child.to(g.DEV)
                nm = make()
                setattr(mod, cname, ChipLinear(child, nm, out_noise, **switches))
                devices += 2 * math.ceil(child.in_features / g.ROWS) * g.ROWS * child.out_features
                layers += 1
                del child
                torch.cuda.empty_cache()
    return layers, devices


def chip_estimate(n_prompt: int, n_gen: int) -> dict:
    from molsim.cim.system import token_costs
    from molsim.labeled import Value
    from molsim.llm import Model
    from molsim.scenarios import card

    m = Model.load(SIM_KEY)
    c = card("molecular_scaled", rows=512, adc_share=16, periph_area=500.0, device_pitch=Value.of(100, "nm", label="assumed", source="design point", range=(100, 100)),
             g_max=10.0, weight_slices=1, kv_bytes_per_value=1)
    t_pre = t_gen = e = e_gen = 0.0
    for ctx in range(1, n_prompt + n_gen + 1):
        r = token_costs(m, c, ctx)
        t, en = r["latency"].q.to("s/tok").magnitude, r["energy"].q.to("J/tok").magnitude
        e += en
        if ctx <= n_prompt:
            t_pre += t
        else:
            t_gen, e_gen = t_gen + t, e_gen + en
    first = token_costs(m, c, 1)["mapping"]
    return {"prefill_s": t_pre, "generate_s": t_gen, "total_s": t_pre + t_gen, "energy_J": e, "J_per_token": e_gen / n_gen, "tiles": first["tiles"]}


def fmt_t(s: float) -> str:
    return f"{s * 1e6:.1f}µs" if s < 1e-3 else (f"{s * 1e3:.4f}ms" if s < 1 else f"{s:.6f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chip", "digital"], required=True)
    ap.add_argument("--readout-noise", type=float, default=0.02)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    gpu = torch.cuda.get_device_name(0)
    print(f"Date   : {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}")
    if args.mode == "chip":
        t0 = time.time()
        model.model.to(g.DEV)  # body to GPU in bf16; linear layers are then reprogrammed as tiles
        model.lm_head.to(g.DEV)
        layers, devices = to_chip(model, args.readout_noise)
        prog_s = time.time() - t0
        print(f"Chip   : molecular compute-in-memory tiles {O}(device physics emulated on this laptop's {gpu}){R}")
        print(f"Devices: {WINDOW_US:g} uS window, {int((WINDOW_US - 0.4) / 0.35 + 1)} levels, 42 nS write error, 1% read noise, 8-bit in / 12-bit out converters,")
        print(f"         readout noise {args.readout_noise:g}, 256-row tiles, per-column gain")
        print(f"Model  : {MODEL}: {layers} weight matrices programmed onto {devices:,} emulated molecular devices in {prog_s:.1f} s")
    else:
        model.to(g.DEV)
        print(f"Chip   : normal digital weights (bf16) on {gpu}")
        print(f"Model  : {MODEL}")
    model.eval()
    print()
    print(f"{Y}$ python accuracy/chip_generate.py --mode {args.mode}{R}")
    import textwrap
    print(textwrap.fill(f"Prompt : {PROMPT}", WIDTH, subsequent_indent="         "))
    print()
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(g.DEV)
    streamer = WrapStreamer(tok)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, streamer=streamer, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    wall = time.time() - t0
    n_prompt, n_gen = ids.shape[1], out.shape[1] - ids.shape[1]
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print()
    print(f"prompt tokens:        {n_prompt}")
    print(f"generated tokens:     {n_gen}")
    est = chip_estimate(n_prompt, n_gen)
    if args.mode == "chip":
        print(f"{B}On the molecular chip (chip model estimate, best design, {est['tiles']:,} tiles):{R}")
        print(f"total duration:       {fmt_t(est['total_s'])}")
        print(f"eval rate:            {n_gen / est['generate_s']:,.2f} tokens/s")
        print(f"energy:               {est['energy_J']:.4f} J ({1e3 * est['J_per_token']:.3f} mJ per generated token)")
        print(f"{D}Emulating the device physics on the laptop took {wall:.1f} s ({n_gen / wall:.1f} tokens/s); that is the emulator, not the chip.{R}")
    else:
        print(f"{B}Measured on the {gpu} (PyTorch, bf16):{R}")
        print(f"total duration:       {wall:.3f}s")
        print(f"eval rate:            {n_gen / wall:,.2f} tokens/s")
    print()
    print(f"{G}Finished: {datetime.now():%H:%M:%S}{R}")
    rec = {"mode": args.mode, "model": MODEL, "prompt": PROMPT, "readout_noise": args.readout_noise, "seed": args.seed, "greedy": True,
           "prompt_tokens": n_prompt, "generated_tokens": n_gen, "wall_s": wall, "text": text, "chip_estimate": est, "gpu": gpu,
           "date": datetime.now().astimezone().isoformat(timespec="seconds")}
    (OUT / f"V9_generate_{args.mode}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    done = os.environ.get("V9_DONE")
    if done:
        Path(done).write_text("done")


if __name__ == "__main__":
    main()
