"""E11: do real question-answering scores hold on the emulated molecular chip? (additions 4, 5, 6, 8)

Runs inside WSL2 (see setup_wsl.sh), or on Colab for 7B and larger:
    python accuracy/bench_tasks.py --model Qwen/Qwen2.5-1.5B-Instruct --mode digital
    python accuracy/bench_tasks.py --model Qwen/Qwen2.5-1.5B-Instruct --mode chip --head analog --seed 0

Tasks (standard multiple-choice scoring: the answer whose continuation the model finds most likely wins):
  ARC-Easy, ARC-Challenge   grade-school science exam questions (allenai/ai2_arc, test split)
  HellaSwag                 common-sense sentence endings (Rowan/hellaswag, validation split), length-normalised
  MMLU                      57-subject general-knowledge exam (cais/mmlu "all", test split), letter answers
  WikiText-2                perplexity on 32 windows of 512 tokens (continuity with E7)
A fixed random subset of each task (same questions for every run) keeps a run within minutes.

Chip mode: every linear layer, and with --head analog also the output head, is programmed onto emulated
molecular tiles (chip_generate.ChipLinear: 28 levels at 10 uS, 42 nS write error, 1% read noise, 8-bit in,
12-bit out, readout noise 0.02, per-column gain). No fine-tuning: the models are used as published.
Each seed draws new device errors (programming and read noise), so seeds show how much results vary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chip_generate as cg  # noqa: E402
import gpt2_common as g  # noqa: E402

N = {"arc_easy": 400, "arc_challenge": 300, "hellaswag": 400, "mmlu": 400}
OUT = g.SIM / "results" / "E11_tasks"


def pick(ds, n: int):
    idx = list(range(len(ds)))
    random.Random(1234).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def items(task: str) -> list[dict]:
    """Each item: context, list of continuations, gold index, normalise-by-length flag."""
    out = []
    if task in ("arc_easy", "arc_challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy" if task == "arc_easy" else "ARC-Challenge", split="test")
        for r in pick(ds, N[task]):
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            out.append({"ctx": f"Question: {r['question']}\nAnswer:", "conts": [" " + t for t in r["choices"]["text"]], "gold": labels.index(r["answerKey"]), "norm": True})
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for r in pick(ds, N[task]):
            ctx = (r["ctx_a"] + " " + r["ctx_b"].capitalize()).strip()
            out.append({"ctx": f"{r['activity_label']}: {ctx}", "conts": [" " + e.strip() for e in r["endings"]], "gold": int(r["label"]), "norm": True})
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        for r in pick(ds, N[task]):
            q = r["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"])) + "\nAnswer:"
            out.append({"ctx": f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n{q}", "conts": [" A", " B", " C", " D"], "gold": int(r["answer"]), "norm": False})
    return out


@torch.no_grad()
def score(model, tok, item: dict) -> int:
    ctx_ids = tok(item["ctx"], add_special_tokens=False).input_ids
    seqs, spans = [], []
    for c in item["conts"]:
        cont_ids = tok(c, add_special_tokens=False).input_ids
        seqs.append((ctx_ids + cont_ids)[-1024:])
        spans.append(len(cont_ids))
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=g.DEV)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=g.DEV)
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
    scores = []
    for k, (s, n) in enumerate(zip(seqs, spans)):
        tgt = torch.tensor(s[len(s) - n:], device=g.DEV)
        lp = logp[k, len(s) - n - 1:len(s) - 1].gather(1, tgt[:, None]).sum().item()
        scores.append(lp / max(1, len(item["conts"][k])) if item["norm"] else lp)
    return int(max(range(len(scores)), key=scores.__getitem__))


@torch.no_grad()
def wikitext_ppl(model, tok, windows: int = 32, width: int = 512) -> float:
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: windows * width].view(windows, width)
    nll, n = 0.0, 0
    for i in range(0, windows, 4):
        x = ids[i:i + 4].to(g.DEV)
        loss = model(input_ids=x, labels=x).loss.float().item()
        nll += loss * x.numel()
        n += x.numel()
    return math.exp(nll / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["digital", "chip"], required=True)
    ap.add_argument("--head", choices=["digital", "analog"], default="analog")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout-noise", type=float, default=0.02)
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,mmlu,wikitext")
    ap.add_argument("--n", type=int, default=0, help="questions per task (0 = the defaults in N)")
    ap.add_argument("--ideal-devices", action="store_true", help="diagnostic: exact weights, no device errors")
    ap.add_argument("--perfect-converters", action="store_true", help="diagnostic: no input/output rounding or readout noise")
    ap.add_argument("--tile-scaling", action="store_true", help="fix: scale each group of 256 inputs separately")
    ap.add_argument("--input-bits", type=int, default=0, help="input converter bits (0 = default 8)")
    ap.add_argument("--output-bits", type=int, default=0, help="output converter bits (0 = default 12)")
    ap.add_argument("--rows", type=int, default=0, help="tile rows (0 = default 256)")
    ap.add_argument("--out-bound", type=float, default=0.0, help="output converter full scale (0 = default 12)")
    ap.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16",
                    help="model number format; auto = bfloat16 where the GPU supports it, else float16 (e.g. a Colab T4)")
    ap.add_argument("--tag", default="", help="extra name for the result file")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.n:
        for k in N:
            N[k] = args.n
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.dtype == "float16" or (args.dtype == "auto" and not torch.cuda.is_bf16_supported()) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    info = {}
    if args.mode == "chip":
        switches = {"ideal_devices": args.ideal_devices, "perfect_converters": args.perfect_converters, "tile_scaling": args.tile_scaling,
                    "input_bits": args.input_bits or None, "output_bits": args.output_bits or None,
                    "rows": args.rows or None, "out_bound": args.out_bound or None}
        layers, devices = cg.to_chip(model, args.readout_noise, analog_head=(args.head == "analog"), **switches)
        info = {"layers_on_chip": layers, "devices": devices, **switches}
    model.to(g.DEV).eval()
    res = {"model": args.model, "mode": args.mode, "n_per_task": args.n or None, "tag": args.tag, "head": args.head if args.mode == "chip" else None, "seed": args.seed,
           "readout_noise": args.readout_noise if args.mode == "chip" else None, **info, "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype).replace("torch.", ""),
           "date": datetime.now().astimezone().isoformat(timespec="seconds"), "tasks": {}}
    for task in args.tasks.split(","):
        t1 = time.time()
        if task == "wikitext":
            res["tasks"][task] = {"perplexity": wikitext_ppl(model, tok), "seconds": time.time() - t1}
        else:
            its = items(task)
            correct = sum(score(model, tok, it) == it["gold"] for it in its)
            res["tasks"][task] = {"accuracy": correct / len(its), "n": len(its), "seconds": time.time() - t1}
        print(f"{args.model} {args.mode} head={res['head']} seed={args.seed} {task}: {res['tasks'][task]}", flush=True)
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.mode}" + (f"_{args.head}head_s{args.seed}" if args.mode == "chip" else "") + (f"_{args.tag}" if args.tag else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
