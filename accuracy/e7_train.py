"""E7 step 1: fine-tune GPT-2 small on WikiText-2 train, digitally (fair baseline) or noise-aware (HWA).

    python accuracy/e7_train.py --mode digital
    python accuracy/e7_train.py --mode hwa
    python accuracy/e7_train.py --mode hwa --steps 20 --smoke

Both runs see the same data order, steps and learning rate, so any difference comes from the noise.
Noise-aware training runs every attention and MLP layer through FastAnalogLinear with weight noise
and readout noise (gpt2_common.py). Checkpoints: ~/molsim-acc/ckpt; log: results/E7_train_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import gpt2_common as g

SEQ, MICRO, ACCUM, LR, WARMUP, STEPS = 512, 4, 2, 5e-5, 100, 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["digital", "hwa"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)

    train, test = g.tokens("train"), g.tokens("test")
    n_seq = len(train) // SEQ
    data = train[: n_seq * SEQ].view(n_seq, SEQ)

    model = g.load_model()
    hwa = g.AnalogSettings(out_noise=g.HWA_OUT_NOISE, weight_noise=g.HWA_WEIGHT_NOISE)
    if args.mode == "hwa":
        model = g.analogize(model, hwa)
    model = model.to(g.DEV)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / args.steps))

    log, t0, eval_every = [], time.time(), (10 if args.smoke else 250)
    gen = torch.Generator().manual_seed(0)
    order, pos = torch.randperm(n_seq, generator=gen), 0
    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        loss_sum = 0.0
        for _ in range(ACCUM):
            if pos + MICRO > n_seq:
                order, pos = torch.randperm(n_seq, generator=gen), 0
            x = data[order[pos : pos + MICRO]].to(g.DEV)
            pos += MICRO
            loss = model(input_ids=x, labels=x).loss / ACCUM
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0 or step == 0:
            model.eval()
            ppl = g.perplexity(model, test, 16)  # quick check under the training conditions; full evaluation is e7_eval.py
            model.train()
            log.append({"step": step + 1, "train_loss": loss_sum, "test_ppl_quick": ppl, "seconds": time.time() - t0})
            print(f"{args.mode} step {step + 1:5d}  loss {loss_sum:.3f}  quick test ppl {ppl:7.2f}  {time.time() - t0:6.0f} s", flush=True)

    g.CKPT.mkdir(parents=True, exist_ok=True)
    name = f"{args.mode}{'_smoke' if args.smoke else ''}"
    torch.save({k: v for k, v in model.state_dict().items() if not k.endswith("device_weight")}, g.CKPT / f"{name}.pt")
    out = {"mode": args.mode, "model": g.MODEL, "steps": args.steps, "seq": SEQ, "batch": MICRO * ACCUM, "lr": LR, "warmup": WARMUP,
           "tokens_seen": args.steps * MICRO * ACCUM * SEQ, "hwa_out_noise": g.HWA_OUT_NOISE if args.mode == "hwa" else None,
           "hwa_weight_noise": g.HWA_WEIGHT_NOISE if args.mode == "hwa" else None, "analog_head": False,
           "gpu": torch.cuda.get_device_name(0), "seconds": time.time() - t0, "log": log}
    (g.SIM / "results" / f"E7_train_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {g.CKPT / (name + '.pt')}")


if __name__ == "__main__":
    main()
