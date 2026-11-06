"""Experiment E5: context-length sweep on Llama-3.1-8B (H1 and H3: the KV-cache bottleneck).

    uv run python experiments/e5_context.py

Once weights stop moving, every token still reads the whole KV cache. This sweeps context from
512 to 128k tokens for:
  - the molecular-scaled card, attention on the card, FP16 and 8-bit KV cache;
  - the RTX 5050 Laptop through the E2-validated digital model (weights + KV per token / measured
    bandwidth x utilisation), flagging contexts whose KV cache no longer fits in 8 GB.
Also reports H3's offload fraction (share of per-token work on the card when the GPU does attention).
All values estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import RESULTS, context, save  # noqa: E402
from molsim.cim.system import token_costs  # noqa: E402
from molsim.llm import Model  # noqa: E402
from molsim.params import load  # noqa: E402
from molsim.scenarios import card  # noqa: E402

CONTEXTS = [512, 4096, 32768, 131072]
TAG = "llama3.1:8b-instruct-q4_K_M"


def main() -> None:
    e2 = json.loads((RESULTS / "E2_digital.json").read_text())
    m = Model.load("llama-3.1-8b")
    sysp = load("systems")["rtx5050_laptop"]
    bw = sysp["memory_bandwidth"].q.to("GB/s").magnitude
    vram_gb = sysp["vram"].q.to("GB").magnitude
    util = sum(x["utilisation"] for x in e2["models"].values()) / len(e2["models"])
    w_gb = e2["models"][TAG]["weight_GB_per_token"]

    rows = []
    for ctx in CONTEXTS:
        row = {"context": ctx}
        for kv_b, tag in [(2, "fp16"), (1, "int8")]:
            r = token_costs(m, card("molecular_scaled", kv_bytes_per_value=kv_b), ctx)
            e, t = r["energy_parts_mJ"], r["latency_parts_ms"]
            row[f"cim_{tag}"] = {
                "tok_s": r["tokens_per_s"].m, "mJ_tok": r["energy"].m,
                "kv_energy_share": e["kv_cache"] / r["energy"].m, "kv_latency_share": t["kv_cache"] / r["latency"].m,
            }
        kv_gb = m.kv_bytes_per_token(2) * ctx / 1e9
        row["gpu_fp16"] = {"tok_s": bw * util / (w_gb + kv_gb), "kv_GB": kv_gb, "fits_in_vram": w_gb + kv_gb <= vram_gb}
        w_ops, a_ops = 2 * m.streamed_params(), 4 * m.num_layers * m.num_heads * m.head_dim * ctx
        row["offload_fraction_gpu_attention"] = w_ops / (w_ops + a_ops)
        rows.append(row)

    payload = {"experiment": "E5", "title": "Context-length sweep on Llama-3.1-8B: the KV-cache bottleneck", "label": "estimated", "context": context(), "rows": rows}
    path = save("E5_context", payload)

    print("Llama-3.1-8B, molecular-scaled card (attention on card) vs RTX 5050 Laptop (digital model)\n")
    print(f"{'context':>8} | {'card FP16 tok/s':>15} {'mJ/tok':>7} {'KV share E/t':>13} | {'card INT8 tok/s':>15} {'mJ/tok':>7} | {'GPU tok/s':>9} {'KV GB':>6} {'fits 8GB':>8} | {'H3 offload':>10}")
    for r in rows:
        a, b, g = r["cim_fp16"], r["cim_int8"], r["gpu_fp16"]
        print(
            f"{r['context']:>8} | {a['tok_s']:15.0f} {a['mJ_tok']:7.1f} {100 * a['kv_energy_share']:5.0f}%/{100 * a['kv_latency_share']:3.0f}%"
            f"  | {b['tok_s']:15.0f} {b['mJ_tok']:7.1f} | {g['tok_s']:9.1f} {g['kv_GB']:6.2f} {'yes' if g['fits_in_vram'] else 'NO':>8} | {100 * r['offload_fraction_gpu_attention']:9.1f}%"
        )
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
