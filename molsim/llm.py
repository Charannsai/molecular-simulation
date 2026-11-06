"""LLM shapes: from a model's published config to matrices, parameters, bytes and work per token.

Covers Llama-style decoders (Llama 3.x, Qwen 2.5): grouped-query attention plus a gated (SwiGLU)
feed-forward block, and mixture-of-experts (MoE) variants where each layer has many expert
feed-forward blocks and a router picks a few per token (Qwen3-30B-A3B, gpt-oss-120b).

Decode at batch size 1, for each generated token:
- every layer's attention matrices are read once (matrix-vector products);
- for MoE only the router and the chosen experts are read; the rest sit idle;
- the output head (vocab x hidden) is read once;
- the token-embedding table is NOT streamed: only one row is looked up. With tied embeddings the
  same matrix is the output head, so it is read once as the head;
- the KV cache of all previous tokens is read once per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .labeled import Value
from .params import load


@dataclass(frozen=True)
class Matrix:
    name: str
    n_in: int  # input features (rows of the weight matrix as applied to the input vector)
    n_out: int  # output features
    expert: int | None = None  # MoE expert index, None for shared matrices

    @property
    def params(self) -> int:
        return self.n_in * self.n_out


@dataclass(frozen=True)
class Model:
    key: str
    source: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool
    reported_params: Value | None = None
    reported_active_params: Value | None = None
    qkv_bias: bool = False
    num_experts: int = 0  # 0 = dense
    experts_per_token: int = 0
    moe_intermediate_size: int = 0

    @classmethod
    def load(cls, key: str) -> "Model":
        spec = load("models")[key]
        return cls(
            key=key,
            source=spec["source"],
            hidden_size=spec["hidden_size"],
            intermediate_size=spec.get("intermediate_size", 0),
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            num_kv_heads=spec["num_kv_heads"],
            head_dim=spec["head_dim"],
            vocab_size=spec["vocab_size"],
            tied_embeddings=spec["tied_embeddings"],
            reported_params=spec.get("reported_params"),
            reported_active_params=spec.get("reported_active_params"),
            qkv_bias=spec.get("qkv_bias", False),
            num_experts=spec.get("num_experts", 0),
            experts_per_token=spec.get("experts_per_token", 0),
            moe_intermediate_size=spec.get("moe_intermediate_size", 0),
        )

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    # ───────── shapes ─────────

    def attention_matrices(self) -> list[Matrix]:
        d, kv, q = self.hidden_size, self.num_kv_heads * self.head_dim, self.num_heads * self.head_dim
        return [Matrix("q", d, q), Matrix("k", d, kv), Matrix("v", d, kv), Matrix("o", q, d)]

    def ffn_matrices(self, active_only: bool = False) -> list[Matrix]:
        d = self.hidden_size
        if not self.is_moe:
            f = self.intermediate_size
            return [Matrix("gate", d, f), Matrix("up", d, f), Matrix("down", f, d)]
        f = self.moe_intermediate_size
        experts = range(self.experts_per_token if active_only else self.num_experts)
        out = [Matrix("router", d, self.num_experts)]
        for e in experts:
            out += [Matrix("gate", d, f, e), Matrix("up", d, f, e), Matrix("down", f, d, e)]
        return out

    def layer_matrices(self, active_only: bool = False) -> list[Matrix]:
        """One layer's weight matrices. active_only: just what one token uses (MoE: router + chosen experts)."""
        return self.attention_matrices() + self.ffn_matrices(active_only)

    def head(self) -> Matrix:
        return Matrix("lm_head", self.hidden_size, self.vocab_size)

    # ───────── parameter counts (plain integers; exact for the given config) ─────────

    def _layer_params(self, active_only: bool) -> int:
        weights = sum(m.params for m in self.layer_matrices(active_only))
        norms = 2 * self.hidden_size
        bias = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim if self.qkv_bias else 0
        return weights + norms + bias

    def layer_params(self) -> int:
        return self._layer_params(active_only=False)

    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    def total_params(self) -> int:
        head = 0 if self.tied_embeddings else self.head().params
        return self.num_layers * self.layer_params() + self.embedding_params() + head + self.hidden_size  # + final norm

    def streamed_params(self) -> int:
        """Parameters read from memory for every generated token (MoE: only the active experts)."""
        return self.num_layers * self._layer_params(active_only=True) + self.head().params + self.hidden_size

    def active_params(self) -> int:
        """Parameters a token touches, counted the way model cards do (embedding row included)."""
        return self.streamed_params() + (0 if self.tied_embeddings else self.embedding_params())

    # ───────── per-token quantities ─────────

    def kv_bytes_per_token(self, bytes_per_value: float = 2.0) -> float:
        """KV cache added per token of context (K and V, every layer)."""
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * bytes_per_value

    def decode_flops(self, context: int) -> float:
        """Operations to generate one token: 2 per streamed weight, plus attention over `context` tokens."""
        attention = 4 * self.num_layers * self.num_heads * self.head_dim * context  # Q.K and attention x V
        return 2 * self.streamed_params() + attention

    def params_value(self) -> Value:
        return Value.of(self.total_params(), "", label="estimated", source=f"computed from {self.source}")
