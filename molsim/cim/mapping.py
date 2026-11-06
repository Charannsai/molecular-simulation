"""Mapping an LLM's weight matrices onto crossbar tiles.

A matrix with n_in inputs and n_out outputs needs ceil(n_in / rows) x ceil(n_out x slices / cols) tiles
(slices = columns per weight when the device needs bit slicing; 1 otherwise).
Tiles along the input dimension each produce a partial sum for the same outputs; those partial
sums are added digitally. Edge tiles may be partly empty (utilisation below 100%).

Decode at batch 1 runs each layer as four sequential analog stages; matrices inside a stage run
at the same time on their own tiles:
    QKV -> (attention) -> O -> gate & up -> down
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..llm import Matrix, Model
from .tile import TileSpec

STAGES = [("qkv", ("q", "k", "v")), ("o", ("o",)), ("gate_up", ("gate", "up")), ("down", ("down",))]


@dataclass(frozen=True)
class Placement:
    matrix: Matrix
    row_tiles: int
    col_tiles: int

    @property
    def tiles(self) -> int:
        return self.row_tiles * self.col_tiles

    def utilisation(self, tile: TileSpec) -> float:
        return self.matrix.params / (self.tiles * tile.weights)

    slices: int = 1

    @property
    def partial_sum_adds(self) -> int:
        """Digital additions to combine partial sums from tiles along the input dimension and weight slices."""
        return (self.row_tiles * self.slices - 1) * self.matrix.n_out


def place(m: Matrix, tile: TileSpec) -> Placement:
    return Placement(m, math.ceil(m.n_in / tile.rows), math.ceil(m.n_out * tile.weight_slices / tile.cols), tile.weight_slices)


@dataclass(frozen=True)
class Mapping:
    model: Model
    tile: TileSpec
    layer: list[Placement]  # one layer's matrices (every layer has the same shapes)
    head: Placement
    active_layer: list[Placement]  # what one token uses (MoE: router + chosen experts; dense: everything)

    @property
    def tiles_per_layer(self) -> int:
        return sum(p.tiles for p in self.layer)

    @property
    def active_tiles_per_token(self) -> int:
        """Tiles that compute for one token. All tiles are installed; for MoE most sit idle."""
        return self.model.num_layers * sum(p.tiles for p in self.active_layer) + self.head.tiles

    @property
    def stages_per_layer(self) -> int:
        """Sequential analog steps per layer (MoE adds the router before the experts)."""
        return len(STAGES) + (1 if self.model.is_moe else 0)

    @property
    def total_tiles(self) -> int:
        return self.model.num_layers * self.tiles_per_layer + self.head.tiles

    @property
    def total_devices(self) -> int:
        return self.total_tiles * self.tile.devices

    @property
    def utilisation(self) -> float:
        stored = self.model.num_layers * sum(p.matrix.params for p in self.layer) + self.head.matrix.params
        return stored / (self.total_tiles * self.tile.weights)

    @property
    def partial_sum_adds_per_token(self) -> int:
        return self.model.num_layers * sum(p.partial_sum_adds for p in self.active_layer) + self.head.partial_sum_adds

    def stage(self, names: tuple[str, ...]) -> list[Placement]:
        return [p for p in self.layer if p.matrix.name in names]


def map_model(model: Model, tile: TileSpec) -> Mapping:
    return Mapping(
        model, tile,
        [place(m, tile) for m in model.layer_matrices()],
        place(model.head(), tile),
        [place(m, tile) for m in model.layer_matrices(active_only=True)],
    )
