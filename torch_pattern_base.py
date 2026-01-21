from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple
import torch


@dataclass(frozen=True)
class PatternContext:
    # time
    t: torch.Tensor          # (N,)
    step_count: torch.Tensor # (N,)
    # state
    phase: torch.Tensor      # (N,)
    boss_pos: torch.Tensor   # (N,2)
    player_pos: torch.Tensor # (N,2)
    # spawn batch indices
    k_idx: torch.Tensor      # (N,max_k) int64
    k_use: torch.Tensor      # (N,) int64
    max_k: int

    # speeds
    bullet_speed_min: float
    bullet_speed_max: float


class BulletPattern(Protocol):
    def angles_speeds(self, ctx: PatternContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          ang: (N,max_k) float32
          spd: (N,max_k) float32
          new_phase: (N,) float32
        """
        ...
