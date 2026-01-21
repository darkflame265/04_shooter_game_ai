from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class EnvState:
    # core
    player_pos: torch.Tensor   # (N,2)
    boss_pos: torch.Tensor     # (N,2)

    # bullets
    bul_pos: torch.Tensor      # (N,B,2)
    bul_vel: torch.Tensor      # (N,B,2)
    bul_alive: torch.Tensor    # (N,B) bool

    # time/episode
    step_count: torch.Tensor   # (N,) int32
    t: torch.Tensor            # (N,) float32
    last_hit: torch.Tensor     # (N,) bool
    episode_count: torch.Tensor  # (N,) int32

    # difficulty accumulator
    spawn_rate_s: torch.Tensor  # (N,) float32
    spawn_accum: torch.Tensor   # (N,) float32

    # pattern phase
    phase: torch.Tensor         # (N,) float32

    @staticmethod
    def allocate(n_envs: int, max_bullets: int, device: torch.device) -> "EnvState":
        N = int(n_envs)
        B = int(max_bullets)

        z2 = torch.zeros((N, 2), device=device, dtype=torch.float32)
        bul_pos = torch.zeros((N, B, 2), device=device, dtype=torch.float32)
        bul_vel = torch.zeros((N, B, 2), device=device, dtype=torch.float32)
        bul_alive = torch.zeros((N, B), device=device, dtype=torch.bool)

        step_count = torch.zeros((N,), device=device, dtype=torch.int32)
        t = torch.zeros((N,), device=device, dtype=torch.float32)
        last_hit = torch.zeros((N,), device=device, dtype=torch.bool)
        episode_count = torch.zeros((N,), device=device, dtype=torch.int32)

        spawn_rate_s = torch.zeros((N,), device=device, dtype=torch.float32)
        spawn_accum = torch.zeros((N,), device=device, dtype=torch.float32)

        phase = torch.zeros((N,), device=device, dtype=torch.float32)

        return EnvState(
            player_pos=z2.clone(),
            boss_pos=z2.clone(),
            bul_pos=bul_pos,
            bul_vel=bul_vel,
            bul_alive=bul_alive,
            step_count=step_count,
            t=t,
            last_hit=last_hit,
            episode_count=episode_count,
            spawn_rate_s=spawn_rate_s,
            spawn_accum=spawn_accum,
            phase=phase,
        )
