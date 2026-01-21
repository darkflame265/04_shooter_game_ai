from __future__ import annotations

from typing import Tuple
import math
import torch


@torch.no_grad()
def advance_boss(
    boss_pos: torch.Tensor,
    t: torch.Tensor,
    world_size: float,
    boss_y: float,
    boss_x_amp: float,
    boss_move_hz: float,
) -> None:
    w = float(world_size)
    amp = float(boss_x_amp)
    hz = float(boss_move_hz)

    ang = (2.0 * math.pi) * hz * t
    x = 0.5 * w + amp * w * torch.sin(ang)
    x = x.clamp(0.05 * w, 0.95 * w)

    y = torch.full_like(x, float(boss_y) * w)
    boss_pos[:, 0] = x
    boss_pos[:, 1] = y


@torch.no_grad()
def advance_bullets(
    bul_pos: torch.Tensor,
    bul_vel: torch.Tensor,
    bul_alive: torch.Tensor,
    dt: float,
    world_size: float,
) -> None:
    if not torch.any(bul_alive):
        return

    bul_pos.add_(bul_vel * float(dt))

    w = float(world_size)
    pad = 0.2 * w
    x = bul_pos[..., 0]
    y = bul_pos[..., 1]
    out = (x < -pad) | (x > w + pad) | (y < -pad) | (y > w + pad)
    bul_alive &= (~out)


@torch.no_grad()
def check_hit(
    player_pos: torch.Tensor,
    bul_pos: torch.Tensor,
    bul_alive: torch.Tensor,
    player_radius: float,
    bullet_radius: float,
) -> torch.Tensor:
    if not torch.any(bul_alive):
        return torch.zeros((player_pos.shape[0],), device=player_pos.device, dtype=torch.bool)

    d = bul_pos - player_pos.view(player_pos.shape[0], 1, 2)
    dist2 = (d * d).sum(dim=-1)
    r = float(player_radius + bullet_radius)
    hit_any = (dist2 <= (r * r)) & bul_alive
    return hit_any.any(dim=1)


@torch.no_grad()
def near_miss_penalty(
    player_pos: torch.Tensor,
    bul_pos: torch.Tensor,
    bul_alive: torch.Tensor,
    player_radius: float,
    bullet_radius: float,
    near_miss_margin: float,
    near_miss_coef: float,
) -> torch.Tensor:
    if not torch.any(bul_alive):
        return torch.zeros((player_pos.shape[0],), device=player_pos.device, dtype=torch.float32)

    d = bul_pos - player_pos.view(player_pos.shape[0], 1, 2)
    dist = torch.sqrt((d * d).sum(dim=-1) + 1e-12)

    huge = torch.full_like(dist, 1e9)
    dist = torch.where(bul_alive, dist, huge)

    min_dist, _ = dist.min(dim=1)

    touch = float(player_radius + bullet_radius)
    thr = touch + float(near_miss_margin)
    x = ((thr - min_dist) / max(1e-8, (thr - touch))).clamp(0.0, 1.0)
    return x * float(near_miss_coef)


@torch.no_grad()
def build_obs(
    player_pos: torch.Tensor,
    boss_pos: torch.Tensor,
    bul_pos: torch.Tensor,
    bul_vel: torch.Tensor,
    bul_alive: torch.Tensor,
    step_count: torch.Tensor,
    world_size: float,
    max_steps: int,
    obs_k: int,
    bullet_speed_max: float,
) -> torch.Tensor:
    device = player_pos.device
    w = float(world_size)

    p = player_pos / w
    p_center = p - 0.5
    b = boss_pos / w
    t01 = (step_count.float() / float(max(1, max_steps))).clamp(0.0, 1.0).view(player_pos.shape[0], 1)

    K = int(obs_k)
    B = int(bul_pos.shape[1])

    rel = (bul_pos / w) - p.view(player_pos.shape[0], 1, 2)
    dist2 = (rel * rel).sum(dim=-1)

    inf = torch.full_like(dist2, 1e9)
    dist2 = torch.where(bul_alive, dist2, inf)

    k = min(K, B)
    _, idx = torch.topk(dist2, k=k, dim=1, largest=False)

    idx2 = idx.unsqueeze(-1).expand(player_pos.shape[0], k, 2)
    rel_k = torch.gather(rel, dim=1, index=idx2)

    v_norm = max(float(bullet_speed_max), 1e-6)
    vel = bul_vel / v_norm
    vel_k = torch.gather(vel, dim=1, index=idx2)

    feat = torch.cat([rel_k, vel_k], dim=-1)
    if k < K:
        pad = torch.zeros((player_pos.shape[0], K - k, 4), device=device, dtype=torch.float32)
        feat = torch.cat([feat, pad], dim=1)

    obs = torch.cat([p, p_center, b, t01, feat.reshape(player_pos.shape[0], -1)], dim=1)
    return obs.to(torch.float32)


@torch.no_grad()
def pick_spawn_slots(
    bul_alive: torch.Tensor,
    k_use: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Deterministic lowest-index free-slot selection.
    Returns: spawn_slots(N,max_k), do(N,max_k), k_idx(N,max_k), max_k
    """
    N, B = bul_alive.shape
    device = bul_alive.device

    max_k = int(k_use.max().item())
    if max_k <= 0:
        return (
            torch.zeros((N, 0), device=device, dtype=torch.int64),
            torch.zeros((N, 0), device=device, dtype=torch.bool),
            torch.zeros((N, 0), device=device, dtype=torch.int64),
            0,
        )

    free = ~bul_alive
    idx = torch.arange(B, device=device).view(1, B).expand(N, B)
    big = torch.full((N, B), B + 1, device=device, dtype=torch.int64)
    free_idx = torch.where(free, idx, big)
    _, free_pos = torch.sort(free_idx, dim=1)
    spawn_slots = free_pos[:, :max_k]

    k_idx = torch.arange(max_k, device=device, dtype=torch.int64).view(1, max_k).expand(N, max_k)
    do = (k_idx < k_use.view(N, 1))
    return spawn_slots, do, k_idx, max_k
