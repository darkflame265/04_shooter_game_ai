from __future__ import annotations

import torch


@torch.no_grad()
def compute_reward_done(
    *,
    player_pos: torch.Tensor,   # (N,2)
    dp: torch.Tensor,           # (N,2)
    hit: torch.Tensor,          # (N,) bool
    step_count: torch.Tensor,   # (N,)
    world_size: float,
    max_steps: int,
    player_speed: float,
    dt: float,
    alive_reward: float,
    hit_penalty: float,
    move_penalty: float,
    wall_penalty: float,
    clear_bonus: float,
    near_miss_enabled: bool,
    near_miss_penalty_v: torch.Tensor,  # (N,) float32 (already computed)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (rew(N,), done(N,))
    """
    device = player_pos.device
    N = player_pos.shape[0]

    rew = torch.zeros((N,), device=device, dtype=torch.float32)
    done = torch.zeros((N,), device=device, dtype=torch.bool)

    rew += float(alive_reward)

    denom = float(player_speed) * float(dt) + 1e-8
    rew -= float(move_penalty) * (torch.linalg.norm(dp, dim=1) / denom)

    # wall penalty
    margin = 0.08 * float(world_size)
    x = player_pos[:, 0]
    y = player_pos[:, 1]
    d_left = x
    d_right = float(world_size) - x
    d_bot = y
    d_top = float(world_size) - y
    dmin = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_bot, d_top))
    wall_frac = ((margin - dmin) / max(1e-8, margin)).clamp(0.0, 1.0)
    rew -= float(wall_penalty) * wall_frac

    if bool(near_miss_enabled):
        rew -= near_miss_penalty_v

    # hit => done + penalty
    rew = torch.where(hit, rew - float(hit_penalty), rew)
    done = done | hit

    # timeout / clear bonus
    timeout = step_count >= int(max_steps)
    done = done | timeout
    rew = torch.where(timeout & (~hit), rew + float(clear_bonus), rew)

    return rew, done
