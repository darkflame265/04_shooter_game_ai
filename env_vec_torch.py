from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch

from torch_config import EnvConfig
from torch_state import EnvState
from torch_util import (
    advance_boss,
    advance_bullets,
    check_hit,
    near_miss_penalty,
    build_obs,
    pick_spawn_slots,
)
from torch_reward import compute_reward_done
from torch_pattern_factory import make_pattern
from torch_pattern_base import PatternContext
from torch_render_tk import TkRenderer


class VecShooterEnvTorch:
    """
    Torch-only vectorized env.
    difficulty == spawn_rate (bullets/sec)
    """

    ACTIONS = torch.tensor(
        [
            [0.0, 0.0],   # STOP
            [0.0, 1.0],   # UP
            [0.0, -1.0],  # DOWN
            [-1.0, 0.0],  # LEFT
            [1.0, 0.0],   # RIGHT
            [-1.0, 1.0],  # UP_LEFT
            [1.0, 1.0],   # UP_RIGHT
            [-1.0, -1.0], # DOWN_LEFT
            [1.0, -1.0],  # DOWN_RIGHT
        ],
        dtype=torch.float32,
    )

    def __init__(self, cfg: EnvConfig, n_envs: int, device: str = "cuda"):
        self.cfg = cfg
        self.n = int(n_envs)
        self.device = torch.device(device)

        self.actions = self.ACTIONS.to(self.device)
        self.action_dim = int(self.actions.shape[0])

        self.s = EnvState.allocate(self.n, int(cfg.max_bullets), self.device)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(cfg.seed))
        self._gen = g

        # obs: player(2) + player_center(2) + boss(2) + t01(1) + K*(rel(2)+vel(2))
        self.obs_dim = 2 + 2 + 2 + 1 + (int(cfg.obs_k) * 4)

        # manual difficulty
        self._manual_spawn = torch.ones((self.n,), device=self.device, dtype=torch.float32) * float(cfg.spawn_rate)
        self._manual_enabled = False

        # pattern strategy
        self._pattern = make_pattern(cfg)

        # renderer (lazy)
        self._renderer: Optional[TkRenderer] = None

        # optional trail (kept minimal; only used if you re-enable)
        self._trail = []

    # -------------------------
    # manual difficulty API
    # -------------------------
    def set_manual_difficulty(self, spawn_rate_s: float, enabled: bool = True) -> None:
        s = float(spawn_rate_s)
        if s != s:
            s = 0.0
        s = max(0.0, min(float(self.cfg.spawn_rate), s))
        self._manual_spawn.fill_(s)
        self._manual_enabled = bool(enabled)
        self.s.spawn_rate_s.fill_(s)

    def get_manual_difficulty(self) -> float:
        return float(self._manual_spawn[0].detach().float().cpu().item())

    @torch.no_grad()
    def get_spawn_rate_s(self) -> float:
        return float(self.s.spawn_rate_s[0].detach().float().cpu().item())

    # -------------------------
    # RNG state API
    # -------------------------
    def get_rng_state(self) -> Optional[torch.Tensor]:
        try:
            return self._gen.get_state()
        except Exception:
            return None

    def set_rng_state(self, state: torch.Tensor) -> None:
        if state is None:
            return
        try:
            self._gen.set_state(state)
        except Exception:
            pass

    # -------------------------
    # env API
    # -------------------------
    def reset(self, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        if mask is None:
            mask = torch.ones((self.n,), device=self.device, dtype=torch.bool)
        else:
            mask = mask.to(self.device).bool()
            if mask.ndim != 1 or mask.shape[0] != self.n:
                raise ValueError("mask must be shape (N,)")

        s = self.s
        s.episode_count[mask] += 1
        self._update_curriculum(mask)

        s.step_count[mask] = 0
        s.t[mask] = 0.0
        s.last_hit[mask] = False
        s.spawn_accum[mask] = 0.0

        # player spawn near bottom-center-ish
        w = float(self.cfg.world_size)
        cx = 0.5 * w
        cy = 0.20 * w
        noise = torch.randn((int(mask.sum().item()), 2), device=self.device, generator=self._gen) * 0.03
        pos = torch.tensor([cx, cy], device=self.device).view(1, 2) + noise
        pos = pos.clamp(0.0, w)
        s.player_pos[mask] = pos

        # phase init
        ph = torch.rand((int(mask.sum().item()),), device=self.device, generator=self._gen) * (2.0 * math.pi)
        s.phase[mask] = ph

        # clear bullets
        s.bul_alive[mask] = False
        s.bul_pos[mask] = 0.0
        s.bul_vel[mask] = 0.0

        # boss initial
        self._advance_boss()

        if self.cfg.render_draw_trails and self.n == 1 and bool(mask[0].item()):
            self._trail.clear()

        obs = self._get_obs()
        info = {"t": s.t.clone(), "episode": s.episode_count.clone(), "spawn_rate": s.spawn_rate_s.clone()}
        return obs, info

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        cfg = self.cfg
        s = self.s
        N = self.n

        action = action.to(self.device).long().clamp(0, self.action_dim - 1)

        s.step_count += 1
        s.t += float(cfg.dt)

        # move player
        move_dir = self.actions[action]
        nrm = torch.linalg.norm(move_dir, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir = move_dir / nrm
        dp = move_dir * (float(cfg.player_speed) * float(cfg.dt))

        s.player_pos.add_(dp)
        s.player_pos.clamp_(0.0, float(cfg.world_size))

        # boss + spawn + bullets
        self._advance_boss()
        self._spawn_bullets_from_boss_accum()
        self._advance_bullets()

        hit = self._check_hit()
        s.last_hit = hit

        nm = self._near_miss_penalty() if bool(cfg.near_miss_enabled) else torch.zeros((N,), device=self.device)

        rew, done = compute_reward_done(
            player_pos=s.player_pos,
            dp=dp,
            hit=hit,
            step_count=s.step_count,
            world_size=cfg.world_size,
            max_steps=int(cfg.max_steps),
            player_speed=cfg.player_speed,
            dt=cfg.dt,
            alive_reward=cfg.alive_reward,
            hit_penalty=cfg.hit_penalty,
            move_penalty=cfg.move_penalty,
            wall_penalty=cfg.wall_penalty,
            clear_bonus=cfg.clear_bonus,
            near_miss_enabled=cfg.near_miss_enabled,
            near_miss_penalty_v=nm,
        )

        if cfg.render_draw_trails and self.n == 1:
            p0 = s.player_pos[0].detach().float().cpu()
            self._trail.append((float(p0[0].item()), float(p0[1].item())))
            if len(self._trail) > 2000:
                self._trail = self._trail[-2000:]

        obs = self._get_obs()
        info = {
            "t": s.t.clone(),
            "step": s.step_count.clone(),
            "hit": hit.clone(),
            "bullets": s.bul_alive.sum(dim=1).to(torch.int32),
            "episode": s.episode_count.clone(),
            "spawn_rate": s.spawn_rate_s.clone(),
        }
        return obs, rew, done, info

    # -------------------------
    # curriculum / difficulty
    # -------------------------
    def _update_curriculum(self, mask: torch.Tensor) -> None:
        cfg = self.cfg
        s = self.s

        if self._manual_enabled:
            s.spawn_rate_s[mask] = self._manual_spawn[mask]
            return

        if not cfg.curriculum:
            s.spawn_rate_s[mask] = float(cfg.spawn_rate)
            return

        ep0 = (s.episode_count[mask].float() - 1.0).clamp_min(0.0)
        ss = float(cfg.spawn_rate_start) + ep0 * float(cfg.spawn_rate_step)
        ss = ss.clamp(0.0, float(cfg.spawn_rate))
        s.spawn_rate_s[mask] = ss

    # -------------------------
    # delegated physics
    # -------------------------
    @torch.no_grad()
    def _advance_boss(self) -> None:
        cfg = self.cfg
        s = self.s
        advance_boss(
            boss_pos=s.boss_pos,
            t=s.t,
            world_size=cfg.world_size,
            boss_y=cfg.boss_y,
            boss_x_amp=cfg.boss_x_amp,
            boss_move_hz=cfg.boss_move_hz,
        )

    @torch.no_grad()
    def _advance_bullets(self) -> None:
        cfg = self.cfg
        s = self.s
        advance_bullets(
            bul_pos=s.bul_pos,
            bul_vel=s.bul_vel,
            bul_alive=s.bul_alive,
            dt=cfg.dt,
            world_size=cfg.world_size,
        )

    @torch.no_grad()
    def _check_hit(self) -> torch.Tensor:
        cfg = self.cfg
        s = self.s
        return check_hit(
            player_pos=s.player_pos,
            bul_pos=s.bul_pos,
            bul_alive=s.bul_alive,
            player_radius=cfg.player_radius,
            bullet_radius=cfg.bullet_radius,
        )

    @torch.no_grad()
    def _near_miss_penalty(self) -> torch.Tensor:
        cfg = self.cfg
        s = self.s
        return near_miss_penalty(
            player_pos=s.player_pos,
            bul_pos=s.bul_pos,
            bul_alive=s.bul_alive,
            player_radius=cfg.player_radius,
            bullet_radius=cfg.bullet_radius,
            near_miss_margin=cfg.near_miss_margin,
            near_miss_coef=cfg.near_miss_coef,
        )

    # -------------------------
    # spawn bullets (glue only)
    # -------------------------
    @torch.no_grad()
    def _spawn_bullets_from_boss_accum(self) -> None:
        cfg = self.cfg
        s = self.s
        N = self.n

        s.spawn_accum += s.spawn_rate_s * float(cfg.dt)
        k = torch.floor(s.spawn_accum).to(torch.int64)
        if not torch.any(k > 0):
            return

        s.spawn_accum -= k.to(torch.float32)

        free_count = (~s.bul_alive).sum(dim=1).to(torch.int64)
        k_use = torch.minimum(k, free_count)
        if not torch.any(k_use > 0):
            return

        pid = int(cfg.pattern_id)

        # pattern2: 8발 단위 스폰 유지
        if pid == 2:
            volley = 8
            k_use_adj = (k_use // volley) * volley
            dropped = (k_use - k_use_adj).to(torch.float32)
            if torch.any(dropped > 0):
                s.spawn_accum += dropped
            k_use = k_use_adj
            if not torch.any(k_use > 0):
                return

        spawn_slots, do, k_idx, max_k = pick_spawn_slots(s.bul_alive, k_use)
        if max_k <= 0 or (not torch.any(do)):
            return

        ctx = PatternContext(
            t=s.t,
            step_count=s.step_count,
            phase=s.phase,
            boss_pos=s.boss_pos,
            player_pos=s.player_pos,
            k_idx=k_idx,
            k_use=k_use,
            max_k=max_k,
            bullet_speed_min=float(cfg.bullet_speed_min),
            bullet_speed_max=float(cfg.bullet_speed_max),
        )

        ang, spd, new_phase = self._pattern.angles_speeds(ctx)
        s.phase = new_phase

        # optional RNG aim noise
        if float(cfg.aim_noise) > 0.0:
            noise = torch.randn((N, max_k), device=self.device, generator=self._gen) * float(cfg.aim_noise)
            ang = ang + noise

        vel = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * spd.unsqueeze(-1)
        pos = s.boss_pos.view(N, 1, 2).expand(N, max_k, 2).contiguous()

        env_i = torch.arange(N, device=self.device).view(N, 1).expand(N, max_k)
        env_i2 = env_i[do]
        bi2 = spawn_slots[do]

        s.bul_alive[env_i2, bi2] = True
        s.bul_pos[env_i2, bi2] = pos[do]
        s.bul_vel[env_i2, bi2] = vel[do]

    # -------------------------
    # observation
    # -------------------------
    @torch.no_grad()
    def _get_obs(self) -> torch.Tensor:
        cfg = self.cfg
        s = self.s
        return build_obs(
            player_pos=s.player_pos,
            boss_pos=s.boss_pos,
            bul_pos=s.bul_pos,
            bul_vel=s.bul_vel,
            bul_alive=s.bul_alive,
            step_count=s.step_count,
            world_size=cfg.world_size,
            max_steps=int(cfg.max_steps),
            obs_k=int(cfg.obs_k),
            bullet_speed_max=float(cfg.bullet_speed_max),
        )

    # -------------------------
    # rendering
    # -------------------------
    def render(self) -> None:
        if self.n != 1:
            raise RuntimeError("render() is only supported when n_envs == 1")

        if self._renderer is None:
            self._renderer = TkRenderer(
                render_size=int(self.cfg.render_size),
                render_fps=float(self.cfg.render_fps),
                max_bullets=int(self.cfg.max_bullets),
            )

        s = self.s
        p0 = s.player_pos[0].detach().float().cpu()
        b0 = s.boss_pos[0].detach().float().cpu()
        alive0 = s.bul_alive[0].detach().cpu()
        bul0 = s.bul_pos[0].detach().float().cpu()

        self._renderer.render(
            world_size=float(self.cfg.world_size),
            player_radius=float(self.cfg.player_radius),
            bullet_radius=float(self.cfg.bullet_radius),
            pattern_id=int(self.cfg.pattern_id),
            episode_count0=int(s.episode_count[0].item()),
            step0=int(s.step_count[0].item()),
            t0=float(s.t[0].item()),
            hit0=int(s.last_hit[0].item()),
            bullets0=int(alive0.sum().item()),
            spawn_rate0=float(s.spawn_rate_s[0].detach().float().cpu().item()),
            boss_xy0=(float(b0[0].item()), float(b0[1].item())),
            player_pos0=p0,
            boss_pos0=b0,
            bul_alive0=alive0,
            bul_pos0=bul0,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
