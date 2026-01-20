from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import time
import math
import torch


@dataclass
class EnvConfig:
    # world
    world_size: float = 1.0
    dt: float = 1.0 / 60.0
    max_steps: int = 60 * 60  # 60 seconds

    # player
    player_speed: float = 0.3
    player_radius: float = 0.01

    # bullets (capacity)
    max_bullets: int = 2048
    bullet_radius: float = 0.01

    # bullet speed is FIXED by cfg (no difficulty scaling)
    bullet_speed_min: float = 0.1
    bullet_speed_max: float = 0.2

    # aim noise (keep variable, but default 0 for clean boss patterns)
    aim_noise: float = 0.0

    # "final/target" spawn (upper bound)
    spawn_rate: float = 300.0

    # observation
    obs_k: int = 32

    # reward shaping
    alive_reward: float = 0.01
    hit_penalty: float = 3.0
    move_penalty: float = 0.0005
    wall_penalty: float = 0.02

    near_miss_enabled: bool = True
    near_miss_margin: float = 0.06
    near_miss_coef: float = 0.02

    clear_bonus: float = 0.25

    # Curriculum (spawn_rate만 선형 증가)
    curriculum: bool = True
    curriculum_episodes: int = 800  # (유지: 의미는 "몇 에피소드에 걸쳐 목표 spawn까지 가는지")
    spawn_rate_start: float = 3.0   # 시작 spawn/s
    spawn_rate_step: float = 0.5    # ✅ 에피소드 1회당 spawn 증가량 (너가 요청한 값)

    # (kept for compatibility; ignored)
    bullet_speed_start_factor: float = 1.0
    aim_noise_start_factor: float = 1.0

    # -------------------------
    # Boss params
    # -------------------------
    boss_y: float = 0.92
    boss_x_amp: float = 0.38
    boss_move_hz: float = 0.1

    # pattern
    # 0: wobble ring
    # 1: simple rotating ring
    # 2: flower/rosette
    pattern_id: int = 2
    phase_speed0: float = 0.08
    phase_speed1: float = 0.12

    # flower/rosette params
    flower_petals: int = 6
    flower_phase_speed: float = 0.10
    flower_speed_mod: float = 0.55
    flower_twist: float = 0.35

    # Rendering (tkinter)
    render_size: int = 720
    render_fps: int = 60
    render_draw_trails: bool = False

    seed: int = 0


class VecShooterEnvTorch:
    """
    Boss-pattern simulator (Torch-only vectorized env).

    ✅ NEW difficulty model:
      - difficulty == spawn_rate (bullets/sec, float)
      - curriculum increments spawn_rate linearly by cfg.spawn_rate_step each episode
      - spawns are realized by accumulator: acc += spawn_rate*dt, k=floor(acc), acc-=k

    This removes:
      - diff01->intensity mapping
      - rate_steps/n integer stepping
      - large spawn jumps
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

        N = self.n
        B = int(cfg.max_bullets)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(cfg.seed))
        self._gen = g

        # ----- state tensors
        self.player_pos = torch.zeros((N, 2), device=self.device, dtype=torch.float32)
        self.boss_pos = torch.zeros((N, 2), device=self.device, dtype=torch.float32)

        self.bul_pos = torch.zeros((N, B, 2), device=self.device, dtype=torch.float32)
        self.bul_vel = torch.zeros((N, B, 2), device=self.device, dtype=torch.float32)
        self.bul_alive = torch.zeros((N, B), device=self.device, dtype=torch.bool)

        self.step_count = torch.zeros((N,), device=self.device, dtype=torch.int32)
        self.t = torch.zeros((N,), device=self.device, dtype=torch.float32)
        self.last_hit = torch.zeros((N,), device=self.device, dtype=torch.bool)

        self.episode_count = torch.zeros((N,), device=self.device, dtype=torch.int32)

        # ✅ current spawn_rate (bullets/sec) per env
        self.spawn_rate_s = torch.zeros((N,), device=self.device, dtype=torch.float32)
        # ✅ accumulator for fractional spawn
        self.spawn_accum = torch.zeros((N,), device=self.device, dtype=torch.float32)

        # pattern internal phase (per env)
        self.phase = torch.zeros((N,), device=self.device, dtype=torch.float32)

        self.action_dim = int(self.actions.shape[0])

        # obs: player(2) + player_center(2) + boss(2) + t01(1) + K*(rel(2)+vel(2))
        self.obs_dim = 2 + 2 + 2 + 1 + (cfg.obs_k * 4)

        # ---- manual spawn override
        self._manual_spawn = torch.ones((N,), device=self.device, dtype=torch.float32) * float(cfg.spawn_rate)
        self._manual_enabled = False

        # ---- render state (lazy)
        self._ui_inited = False
        self._tk = None
        self._root = None
        self._canvas = None
        self._hud_var = None
        self._last_render_ts = 0.0
        self._item_border = None
        self._item_player = None
        self._item_boss = None
        self._item_bullets = []
        self._item_trail = []
        self._trail = []

    # -------------------------
    # manual difficulty API (trainer)
    # -------------------------
    def set_manual_difficulty(self, spawn_rate_s: float, enabled: bool = True) -> None:
        """
        ✅ 이제 difficulty는 diff01이 아니라 spawn_rate (bullets/sec) 자체.
        """
        s = float(spawn_rate_s)
        if s != s:
            s = 0.0
        s = max(0.0, min(float(self.cfg.spawn_rate), s))
        self._manual_spawn.fill_(s)
        self._manual_enabled = bool(enabled)
        self.spawn_rate_s.fill_(s)

    def get_manual_difficulty(self) -> float:
        return float(self._manual_spawn[0].detach().float().cpu().item())

    @torch.no_grad()
    def get_spawn_rate_s(self) -> float:
        return float(self.spawn_rate_s[0].detach().float().cpu().item())

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

        self.episode_count[mask] += 1
        self._update_curriculum(mask)

        self.step_count[mask] = 0
        self.t[mask] = 0.0
        self.last_hit[mask] = False

        # ✅ accumulator reset (에피소드마다 분수 누적 리셋)
        self.spawn_accum[mask] = 0.0

        # player spawn near bottom-center-ish
        w = float(self.cfg.world_size)
        cx = 0.5 * w
        cy = 0.20 * w
        noise = torch.randn((int(mask.sum().item()), 2), device=self.device, generator=self._gen) * 0.03
        pos = torch.tensor([cx, cy], device=self.device).view(1, 2) + noise
        pos = pos.clamp(0.0, w)
        self.player_pos[mask] = pos

        # phase init
        ph = torch.rand((int(mask.sum().item()),), device=self.device, generator=self._gen) * (2.0 * math.pi)
        self.phase[mask] = ph

        # clear bullets
        self.bul_alive[mask] = False
        self.bul_pos[mask] = 0.0
        self.bul_vel[mask] = 0.0

        # set boss initial position
        self._advance_boss()

        if self.cfg.render_draw_trails and self.n == 1 and bool(mask[0].item()):
            self._trail.clear()

        obs = self._get_obs()
        info = {"t": self.t.clone(), "episode": self.episode_count.clone(), "spawn_rate": self.spawn_rate_s.clone()}
        return obs, info

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        cfg = self.cfg
        N = self.n

        action = action.to(self.device).long().clamp(0, self.action_dim - 1)

        self.step_count += 1
        self.t += float(cfg.dt)

        # ---- move player
        move_dir = self.actions[action]
        nrm = torch.linalg.norm(move_dir, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir = move_dir / nrm

        dp = move_dir * (float(cfg.player_speed) * float(cfg.dt))
        self.player_pos += dp
        self.player_pos.clamp_(0.0, float(cfg.world_size))

        # ---- move boss + spawn/advance bullets
        self._advance_boss()
        self._spawn_bullets_from_boss_accum()
        self._advance_bullets()

        hit = self._check_hit()
        self.last_hit = hit

        done = torch.zeros((N,), device=self.device, dtype=torch.bool)
        rew = torch.zeros((N,), device=self.device, dtype=torch.float32)

        rew += float(cfg.alive_reward)

        denom = float(cfg.player_speed) * float(cfg.dt) + 1e-8
        rew -= float(cfg.move_penalty) * (torch.linalg.norm(dp, dim=1) / denom)

        margin = 0.08 * float(cfg.world_size)
        x = self.player_pos[:, 0]
        y = self.player_pos[:, 1]
        d_left = x
        d_right = float(cfg.world_size) - x
        d_bot = y
        d_top = float(cfg.world_size) - y
        dmin = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_bot, d_top))
        wall_frac = ((margin - dmin) / max(1e-8, margin)).clamp(0.0, 1.0)
        rew -= float(cfg.wall_penalty) * wall_frac

        if cfg.near_miss_enabled:
            rew -= self._near_miss_penalty()

        rew = torch.where(hit, rew - float(cfg.hit_penalty), rew)
        done = done | hit

        timeout = self.step_count >= int(cfg.max_steps)
        done = done | timeout
        rew = torch.where(timeout & (~hit), rew + float(cfg.clear_bonus), rew)

        if cfg.render_draw_trails and self.n == 1:
            p0 = self.player_pos[0].detach().float().cpu()
            self._trail.append((float(p0[0].item()), float(p0[1].item())))
            if len(self._trail) > 2000:
                self._trail = self._trail[-2000:]

        obs = self._get_obs()
        info = {
            "t": self.t.clone(),
            "step": self.step_count.clone(),
            "hit": hit.clone(),
            "bullets": self.bul_alive.sum(dim=1).to(torch.int32),
            "episode": self.episode_count.clone(),
            "spawn_rate": self.spawn_rate_s.clone(),
        }
        return obs, rew, done, info

    # -------------------------
    # curriculum / difficulty (spawn-rate only)
    # -------------------------
    def _update_curriculum(self, mask: torch.Tensor) -> None:
        cfg = self.cfg

        if self._manual_enabled:
            self.spawn_rate_s[mask] = self._manual_spawn[mask]
            return

        if not cfg.curriculum:
            self.spawn_rate_s[mask] = float(cfg.spawn_rate)
            return

        # ✅ simple linear increment per episode
        ep0 = (self.episode_count[mask].float() - 1.0).clamp_min(0.0)
        s = float(cfg.spawn_rate_start) + ep0 * float(cfg.spawn_rate_step)

        # ✅ "복잡한 clamp"는 안 하고, 최대치만 cfg.spawn_rate로 제한 (원하면 이 줄도 제거 가능)
        s = s.clamp(0.0, float(cfg.spawn_rate))
        self.spawn_rate_s[mask] = s

    # -------------------------
    # boss motion
    # -------------------------
    @torch.no_grad()
    def _advance_boss(self) -> None:
        cfg = self.cfg
        w = float(cfg.world_size)

        amp = float(cfg.boss_x_amp)
        hz = float(cfg.boss_move_hz)

        ang = (2.0 * math.pi) * hz * self.t
        x = 0.5 * w + amp * w * torch.sin(ang)
        x = x.clamp(0.05 * w, 0.95 * w)

        y = torch.full_like(x, float(cfg.boss_y) * w)
        self.boss_pos[:, 0] = x
        self.boss_pos[:, 1] = y

    # -------------------------
    # bullets spawn (accumulator-based, supports float spawn-rate)
    # -------------------------
    @torch.no_grad()
    def _spawn_bullets_from_boss_accum(self) -> None:
        cfg = self.cfg
        N = self.n
        B = int(cfg.max_bullets)

        # acc += spawn_rate*dt, k=floor(acc)
        self.spawn_accum += self.spawn_rate_s * float(cfg.dt)
        k = torch.floor(self.spawn_accum).to(torch.int64)
        if not torch.any(k > 0):
            return
        self.spawn_accum -= k.to(torch.float32)

        # free slots
        free = ~self.bul_alive
        free_count = free.sum(dim=1).to(torch.int64)
        k_use = torch.minimum(k, free_count)
        if not torch.any(k_use > 0):
            return

        max_k = int(k_use.max().item())
        if max_k <= 0:
            return

        # deterministic free-slot picking: take lowest indices
        idx = torch.arange(B, device=self.device).view(1, B).expand(N, B)
        big = torch.full((N, B), B + 1, device=self.device, dtype=torch.int64)
        free_idx = torch.where(free, idx, big)
        _, free_pos = torch.sort(free_idx, dim=1)
        spawn_slots = free_pos[:, :max_k]  # (N, max_k)

        k_idx = torch.arange(max_k, device=self.device, dtype=torch.int64).view(1, max_k).expand(N, max_k)
        do = (k_idx < k_use.view(N, 1))
        if not torch.any(do):
            return

        # ring angles
        kf = k_idx.to(torch.float32)
        nf = k_use.to(torch.float32).clamp_min(1.0).view(N, 1)
        ang_base = (2.0 * math.pi) * (kf / nf)  # (N, max_k)

        t_steps = self.step_count.to(torch.float32)
        pid = int(cfg.pattern_id)

        if pid == 0:
            a0 = self.phase + 0.25 * torch.sin(0.03 * t_steps)
            a0 = a0.view(N, 1).expand(N, max_k)
            ang = a0 + ang_base

            # phase increment when bullets spawned
            phase_inc = float(cfg.phase_speed0)
            self.phase = self.phase + phase_inc * (k_use > 0).to(torch.float32)

            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            frac = (kf / max(1.0, float(max_k - 1))).clamp(0.0, 1.0)
            spd = sp_min + (sp_max - sp_min) * frac

        elif pid == 1:
            a0 = self.phase + float(cfg.phase_speed1) * t_steps
            a0 = a0.view(N, 1).expand(N, max_k)
            ang = a0 + ang_base

            self.phase = self.phase + 0.0

            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            frac = ((kf * 1.6180339) % max(1.0, float(max_k))).clamp(0.0, float(max_k))
            frac = (frac / max(1.0, float(max_k - 1))).clamp(0.0, 1.0)
            spd = sp_min + (sp_max - sp_min) * frac

        else:
            m = max(2, int(cfg.flower_petals))
            t2 = t_steps.view(N, 1)

            a0 = self.phase.view(N, 1) + 0.02 * t2
            a0 = a0.expand(N, max_k)

            twist = float(cfg.flower_twist)
            ang = a0 + ang_base + twist * torch.sin(float(m) * ang_base + 0.015 * t2).to(torch.float32)

            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            base = 0.5 * (sp_min + sp_max)
            amp = 0.5 * (sp_max - sp_min)

            mod = float(cfg.flower_speed_mod)
            ph = self.phase.view(N, 1)
            pet = torch.cos(float(m) * ang + ph)
            pet2 = 0.35 * torch.cos(float(2 * m) * ang - 0.5 * ph)
            shape = (pet + pet2).clamp(-1.0, 1.0)
            spd = base + amp * (mod * shape)
            spd = spd.clamp(sp_min, sp_max)

            phase_inc = float(cfg.flower_phase_speed)
            self.phase = self.phase + phase_inc * (k_use > 0).to(torch.float32)

        # optional aim noise (uses RNG; reproducible via rng_state)
        if float(cfg.aim_noise) > 0.0:
            noise = torch.randn((N, max_k), device=self.device, generator=self._gen) * float(cfg.aim_noise)
            ang = ang + noise

        vel = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * spd.unsqueeze(-1)
        pos = self.boss_pos.view(N, 1, 2).expand(N, max_k, 2).contiguous()

        env_i = torch.arange(N, device=self.device).view(N, 1).expand(N, max_k)
        bi = spawn_slots

        env_i2 = env_i[do]
        bi2 = bi[do]
        pos2 = pos[do]
        vel2 = vel[do]

        self.bul_alive[env_i2, bi2] = True
        self.bul_pos[env_i2, bi2] = pos2
        self.bul_vel[env_i2, bi2] = vel2

    @torch.no_grad()
    def _advance_bullets(self) -> None:
        cfg = self.cfg
        alive = self.bul_alive
        if not torch.any(alive):
            return

        self.bul_pos = self.bul_pos + self.bul_vel * float(cfg.dt)

        w = float(cfg.world_size)
        pad = 0.2 * w
        x = self.bul_pos[..., 0]
        y = self.bul_pos[..., 1]
        out = (x < -pad) | (x > w + pad) | (y < -pad) | (y > w + pad)
        self.bul_alive = self.bul_alive & (~out)

    @torch.no_grad()
    def _check_hit(self) -> torch.Tensor:
        cfg = self.cfg
        alive = self.bul_alive
        if not torch.any(alive):
            return torch.zeros((self.n,), device=self.device, dtype=torch.bool)

        d = self.bul_pos - self.player_pos.view(self.n, 1, 2)
        dist2 = (d * d).sum(dim=-1)
        r = float(cfg.player_radius + cfg.bullet_radius)
        hit_any = (dist2 <= (r * r)) & alive
        return hit_any.any(dim=1)

    @torch.no_grad()
    def _near_miss_penalty(self) -> torch.Tensor:
        cfg = self.cfg
        alive = self.bul_alive
        if not torch.any(alive):
            return torch.zeros((self.n,), device=self.device, dtype=torch.float32)

        d = self.bul_pos - self.player_pos.view(self.n, 1, 2)
        dist = torch.sqrt((d * d).sum(dim=-1) + 1e-12)
        huge = torch.full_like(dist, 1e9)
        dist = torch.where(alive, dist, huge)

        min_dist, _ = dist.min(dim=1)

        touch = float(cfg.player_radius + cfg.bullet_radius)
        thr = touch + float(cfg.near_miss_margin)
        x = ((thr - min_dist) / max(1e-8, (thr - touch))).clamp(0.0, 1.0)
        return x * float(cfg.near_miss_coef)

    # -------------------------
    # observation
    # -------------------------
    @torch.no_grad()
    def _get_obs(self) -> torch.Tensor:
        cfg = self.cfg
        w = float(cfg.world_size)

        p = self.player_pos / w
        p_center = p - 0.5
        b = (self.boss_pos / w)
        t01 = (self.step_count.float() / float(max(1, cfg.max_steps))).clamp(0.0, 1.0).view(self.n, 1)

        K = int(cfg.obs_k)
        B = int(cfg.max_bullets)

        rel = (self.bul_pos / w) - p.view(self.n, 1, 2)
        dist2 = (rel * rel).sum(dim=-1)
        inf = torch.full_like(dist2, 1e9)
        dist2 = torch.where(self.bul_alive, dist2, inf)

        k = min(K, B)
        _, idx = torch.topk(dist2, k=k, dim=1, largest=False)

        idx2 = idx.unsqueeze(-1).expand(self.n, k, 2)
        rel_k = torch.gather(rel, dim=1, index=idx2)

        v_norm = max(float(cfg.bullet_speed_max), 1e-6)
        vel = self.bul_vel / v_norm
        vel_k = torch.gather(vel, dim=1, index=idx2)

        feat = torch.cat([rel_k, vel_k], dim=-1)
        if k < K:
            pad = torch.zeros((self.n, K - k, 4), device=self.device, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=1)

        obs = torch.cat([p, p_center, b, t01, feat.reshape(self.n, -1)], dim=1)
        return obs.to(torch.float32)

    # -------------------------
    # tkinter rendering (N==1 only)
    # -------------------------
    def render(self) -> None:
        if self.n != 1:
            raise RuntimeError("render() is only supported when n_envs == 1")

        self._ui_lazy_init()

        now = time.time()
        min_dt = 1.0 / max(1, int(self.cfg.render_fps))
        if (now - self._last_render_ts) < min_dt:
            return
        self._last_render_ts = now

        root = self._root
        canvas = self._canvas
        if root is None or canvas is None:
            return

        try:
            root.update_idletasks()
        except Exception:
            return

        W = int(self.cfg.render_size)
        w = float(self.cfg.world_size)

        p0 = self.player_pos[0].detach().float().cpu()
        b0 = self.boss_pos[0].detach().float().cpu()
        alive0 = self.bul_alive[0].detach().cpu()
        bul0 = self.bul_pos[0].detach().float().cpu()

        def to_px_xy(xy0: torch.Tensor) -> Tuple[float, float]:
            x = (float(xy0[0].item()) / w) * (W - 1)
            y = (1.0 - float(xy0[1].item()) / w) * (W - 1)
            return x, y

        spawn_s = self.get_spawn_rate_s()
        hud_txt = (
            f"EP {int(self.episode_count[0].item())}  step {int(self.step_count[0].item())}  "
            f"t={float(self.t[0].item()):.2f}s  hit={int(self.last_hit[0].item())}\n"
            f"bullets={int(alive0.sum().item())}  spawn={spawn_s:.2f}/s  "
            f"boss=({float(b0[0].item()):.2f},{float(b0[1].item()):.2f})  pattern_id={int(self.cfg.pattern_id)}"
        )
        try:
            self._hud_var.set(hud_txt)
        except Exception:
            pass

        px, py = to_px_xy(p0)
        pr = max(3, int(self.cfg.player_radius / w * W))
        canvas.coords(self._item_player, px - pr, py - pr, px + pr, py + pr)

        bx, by = to_px_xy(b0)
        br0 = max(4, int(self.cfg.player_radius / w * W) + 2)
        canvas.coords(self._item_boss, bx - br0, by - br0, bx + br0, by + br0)

        alive_idx = torch.where(alive0)[0]
        br = max(2, int(self.cfg.bullet_radius / w * W))

        for it in self._item_bullets:
            canvas.itemconfigure(it, state="hidden")

        for j in range(min(int(alive_idx.numel()), len(self._item_bullets))):
            i = int(alive_idx[j].item())
            x, y = to_px_xy(bul0[i])
            it = self._item_bullets[j]
            canvas.coords(it, x - br, y - br, x + br, y + br)
            canvas.itemconfigure(it, state="normal")

        try:
            root.update()
        except Exception:
            self.close()

    def _ui_lazy_init(self) -> None:
        if self._ui_inited:
            return
        try:
            import tkinter as tk
        except Exception as e:
            raise RuntimeError("tkinter is not available.") from e

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("VecShooterEnvTorch (spawn-rate curriculum)")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        W = int(self.cfg.render_size)

        self._hud_var = tk.StringVar(value="")
        hud = tk.Label(self._root, textvariable=self._hud_var, justify="left", anchor="w", font=("Consolas", 10))
        hud.pack(fill="x")

        canvas = tk.Canvas(self._root, width=W, height=W, bg="#121216", highlightthickness=0)
        canvas.pack()

        self._item_border = canvas.create_rectangle(2, 2, W - 2, W - 2, outline="#505058", width=2)

        self._item_bullets = []
        for _ in range(int(self.cfg.max_bullets)):
            it = canvas.create_oval(0, 0, 0, 0, fill="#f0e65a", outline="", state="hidden")
            self._item_bullets.append(it)

        self._item_player = canvas.create_oval(0, 0, 0, 0, fill="#5adcff", outline="")
        self._item_boss = canvas.create_oval(0, 0, 0, 0, fill="#ff5ac8", outline="")

        self._canvas = canvas
        self._last_render_ts = 0.0
        self._ui_inited = True

    def close(self) -> None:
        if not self._ui_inited:
            return
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        self._ui_inited = False
        self._tk = None
        self._root = None
        self._canvas = None
        self._hud_var = None
        self._item_border = None
        self._item_player = None
        self._item_boss = None
        self._item_bullets = []
        self._item_trail = []
        self._trail = []
