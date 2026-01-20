from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import time
import torch


@dataclass
class EnvConfig:
    # world
    world_size: float = 1.0
    dt: float = 1.0 / 60.0
    max_steps: int = 60 * 60  # 60 seconds

    # player
    player_speed: float = 0.55
    player_radius: float = 0.01

    # bullets (TARGET / final difficulty)
    max_bullets: int = 512
    bullet_radius: float = 0.01

    # ✅ speed/noise are FIXED (no curriculum)
    bullet_speed_min: float = 0.1
    bullet_speed_max: float = 0.2
    aim_noise: float = 0.50

    # ✅ only spawn_rate is scheduled by (curriculum/manual diff)
    spawn_rate: float = 30.0

    # observation
    obs_k: int = 32

    # reward shaping
    alive_reward: float = 0.01
    hit_penalty: float = 1.0
    move_penalty: float = 0.0005
    wall_penalty: float = 0.002

    near_miss_enabled: bool = True
    near_miss_margin: float = 0.03
    near_miss_coef: float = 0.05

    clear_bonus: float = 0.25

    # Curriculum (only affects spawn_rate)
    curriculum: bool = True
    curriculum_episodes: int = 800
    spawn_rate_start: float = 3.0

    # (kept for compatibility; ignored)
    bullet_speed_start_factor: float = 1.0
    aim_noise_start_factor: float = 1.0

    seed: int = 0

    # Rendering (tkinter)
    render_size: int = 720
    render_fps: int = 60
    render_draw_trails: bool = False


class VecShooterEnvTorch:
    """
    Torch-only vectorized Shooter env.
    - Runs N envs in parallel on a single device (cuda recommended)
    - No numpy. No python loops over bullets (only tensor ops).
    - Bullets are fixed-size (N,B,2) with alive mask (N,B).

    Rendering:
      - Provided via tkinter, BUT only supported for n_envs == 1.

    Manual difficulty:
      - Trainer can call set_manual_difficulty(diff01 in [0,1]) to override episode-based curriculum.
      - If cfg.curriculum is False -> diff01 is controlled only by manual difficulty (default 1.0 unless set).
      - If cfg.curriculum is True  -> diff01 uses episode-based curriculum AND can be overridden by manual if enabled.

    Difficulty scheduling in this version:
      - ✅ ONLY spawn_rate is scheduled (spawn_rate_start -> spawn_rate).
      - bullet speeds + aim_noise are fixed (no curriculum scaling).
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

        self.player_pos = torch.zeros((N, 2), device=self.device, dtype=torch.float32)
        self.bul_pos = torch.zeros((N, B, 2), device=self.device, dtype=torch.float32)
        self.bul_vel = torch.zeros((N, B, 2), device=self.device, dtype=torch.float32)
        self.bul_alive = torch.zeros((N, B), device=self.device, dtype=torch.bool)

        self.step_count = torch.zeros((N,), device=self.device, dtype=torch.int32)
        self.t = torch.zeros((N,), device=self.device, dtype=torch.float32)

        self.last_hit = torch.zeros((N,), device=self.device, dtype=torch.bool)

        self.episode_count = torch.zeros((N,), device=self.device, dtype=torch.int32)
        self.diff01 = torch.zeros((N,), device=self.device, dtype=torch.float32)

        self.action_dim = int(self.actions.shape[0])
        self.obs_dim = 2 + 2 + 1 + (cfg.obs_k * 4)

        # ---- manual difficulty override (0..1)
        self._manual_diff01 = torch.ones((N,), device=self.device, dtype=torch.float32)
        self._manual_diff_enabled = False

        # ---- render state (lazy)
        self._ui_inited = False
        self._tk = None
        self._root = None
        self._canvas = None
        self._hud_var = None
        self._last_render_ts = 0.0
        self._item_border = None
        self._item_player = None
        self._item_bullets = []
        self._item_trail = []
        self._trail = []

    # -------------------------
    # manual difficulty API (for trainer)
    # -------------------------
    def set_manual_difficulty(self, diff01: float, enabled: bool = True) -> None:
        """
        Set a global/manual difficulty in [0,1] applied to all envs.
        When enabled=True, diff01 used in _current_difficulty_params() becomes manual value.
        """
        d = float(diff01)
        if d != d:  # NaN guard
            d = 0.0
        d = max(0.0, min(1.0, d))
        self._manual_diff01.fill_(d)
        self._manual_diff_enabled = bool(enabled)
        # keep diff01 synced for debug/info
        self.diff01.fill_(d)

    def get_manual_difficulty(self) -> float:
        return float(self._manual_diff01[0].detach().float().cpu().item())

    @torch.no_grad()
    def get_spawn_rate_s(self) -> float:
        """
        Current (effective) spawn rate in bullets/sec for env0, after curriculum/manual diff applied.
        """
        spawn = self._current_spawn_rate()
        return float(spawn[0].detach().float().cpu().item())

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

        c = float(self.cfg.world_size) * 0.5
        noise = torch.randn((int(mask.sum().item()), 2), device=self.device, generator=self._gen) * 0.02
        pos = torch.tensor([c, c], device=self.device).view(1, 2) + noise
        pos = pos.clamp(0.0, float(self.cfg.world_size))

        self.player_pos[mask] = pos

        self.bul_alive[mask] = False
        self.bul_pos[mask] = 0.0
        self.bul_vel[mask] = 0.0

        if self.cfg.render_draw_trails and self.n == 1 and bool(mask[0].item()):
            self._trail.clear()

        obs = self._get_obs()
        info = {
            "t": self.t.clone(),
            "episode": self.episode_count.clone(),
            "diff01": self.diff01.clone(),
        }
        return obs, info

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        cfg = self.cfg
        N = self.n

        action = action.to(self.device).long().clamp(0, self.action_dim - 1)

        self.step_count += 1
        self.t += float(cfg.dt)

        move_dir = self.actions[action]
        nrm = torch.linalg.norm(move_dir, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir = move_dir / nrm

        dp = move_dir * (float(cfg.player_speed) * float(cfg.dt))
        self.player_pos += dp
        self.player_pos.clamp_(0.0, float(cfg.world_size))

        self._spawn_bullets()
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
            "diff01": self.diff01.clone(),
        }
        return obs, rew, done, info

    # -------------------------
    # curriculum / difficulty
    # -------------------------
    def _update_curriculum(self, mask: torch.Tensor) -> None:
        cfg = self.cfg

        if self._manual_diff_enabled:
            self.diff01[mask] = self._manual_diff01[mask]
            return

        if not cfg.curriculum:
            self.diff01[mask] = 1.0
            return

        denom = max(1, int(cfg.curriculum_episodes))
        d = (self.episode_count[mask].float() - 1.0) / float(denom)
        self.diff01[mask] = d.clamp(0.0, 1.0)

    def _effective_diff01(self) -> torch.Tensor:
        if self._manual_diff_enabled:
            return self._manual_diff01
        return self.diff01

    def _current_spawn_rate(self) -> torch.Tensor:
        """
        ✅ the only scheduled difficulty parameter.
        """
        cfg = self.cfg
        d = self._effective_diff01()
        return (1.0 - d) * float(cfg.spawn_rate_start) + d * float(cfg.spawn_rate)

    # -------------------------
    # internals
    # -------------------------
    @torch.no_grad()
    def _spawn_bullets(self) -> None:
        cfg = self.cfg
        N = self.n
        B = int(cfg.max_bullets)
        w = float(cfg.world_size)

        # ✅ spawn rate only is difficulty-dependent
        spawn_rate = self._current_spawn_rate()
        lam = spawn_rate * float(cfg.dt)

        n_spawn = torch.poisson(lam, generator=self._gen).to(torch.int64)
        if int(n_spawn.max().item()) <= 0:
            return

        free = ~self.bul_alive
        free_count = free.sum(dim=1)
        n_spawn = torch.minimum(n_spawn, free_count.to(torch.int64))
        if int(n_spawn.max().item()) <= 0:
            return

        idx = torch.arange(B, device=self.device).view(1, B).expand(N, B)
        big = torch.full((N, B), B + 1, device=self.device, dtype=torch.int64)
        free_idx = torch.where(free, idx, big)
        _, free_pos = torch.sort(free_idx, dim=1)

        max_k = int(n_spawn.max().item())
        spawn_slots = free_pos[:, :max_k]

        k_idx = torch.arange(max_k, device=self.device).view(1, max_k).expand(N, max_k)
        do = k_idx < n_spawn.view(N, 1)
        if not torch.any(do):
            return

        edge = torch.randint(0, 4, (N, max_k), device=self.device, generator=self._gen)
        u = torch.rand((N, max_k), device=self.device, generator=self._gen)

        pos = torch.zeros((N, max_k, 2), device=self.device, dtype=torch.float32)

        m = edge == 0
        pos[m, 0] = 0.0
        pos[m, 1] = u[m] * w

        m = edge == 1
        pos[m, 0] = w
        pos[m, 1] = u[m] * w

        m = edge == 2
        pos[m, 0] = u[m] * w
        pos[m, 1] = 0.0

        m = edge == 3
        pos[m, 0] = u[m] * w
        pos[m, 1] = w

        # ✅ speed + aim noise are FIXED (no curriculum scaling)
        to_p = self.player_pos.view(N, 1, 2) - pos
        ang = torch.atan2(to_p[..., 1], to_p[..., 0])
        ang = ang + torch.randn((N, max_k), device=self.device, generator=self._gen) * float(cfg.aim_noise)

        sp_min = float(cfg.bullet_speed_min)
        sp_max = float(cfg.bullet_speed_max)
        spd = torch.rand((N, max_k), device=self.device, generator=self._gen) * (sp_max - sp_min) + sp_min
        vel = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * spd.unsqueeze(-1)

        env_i = torch.arange(N, device=self.device).view(N, 1).expand(N, max_k)
        bi = spawn_slots

        env_i = env_i[do]
        bi = bi[do]
        pos2 = pos[do]
        vel2 = vel[do]

        self.bul_alive[env_i, bi] = True
        self.bul_pos[env_i, bi] = pos2
        self.bul_vel[env_i, bi] = vel2

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

    @torch.no_grad()
    def _get_obs(self) -> torch.Tensor:
        cfg = self.cfg
        w = float(cfg.world_size)

        p = self.player_pos / w
        p_center = p - 0.5
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

        obs = torch.cat([p, p_center, t01, feat.reshape(self.n, -1)], dim=1)
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
        alive0 = self.bul_alive[0].detach().cpu()
        bul0 = self.bul_pos[0].detach().float().cpu()

        def to_px(xy: torch.Tensor) -> Tuple[float, float]:
            x = (float(xy[0].item()) / w) * (W - 1)
            y = (1.0 - float(xy[1].item()) / w) * (W - 1)
            return x, y

        spawn_rate = self._current_spawn_rate()
        hud_txt = (
            f"EP {int(self.episode_count[0].item())}  step {int(self.step_count[0].item())}  t={float(self.t[0].item()):.2f}s  hit={int(self.last_hit[0].item())}\n"
            f"bullets={int(alive0.sum().item())}  diff={float(self._effective_diff01()[0].item()):.2f}\n"
            f"spawn={float(spawn_rate[0].item()):.2f}/s"
        )
        try:
            self._hud_var.set(hud_txt)
        except Exception:
            pass

        px, py = to_px(p0)
        pr = max(3, int(self.cfg.player_radius / w * W))
        canvas.coords(self._item_player, px - pr, py - pr, px + pr, py + pr)

        alive_idx = torch.where(alive0)[0]
        br = max(2, int(self.cfg.bullet_radius / w * W))

        for it in self._item_bullets:
            canvas.itemconfigure(it, state="hidden")

        for j in range(min(int(alive_idx.numel()), len(self._item_bullets))):
            i = int(alive_idx[j].item())
            bx, by = to_px(bul0[i])
            it = self._item_bullets[j]
            canvas.coords(it, bx - br, by - br, bx + br, by + br)
            canvas.itemconfigure(it, state="normal")

        if self.cfg.render_draw_trails:
            pts = self._trail[-600:] if len(self._trail) > 0 else []
            for it in self._item_trail:
                canvas.itemconfigure(it, state="hidden")
            need = max(0, len(pts) - 1)
            while len(self._item_trail) < need:
                self._item_trail.append(self._canvas.create_line(0, 0, 0, 0, fill="#7a7a86", width=1))
            for k in range(need):
                x0 = (pts[k][0] / w) * (W - 1)
                y0 = (1.0 - pts[k][1] / w) * (W - 1)
                x1 = (pts[k + 1][0] / w) * (W - 1)
                y1 = (1.0 - pts[k + 1][1] / w) * (W - 1)
                it = self._item_trail[k]
                canvas.coords(it, x0, y0, x1, y1)
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
        self._root.title("VecShooterEnvTorch (render)")
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
        self._item_bullets = []
        self._item_trail = []
        self._trail = []
