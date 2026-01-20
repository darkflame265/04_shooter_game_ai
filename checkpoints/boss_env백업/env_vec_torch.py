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
    player_speed: float = 0.1
    player_radius: float = 0.01

    # bullets (capacity)
    max_bullets: int = 2048
    bullet_radius: float = 0.01

    # bullet speed is FIXED by cfg (no difficulty scaling)
    bullet_speed_min: float = 0.1
    bullet_speed_max: float = 0.2

    # aim noise (keep variable, but default 0 for clean boss patterns)
    aim_noise: float = 0.0

    # keep for train compatibility: used as "target/final density"
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

    # Curriculum (only affects density/intensity)
    curriculum: bool = True
    curriculum_episodes: int = 800
    spawn_rate_start: float = 3.0  # "초기 난이도 지표" (실제 계산에 반영)

    # (kept for compatibility; ignored)
    bullet_speed_start_factor: float = 1.0
    aim_noise_start_factor: float = 1.0

    # -------------------------
    # Boss params
    # -------------------------
    boss_y: float = 0.92          # boss fixed Y (top area)
    boss_x_amp: float = 0.38      # horizontal amplitude around center
    boss_move_hz: float = 0.1    # user-controlled (NOT scaled by difficulty)

    # pattern
    # 0: wobble ring
    # 1: simple rotating ring
    # 2: flower/rosette (new)
    pattern_id: int = 2
    phase_speed0: float = 0.08
    phase_speed1: float = 0.12

    # flower/rosette params (new)
    flower_petals: int = 6            # m (number of petals)
    flower_phase_speed: float = 0.10  # phase advance per fire (scaled by intensity slightly)
    flower_speed_mod: float = 0.55    # speed modulation strength (0..~0.9 recommended)
    flower_twist: float = 0.35        # adds secondary swirl (0..1)

    # density mapping (deterministic)
    intensity_gamma: float = 1.2
    intensity_min: float = 0.25
    intensity_max: float = 10.0

    # baseline (difficulty=~1.0 기준 느낌)
    base_rate_steps: int = 6
    base_n0: int = 14
    base_n1: int = 20

    # clamp for safety / bullet cap
    rate_steps_min: int = 4
    rate_steps_max: int = 60
    n0_max: int = 96
    n1_max: int = 110

    # Rendering (tkinter)
    render_size: int = 720
    render_fps: int = 60
    render_draw_trails: bool = False

    seed: int = 0


class VecShooterEnvTorch:
    """
    Boss-pattern simulator (Torch-only vectorized env).

    Deterministic firing schedule:
      fire every `rate_steps` steps
      spawn exactly `n` bullets in a ring from boss position

    Difficulty/Curriculum:
      diff01 -> intensity -> (rate_steps down, n up, phase denser)
      bullet speed is NOT scaled by difficulty

    Added:
      pattern_id=2: flower/rosette style (petal-like) with fully deterministic speeds.
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
        self.diff01 = torch.zeros((N,), device=self.device, dtype=torch.float32)

        # pattern internal phase (per env)
        self.phase = torch.zeros((N,), device=self.device, dtype=torch.float32)

        self.action_dim = int(self.actions.shape[0])

        # obs: player(2) + player_center(2) + boss(2) + t01(1) + K*(rel(2)+vel(2))
        self.obs_dim = 2 + 2 + 2 + 1 + (cfg.obs_k * 4)

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
        self._item_boss = None
        self._item_bullets = []
        self._item_trail = []
        self._trail = []

    # -------------------------
    # manual difficulty API (for trainer)
    # -------------------------
    def set_manual_difficulty(self, diff01: float, enabled: bool = True) -> None:
        d = float(diff01)
        if d != d:
            d = 0.0
        d = max(0.0, min(1.0, d))
        self._manual_diff01.fill_(d)
        self._manual_diff_enabled = bool(enabled)
        self.diff01.fill_(d)

    def get_manual_difficulty(self) -> float:
        return float(self._manual_diff01[0].detach().float().cpu().item())

    @torch.no_grad()
    def get_spawn_rate_s(self) -> float:
        """
        train.py가 난이도 지표로 출력/비교하는 값.
        이 env는 Poisson이 아니라 '주기(rate_steps)마다 n발'이므로,
        현재 diff01 기준으로 계산된 평균 bullets/sec를 반환.
        """
        rate_steps, n_bul = self._current_density_params()
        dt = float(self.cfg.dt)
        spawn = n_bul.to(torch.float32) / (rate_steps.to(torch.float32) * dt + 1e-12)
        return float(spawn[0].detach().float().cpu().item())

    # -------------------------
    # RNG state API (for checkpoint reproducibility)
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

        # player spawn near bottom-center-ish
        w = float(self.cfg.world_size)
        cx = 0.5 * w
        cy = 0.20 * w
        noise = torch.randn((int(mask.sum().item()), 2), device=self.device, generator=self._gen) * 0.03
        pos = torch.tensor([cx, cy], device=self.device).view(1, 2) + noise
        pos = pos.clamp(0.0, w)
        self.player_pos[mask] = pos

        # phase init (keep some variety; set to 0 if you want fully identical episodes)
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
        info = {"t": self.t.clone(), "episode": self.episode_count.clone(), "diff01": self.diff01.clone()}
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
        self._spawn_bullets_from_boss_deterministic()
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

    @torch.no_grad()
    def _current_intensity(self) -> torch.Tensor:
        """
        diff01 -> intensity (>= intensity_min)
        - diff01이 0이어도 탄막이 아예 죽지 않게 intensity_min 적용
        """
        cfg = self.cfg
        d = self._effective_diff01()
        d5 = d * 5.0
        gamma = float(cfg.intensity_gamma)
        inten = torch.pow(d5.clamp_min(0.0), gamma)
        inten = inten.clamp(float(cfg.intensity_min), float(cfg.intensity_max))
        return inten

    @torch.no_grad()
    def _current_density_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          rate_steps: (N,) int64  - fire every rate_steps
          n_bullets:  (N,) int64  - bullets per fire
        """
        cfg = self.cfg
        N = self.n
        inten = self._current_intensity()

        denom = (0.15 + 0.85 * inten).clamp_min(1e-6)
        base_rate = float(cfg.base_rate_steps)
        rate = torch.round(base_rate / denom).to(torch.int64)
        rate = rate.clamp(int(cfg.rate_steps_min), int(cfg.rate_steps_max))

        # n = 1 + (base_n - 1) * intensity
        if int(cfg.pattern_id) == 0:
            base_n = float(cfg.base_n0)
            n = torch.round(1.0 + (base_n - 1.0) * inten).to(torch.int64)
            n = n.clamp(8, int(cfg.n0_max))
        else:
            base_n = float(cfg.base_n1)
            n = torch.round(1.0 + (base_n - 1.0) * inten).to(torch.int64)
            n = n.clamp(8, int(cfg.n1_max))

        n = torch.minimum(n, torch.full((N,), int(cfg.max_bullets), device=self.device, dtype=torch.int64))
        return rate, n

    # -------------------------
    # boss motion (user-controlled)
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
    # bullets (deterministic boss pattern)
    # -------------------------
    @torch.no_grad()
    def _spawn_bullets_from_boss_deterministic(self) -> None:
        """
        Deterministic patterns:
          pattern_id 0: wobble ring + phase increments
          pattern_id 1: rotating ring (a0 = phase + phase_speed1*t_steps)
          pattern_id 2: flower/rosette (petal-like) via speed modulation:
              speed(angle) = base + mod*cos(m*angle + phase) + small swirl term
              -> produces clear petal lobes over time (Touhou-like feel)

        All bullet speeds are deterministic functions (no torch.rand).
        """
        cfg = self.cfg
        N = self.n
        B = int(cfg.max_bullets)

        rate_steps, n_bul = self._current_density_params()

        sc = self.step_count.to(torch.int64)
        fire = (sc % rate_steps) == 0
        if not torch.any(fire):
            return

        # free slots
        free = ~self.bul_alive
        free_count = free.sum(dim=1).to(torch.int64)

        n_use = torch.minimum(n_bul, free_count)
        fire = fire & (n_use > 0)
        if not torch.any(fire):
            return

        max_k = int(n_use[fire].max().item())
        if max_k <= 0:
            return

        # deterministic free-slot picking: take lowest indices
        idx = torch.arange(B, device=self.device).view(1, B).expand(N, B)
        big = torch.full((N, B), B + 1, device=self.device, dtype=torch.int64)
        free_idx = torch.where(free, idx, big)
        _, free_pos = torch.sort(free_idx, dim=1)
        spawn_slots = free_pos[:, :max_k]  # (N, max_k)

        k_idx = torch.arange(max_k, device=self.device, dtype=torch.int64).view(1, max_k).expand(N, max_k)
        do = (k_idx < n_use.view(N, 1)) & fire.view(N, 1)
        if not torch.any(do):
            return

        # base ring angles: 2pi*(k/n)
        kf = k_idx.to(torch.float32)
        nf = n_use.to(torch.float32).clamp_min(1.0).view(N, 1)
        ang_base = (2.0 * math.pi) * (kf / nf)  # (N, max_k)

        t_steps = self.step_count.to(torch.float32)
        inten = self._current_intensity()

        # pattern-specific angle offset and speed
        pid = int(cfg.pattern_id)

        if pid == 0:
            a0 = self.phase + 0.25 * torch.sin(0.03 * t_steps)
            a0 = a0.view(N, 1).expand(N, max_k)
            ang = a0 + ang_base

            # phase increment only on fire steps
            phase_scale = (0.60 + 0.10 * inten).clamp(0.60, 1.60)
            phase_inc = float(cfg.phase_speed0) * phase_scale
            self.phase = self.phase + phase_inc * fire.to(torch.float32)

            # deterministic speed: interpolate by k position (no randomness)
            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            frac = (kf / max(1.0, float(max_k - 1))).clamp(0.0, 1.0)
            spd = sp_min + (sp_max - sp_min) * frac

        elif pid == 1:
            a0 = self.phase + float(cfg.phase_speed1) * t_steps
            a0 = a0.view(N, 1).expand(N, max_k)
            ang = a0 + ang_base

            # keep phase fixed (or minimal drift if you want)
            self.phase = self.phase + 0.0

            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            # deterministic saw-like spread
            frac = ((kf * 1.6180339) % max(1.0, float(max_k))).clamp(0.0, float(max_k))
            frac = (frac / max(1.0, float(max_k - 1))).clamp(0.0, 1.0)
            spd = sp_min + (sp_max - sp_min) * frac

        else:
            # -------- pattern_id == 2: flower/rosette --------
            m = max(2, int(cfg.flower_petals))

            # ✅ broadcast-safe time term
            t2 = t_steps.view(N, 1)  # (N,1)

            # base rotation
            a0 = self.phase.view(N, 1) + 0.02 * t2
            a0 = a0.expand(N, max_k)

            # extra twist: small angular perturbation dependent on angle itself
            twist = float(cfg.flower_twist)
            ang = a0 + ang_base + twist * torch.sin(float(m) * ang_base + 0.015 * t2).to(torch.float32)

            # deterministic speed modulation => petal lobes
            sp_min = float(cfg.bullet_speed_min)
            sp_max = float(cfg.bullet_speed_max)
            base = 0.5 * (sp_min + sp_max)
            amp = 0.5 * (sp_max - sp_min)

            mod = float(cfg.flower_speed_mod)
            # cos(m*angle + phase) creates m petals
            ph = self.phase.view(N, 1)
            pet = torch.cos(float(m) * ang + ph)
            # slight secondary harmonic (makes petals sharper without noise)
            pet2 = 0.35 * torch.cos(float(2 * m) * ang - 0.5 * ph)

            # clamp modulation to avoid negative speeds / too wide range
            shape = (pet + pet2).clamp(-1.0, 1.0)
            spd = base + amp * (mod * shape)
            spd = spd.clamp(sp_min, sp_max)

            # phase advance on fire only; slightly intensity-scaled
            phase_scale = (0.85 + 0.06 * inten).clamp(0.85, 1.35)
            phase_inc = float(cfg.flower_phase_speed) * phase_scale
            self.phase = self.phase + phase_inc * fire.to(torch.float32)


        # optional small aim noise (still deterministic given RNG state)
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
        rate_steps, n_bul = self._current_density_params()
        hud_txt = (
            f"EP {int(self.episode_count[0].item())}  step {int(self.step_count[0].item())}  "
            f"t={float(self.t[0].item()):.2f}s  hit={int(self.last_hit[0].item())}\n"
            f"bullets={int(alive0.sum().item())}  diff={float(self._effective_diff01()[0].item()):.2f}  "
            f"spawn~={spawn_s:.2f}/s  rate={int(rate_steps[0].item())}  n={int(n_bul[0].item())}\n"
            f"boss=({float(b0[0].item()):.2f},{float(b0[1].item()):.2f})  pattern_id={int(self.cfg.pattern_id)}"
        )
        try:
            self._hud_var.set(hud_txt)
        except Exception:
            pass

        # player
        px, py = to_px_xy(p0)
        pr = max(3, int(self.cfg.player_radius / w * W))
        canvas.coords(self._item_player, px - pr, py - pr, px + pr, py + pr)

        # boss
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
        self._root.title("VecShooterEnvTorch (boss deterministic)")
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
