from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import time
import numpy as np


@dataclass
class EnvConfig:
    # world
    world_size: float = 1.0  # coordinate range: [0, world_size]
    dt: float = 1.0 / 60.0
    max_steps: int = 60 * 20  # 20 seconds

    # player
    player_speed: float = 0.55  # units / sec
    player_radius: float = 0.01

    # bullets (TARGET / final difficulty)
    max_bullets: int = 64
    bullet_radius: float = 0.015
    bullet_speed_min: float = 0.25
    bullet_speed_max: float = 0.80
    spawn_rate: float = 6.0  # bullets / sec (Poisson-like)
    aim_noise: float = 0.15  # radians

    # observation
    obs_k: int = 16  # number of nearest bullets encoded

    # reward shaping
    alive_reward: float = 0.01
    hit_penalty: float = 1.0
    move_penalty: float = 0.0005
    wall_penalty: float = 0.002  # penalize hugging boundary

    # --------
    # Curriculum (start easy -> ramp to target)
    # --------
    curriculum: bool = True
    curriculum_episodes: int = 800  # number of episodes to reach target difficulty
    spawn_rate_start: float = 2.0   # easier starting spawn rate
    bullet_speed_start_factor: float = 0.55  # start speeds scaled down
    aim_noise_start_factor: float = 2.0      # start with more noise (easier)

    # --------
    # Rendering (tkinter; no extra deps)
    # --------
    render_size: int = 720      # window width/height in pixels
    render_fps: int = 60        # cap render refresh (only affects display)
    render_draw_trails: bool = False  # keep false for speed

    seed: int = 0


class ShooterEnv:
    """
    Simple 2D survival shooter simulator (numpy) + tkinter renderer (optional).
    - Player moves in a unit square.
    - Bullets spawn at edges and fly roughly toward the player.
    - Episode ends on hit or timeout.

    Discrete action space (9):
      0: STOP
      1: UP
      2: DOWN
      3: LEFT
      4: RIGHT
      5: UP_LEFT
      6: UP_RIGHT
      7: DOWN_LEFT
      8: DOWN_RIGHT
    """

    ACTIONS = np.array(
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
        dtype=np.float32,
    )

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.player_pos = np.zeros(2, dtype=np.float32)
        self.bul_pos = np.zeros((cfg.max_bullets, 2), dtype=np.float32)
        self.bul_vel = np.zeros((cfg.max_bullets, 2), dtype=np.float32)
        self.bul_alive = np.zeros((cfg.max_bullets,), dtype=np.bool_)

        self.step_count = 0
        self.t = 0.0

        # stats
        self.last_hit = False

        # curriculum state
        self.episode_count = 0
        self.diff01 = 0.0  # 0..1 difficulty progress

        # derived
        self.action_dim = int(self.ACTIONS.shape[0])
        self.obs_dim = 2 + 2 + 1 + (cfg.obs_k * 4)  # player xy + player xy-0.5 + time + K*(dx,dy,vx,vy)

        # rendering state (tkinter; lazy init)
        self._ui_inited = False
        self._tk = None
        self._root = None
        self._canvas = None
        self._hud = None
        self._last_render_ts = 0.0

        # cached canvas items
        self._item_border = None
        self._item_player = None
        self._item_bullets = []
        self._item_trail = []

        # optional trail points (world coords)
        self._trail = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_count += 1
        self._update_curriculum()

        self.step_count = 0
        self.t = 0.0
        self.last_hit = False

        # player starts near center
        c = self.cfg.world_size * 0.5
        self.player_pos[:] = np.array([c, c], dtype=np.float32) + self.rng.normal(0.0, 0.02, size=(2,)).astype(np.float32)
        self.player_pos[:] = np.clip(self.player_pos, 0.0, self.cfg.world_size)

        # clear bullets
        self.bul_alive[:] = False
        self.bul_pos[:] = 0.0
        self.bul_vel[:] = 0.0

        if self.cfg.render_draw_trails:
            self._trail.clear()

        obs = self._get_obs()
        info = {"t": self.t, "episode": self.episode_count, "diff01": float(self.diff01)}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.cfg
        self.step_count += 1
        self.t += cfg.dt

        # ---- player move
        a = int(action)
        if a < 0 or a >= self.action_dim:
            a = 0

        move_dir = self.ACTIONS[a].copy()
        n = float(np.linalg.norm(move_dir))
        if n > 1e-6:
            move_dir /= n

        dp = move_dir * (cfg.player_speed * cfg.dt)
        self.player_pos += dp

        # boundary clamp
        self.player_pos[:] = np.clip(self.player_pos, 0.0, cfg.world_size)

        # ---- bullets
        self._spawn_bullets()
        self._advance_bullets()
        hit = self._check_hit()
        self.last_hit = bool(hit)

        done = False
        reward = 0.0

        # base reward
        reward += cfg.alive_reward

        # movement penalty (encourage efficiency; not too strong)
        reward -= cfg.move_penalty * float(np.linalg.norm(dp) / (cfg.player_speed * cfg.dt + 1e-8))

        # wall penalty: discourage boundary hugging
        margin = 0.08 * cfg.world_size
        d_left = self.player_pos[0]
        d_right = cfg.world_size - self.player_pos[0]
        d_bot = self.player_pos[1]
        d_top = cfg.world_size - self.player_pos[1]
        dmin = float(min(d_left, d_right, d_bot, d_top))
        if dmin < margin:
            reward -= cfg.wall_penalty * float((margin - dmin) / margin)

        if hit:
            reward -= cfg.hit_penalty
            done = True

        if self.step_count >= cfg.max_steps:
            done = True

        if cfg.render_draw_trails:
            self._trail.append(self.player_pos.copy())
            if len(self._trail) > 2000:
                self._trail = self._trail[-2000:]

        obs = self._get_obs()
        info = {
            "t": self.t,
            "step": self.step_count,
            "hit": bool(hit),
            "bullets": int(self.bul_alive.sum()),
            "episode": self.episode_count,
            "diff01": float(self.diff01),
        }
        return obs, float(reward), bool(done), info

    # -------------------------
    # curriculum
    # -------------------------
    def _update_curriculum(self) -> None:
        cfg = self.cfg
        if not cfg.curriculum:
            self.diff01 = 1.0
            return
        denom = max(1, int(cfg.curriculum_episodes))
        self.diff01 = float(min(1.0, (self.episode_count - 1) / denom))

    def _current_difficulty_params(self) -> Tuple[float, float, float, float]:
        """
        Returns (spawn_rate, bullet_speed_min, bullet_speed_max, aim_noise) for current diff01.
        - diff01=0: easy
        - diff01=1: target (cfg.spawn_rate, cfg.bullet_speed_min/max, cfg.aim_noise)
        """
        cfg = self.cfg
        d = float(self.diff01)

        spawn = float((1.0 - d) * cfg.spawn_rate_start + d * cfg.spawn_rate)

        s0 = float(cfg.bullet_speed_start_factor)
        sp_min = float(cfg.bullet_speed_min * ((1.0 - d) * s0 + d * 1.0))
        sp_max = float(cfg.bullet_speed_max * ((1.0 - d) * s0 + d * 1.0))

        n0 = float(cfg.aim_noise_start_factor)
        aim = float(cfg.aim_noise * ((1.0 - d) * n0 + d * 1.0))

        return spawn, sp_min, sp_max, aim

    # -------------------------
    # internals
    # -------------------------
    def _spawn_bullets(self) -> None:
        cfg = self.cfg
        spawn_rate, sp_min, sp_max, aim_noise = self._current_difficulty_params()

        lam = spawn_rate * cfg.dt
        n_spawn = int(self.rng.poisson(lam))
        if n_spawn <= 0:
            return

        free_idx = np.where(~self.bul_alive)[0]
        if free_idx.size == 0:
            return

        n_spawn = min(n_spawn, int(free_idx.size))
        idxs = free_idx[:n_spawn]

        w = cfg.world_size
        for i in idxs:
            edge = int(self.rng.integers(0, 4))
            u = float(self.rng.random())
            if edge == 0:      # left
                pos = np.array([0.0, u * w], dtype=np.float32)
            elif edge == 1:    # right
                pos = np.array([w, u * w], dtype=np.float32)
            elif edge == 2:    # bottom
                pos = np.array([u * w, 0.0], dtype=np.float32)
            else:              # top
                pos = np.array([u * w, w], dtype=np.float32)

            to_p = (self.player_pos - pos).astype(np.float32)
            ang = float(np.arctan2(to_p[1], to_p[0]))
            ang += float(self.rng.normal(0.0, aim_noise))

            spd = float(self.rng.uniform(sp_min, sp_max))
            vel = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32) * spd

            self.bul_alive[i] = True
            self.bul_pos[i] = pos
            self.bul_vel[i] = vel

    def _advance_bullets(self) -> None:
        cfg = self.cfg
        alive = self.bul_alive
        if not np.any(alive):
            return

        self.bul_pos[alive] += self.bul_vel[alive] * cfg.dt

        w = cfg.world_size
        pad = 0.2 * w
        x = self.bul_pos[:, 0]
        y = self.bul_pos[:, 1]
        out = (x < -pad) | (x > w + pad) | (y < -pad) | (y > w + pad)
        self.bul_alive &= ~out

    def _check_hit(self) -> bool:
        cfg = self.cfg
        alive = self.bul_alive
        if not np.any(alive):
            return False

        d = self.bul_pos[alive] - self.player_pos[None, :]
        dist2 = np.sum(d * d, axis=1)
        r = cfg.player_radius + cfg.bullet_radius
        return bool(np.any(dist2 <= (r * r)))

    def _get_obs(self) -> np.ndarray:
        """
        Observation: float32 vector
          - player xy in [0,1]
          - player xy centered (xy - 0.5)
          - time progress in [0,1]
          - K nearest bullets: (dx, dy, vx, vy) in roughly [-1,1] scale
        """
        cfg = self.cfg
        w = cfg.world_size

        p = self.player_pos / w
        p_center = p - 0.5
        t01 = np.array([min(1.0, self.step_count / max(1, cfg.max_steps))], dtype=np.float32)

        K = cfg.obs_k
        feat = np.zeros((K, 4), dtype=np.float32)

        alive_idx = np.where(self.bul_alive)[0]
        if alive_idx.size > 0:
            v_norm = max(cfg.bullet_speed_max, 1e-6)

            bul_p = self.bul_pos[alive_idx] / w
            rel = bul_p - p[None, :]
            dist2 = np.sum(rel * rel, axis=1)
            order = np.argsort(dist2)[:K]
            sel = alive_idx[order]

            rel = (self.bul_pos[sel] / w) - p[None, :]
            vel = self.bul_vel[sel] / v_norm

            feat[: rel.shape[0], 0:2] = rel.astype(np.float32)
            feat[: vel.shape[0], 2:4] = vel.astype(np.float32)

        obs = np.concatenate([p.astype(np.float32), p_center.astype(np.float32), t01, feat.reshape(-1)], axis=0)
        return obs.astype(np.float32)

    # -------------------------
    # tkinter rendering (optional; no extra deps)
    # -------------------------
    def render(self) -> None:
        """
        tkinter visualization. Call this each step if you want a live window.
        """
        self._ui_lazy_init()

        # FPS cap (render-only)
        now = time.time()
        min_dt = 1.0 / max(1, int(self.cfg.render_fps))
        if (now - self._last_render_ts) < min_dt:
            return
        self._last_render_ts = now

        root = self._root
        canvas = self._canvas

        # if window closed, just stop rendering
        if root is None or canvas is None:
            return

        try:
            root.update_idletasks()
        except Exception:
            return

        W = int(self.cfg.render_size)
        w = float(self.cfg.world_size)

        def to_px(pos: np.ndarray) -> Tuple[float, float]:
            # world (0..w) -> screen (0..W), y flipped
            x = (float(pos[0]) / w) * (W - 1)
            y = (1.0 - float(pos[1]) / w) * (W - 1)
            return x, y

        # update HUD
        spawn_rate, sp_min, sp_max, aim = self._current_difficulty_params()
        hud_txt = (
            f"EP {self.episode_count}  step {self.step_count}  t={self.t:.2f}s  hit={int(self.last_hit)}\n"
            f"bullets={int(self.bul_alive.sum())}  diff={self.diff01:.2f}\n"
            f"spawn={spawn_rate:.2f}/s  spd=[{sp_min:.2f},{sp_max:.2f}]  aim_noise={aim:.3f}"
        )
        try:
            self._hud_var.set(hud_txt)
        except Exception:
            pass

        # player
        px, py = to_px(self.player_pos)
        pr = max(3, int(self.cfg.player_radius / w * W))
        canvas.coords(self._item_player, px - pr, py - pr, px + pr, py + pr)

        # bullets: keep a pool of oval items up to max_bullets
        alive_idx = np.where(self.bul_alive)[0]
        br = max(2, int(self.cfg.bullet_radius / w * W))

        # hide all bullets first (cheap)
        for it in self._item_bullets:
            canvas.itemconfigure(it, state="hidden")

        # show alive bullets
        for j, i in enumerate(alive_idx.tolist()):
            if j >= len(self._item_bullets):
                break
            bx, by = to_px(self.bul_pos[i])
            it = self._item_bullets[j]
            canvas.coords(it, bx - br, by - br, bx + br, by + br)
            canvas.itemconfigure(it, state="normal")

        # optional trail (limited)
        if self.cfg.render_draw_trails:
            # rebuild occasionally; keep it simple
            pts = self._trail[-600:] if len(self._trail) > 0 else []
            # hide all
            for it in self._item_trail:
                canvas.itemconfigure(it, state="hidden")
            # draw lines segments
            need = max(0, len(pts) - 1)
            while len(self._item_trail) < need:
                self._item_trail.append(canvas.create_line(0, 0, 0, 0, fill="#7a7a86", width=1))
            for k in range(need):
                x0, y0 = to_px(pts[k])
                x1, y1 = to_px(pts[k + 1])
                it = self._item_trail[k]
                canvas.coords(it, x0, y0, x1, y1)
                canvas.itemconfigure(it, state="normal")

        # flush UI
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
            raise RuntimeError("tkinter is not available. On Windows it should be included with Python.") from e

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("ShooterEnv (tkinter)")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        W = int(self.cfg.render_size)

        # HUD
        self._hud_var = tk.StringVar(value="")
        hud = tk.Label(self._root, textvariable=self._hud_var, justify="left", anchor="w", font=("Consolas", 10))
        hud.pack(fill="x")

        # Canvas
        canvas = tk.Canvas(self._root, width=W, height=W, bg="#121216", highlightthickness=0)
        canvas.pack()

        # border
        self._item_border = canvas.create_rectangle(2, 2, W - 2, W - 2, outline="#505058", width=2)

        # pre-create bullets pool
        self._item_bullets = []
        for _ in range(int(self.cfg.max_bullets)):
            it = canvas.create_oval(0, 0, 0, 0, fill="#f0e65a", outline="", state="hidden")
            self._item_bullets.append(it)

        # player
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
        self._hud = None
        self._item_border = None
        self._item_player = None
        self._item_bullets = []
        self._item_trail = []
        self._trail = []
