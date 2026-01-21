from __future__ import annotations

from typing import Optional, Tuple
import time
import torch


class TkRenderer:
    """
    Tk renderer for N==1 only. 완전 분리 버전.
    env는 state만 넘겨주면 됨.
    """
    def __init__(self, render_size: int, render_fps: float, max_bullets: int):
        self.render_size = int(render_size)
        self.render_fps = float(render_fps)
        self.max_bullets = int(max_bullets)

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
        self._last_render_ts = 0.0

    def _lazy_init(self) -> None:
        if self._ui_inited:
            return
        try:
            import tkinter as tk
        except Exception as e:
            raise RuntimeError("tkinter is not available.") from e

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("VecShooterEnvTorch")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        W = int(self.render_size)

        self._hud_var = tk.StringVar(value="")
        hud = tk.Label(self._root, textvariable=self._hud_var, justify="left", anchor="w", font=("Consolas", 10))
        hud.pack(fill="x")

        canvas = tk.Canvas(self._root, width=W, height=W, bg="#121216", highlightthickness=0)
        canvas.pack()

        self._item_border = canvas.create_rectangle(2, 2, W - 2, W - 2, outline="#505058", width=2)

        self._item_bullets = []
        for _ in range(int(self.max_bullets)):
            it = canvas.create_oval(0, 0, 0, 0, fill="#f0e65a", outline="", state="hidden")
            self._item_bullets.append(it)

        self._item_player = canvas.create_oval(0, 0, 0, 0, fill="#5adcff", outline="")
        self._item_boss = canvas.create_oval(0, 0, 0, 0, fill="#ff5ac8", outline="")

        self._canvas = canvas
        self._ui_inited = True
        self._last_render_ts = 0.0

    @torch.no_grad()
    def render(
        self,
        *,
        world_size: float,
        player_radius: float,
        bullet_radius: float,
        pattern_id: int,
        episode_count0: int,
        step0: int,
        t0: float,
        hit0: int,
        bullets0: int,
        spawn_rate0: float,
        boss_xy0: Tuple[float, float],
        player_pos0: torch.Tensor,   # (2,) cpu/torch ok
        boss_pos0: torch.Tensor,     # (2,)
        bul_alive0: torch.Tensor,    # (B,) bool
        bul_pos0: torch.Tensor,      # (B,2)
    ) -> None:
        self._lazy_init()

        now = time.time()
        min_dt = 1.0 / max(1.0, float(self.render_fps))
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

        W = int(self.render_size)
        w = float(world_size)

        def to_px_xy(xy0: torch.Tensor) -> Tuple[float, float]:
            x = (float(xy0[0].item()) / w) * (W - 1)
            y = (1.0 - float(xy0[1].item()) / w) * (W - 1)
            return x, y

        hud_txt = (
            f"EP {episode_count0}  step {step0}  t={t0:.2f}s  hit={hit0}\n"
            f"bullets={bullets0}  spawn={spawn_rate0:.2f}/s  "
            f"boss=({boss_xy0[0]:.2f},{boss_xy0[1]:.2f})  pattern_id={pattern_id}"
        )
        try:
            self._hud_var.set(hud_txt)
        except Exception:
            pass

        px, py = to_px_xy(player_pos0)
        pr = max(3, int(float(player_radius) / w * W))
        canvas.coords(self._item_player, px - pr, py - pr, px + pr, py + pr)

        bx, by = to_px_xy(boss_pos0)
        br0 = max(4, int(float(player_radius) / w * W) + 2)
        canvas.coords(self._item_boss, bx - br0, by - br0, bx + br0, by + br0)

        alive_idx = torch.where(bul_alive0)[0]
        br = max(2, int(float(bullet_radius) / w * W))

        for it in self._item_bullets:
            canvas.itemconfigure(it, state="hidden")

        for j in range(min(int(alive_idx.numel()), len(self._item_bullets))):
            i = int(alive_idx[j].item())
            x, y = to_px_xy(bul_pos0[i])
            it = self._item_bullets[j]
            canvas.coords(it, x - br, y - br, x + br, y + br)
            canvas.itemconfigure(it, state="normal")

        try:
            root.update()
        except Exception:
            self.close()
