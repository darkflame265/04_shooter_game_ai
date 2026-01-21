from __future__ import annotations

import math
import torch
from torch_pattern_base import BulletPattern, PatternContext


class Pattern2RosetteVolley(BulletPattern):
    """
    Env 쪽에서 k_use를 8의 배수로 잘라서 넣어주면,
    여기서는 "8방향 기본 + rosette modulation"만 담당.
    """
    def __init__(self, phase_speed2: float, petals2: int, twist2: float, speed_mod2: float):
        self.phase_speed2 = float(phase_speed2)
        self.petals2 = int(petals2)
        self.twist2 = float(twist2)
        self.speed_mod2 = float(speed_mod2)

    @torch.no_grad()
    def angles_speeds(self, ctx: PatternContext):
        N = ctx.phase.shape[0]
        device = ctx.phase.device
        t_steps = ctx.step_count.to(torch.float32)

        base_dirs = (ctx.k_idx % 8).to(torch.float32)
        ang_base = (2.0 * math.pi) * (base_dirs / 8.0)

        ang = ctx.phase.view(N, 1).expand(N, ctx.max_k) + ang_base

        m = max(0, int(self.petals2))
        tw = float(self.twist2)
        if m > 0 and abs(tw) > 1e-12:
            t2 = t_steps.view(N, 1)
            ang = ang + tw * torch.sin(float(m) * ang_base + 0.015 * t2).to(torch.float32)

        sp_min = float(ctx.bullet_speed_min)
        sp_max = float(ctx.bullet_speed_max)
        base = 0.5 * (sp_min + sp_max)
        amp = 0.5 * (sp_max - sp_min)

        sm = float(self.speed_mod2)
        if abs(sm) > 1e-12 and m > 0:
            ph = ctx.phase.view(N, 1)
            shape = torch.cos(float(m) * ang + ph).clamp(-1.0, 1.0)
            spd = (base + amp * (sm * shape)).clamp(sp_min, sp_max)
        else:
            spd = torch.full((N, ctx.max_k), base, device=device, dtype=torch.float32)

        new_phase = ctx.phase + self.phase_speed2 * (ctx.k_use > 0).to(torch.float32)
        return ang.to(torch.float32), spd.to(torch.float32), new_phase
