from __future__ import annotations

import math
import torch
from torch_pattern_base import BulletPattern, PatternContext


class Pattern1RotatingRing(BulletPattern):
    def __init__(self, phase_speed1: float):
        self.phase_speed1 = float(phase_speed1)

    @torch.no_grad()
    def angles_speeds(self, ctx: PatternContext):
        N = ctx.phase.shape[0]
        t_steps = ctx.step_count.to(torch.float32)

        kf = ctx.k_idx.to(torch.float32)
        nf = ctx.k_use.to(torch.float32).clamp_min(1.0).view(N, 1)

        ang_base = (2.0 * math.pi) * (kf / nf)

        a0 = ctx.phase + self.phase_speed1 * t_steps
        a0 = a0.view(N, 1).expand(N, ctx.max_k)
        ang = a0 + ang_base

        # phase is not advanced here (keep as-is)
        new_phase = ctx.phase

        sp_min = float(ctx.bullet_speed_min)
        sp_max = float(ctx.bullet_speed_max)
        frac = ((kf * 1.6180339) % max(1.0, float(ctx.max_k))).clamp(0.0, float(ctx.max_k))
        frac = (frac / max(1.0, float(ctx.max_k - 1))).clamp(0.0, 1.0)
        spd = sp_min + (sp_max - sp_min) * frac

        return ang, spd.to(torch.float32), new_phase
