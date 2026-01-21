from __future__ import annotations

import math
import torch
from torch_pattern_base import BulletPattern, PatternContext


class Pattern3ChaoticLissajousGate(BulletPattern):
    """
    deterministic(학습 가능) + 사람이 보기엔 불규칙/까다로운 패턴.
    """
    def __init__(
        self,
        phase_speed3: float,
        p3_aim_mix3: float,
        p3_sweep_amp3: float,
        p3_f1_3: float,
        p3_f2_3: float,
        p3_gate_hz3: float,
        p3_gate_duty3: float,
        p3_burst_mul3: float,
        p3_speed_mod3: float,
        p3_spread3: float,
    ):
        self.phase_speed3 = float(phase_speed3)

        self.aim_mix = float(p3_aim_mix3)
        self.sweep_amp = float(p3_sweep_amp3)
        self.f1 = float(p3_f1_3)
        self.f2 = float(p3_f2_3)
        self.gate_hz = float(p3_gate_hz3)
        self.gate_duty = float(p3_gate_duty3)
        self.burst_mul = float(p3_burst_mul3)
        self.speed_mod = float(p3_speed_mod3)
        self.spread = float(p3_spread3)

    @torch.no_grad()
    def angles_speeds(self, ctx: PatternContext):
        N = ctx.phase.shape[0]

        sp_min = float(ctx.bullet_speed_min)
        sp_max = float(ctx.bullet_speed_max)
        base_sp = 0.5 * (sp_min + sp_max)
        amp_sp = 0.5 * (sp_max - sp_min)

        # gate (0/1)
        gate_hz = max(1e-6, self.gate_hz)
        duty = max(0.0, min(1.0, self.gate_duty))
        gphase = (gate_hz * ctx.t) % 1.0
        gate1 = (gphase < duty).to(torch.float32)     # (N,)
        gate = gate1.view(N, 1).expand(N, ctx.max_k)  # (N,max_k)

        # aim angle
        d = (ctx.player_pos - ctx.boss_pos)  # (N,2)
        aim = torch.atan2(d[:, 1], d[:, 0]).view(N, 1).expand(N, ctx.max_k)

        # two-nozzle lissajous sweep
        nozzle = (ctx.k_idx % 2).to(torch.float32)  # (N,max_k) in {0,1}
        tt = ctx.t.view(N, 1)

        s = torch.sin(2.0 * math.pi * self.f1 * tt + ctx.phase.view(N, 1)) \
            + 0.7 * torch.sin(2.0 * math.pi * self.f2 * tt + 1.7 * ctx.phase.view(N, 1))
        s = (s / 1.7).clamp(-1.0, 1.0)

        center = (-0.5 * math.pi) + (nozzle * 0.55)
        sweep = center + self.sweep_amp * s

        # deterministic spread inside the batch
        kf = ctx.k_idx.to(torch.float32)
        denom = max(1.0, float(ctx.max_k - 1))
        u = (2.0 * (kf / denom) - 1.0).clamp(-1.0, 1.0)
        aim_spread = aim + self.spread * u

        mix = max(0.0, min(1.0, self.aim_mix))
        mix_eff = (mix + 0.15 * gate).clamp(0.0, 1.0)

        ang = (1.0 - mix_eff) * sweep + mix_eff * aim_spread

        sm = max(0.0, min(1.0, self.speed_mod))
        sp_wave = torch.cos(3.0 * ang + 0.6 * ctx.phase.view(N, 1)).clamp(-1.0, 1.0)
        spd = (base_sp + amp_sp * (sm * sp_wave)).clamp(sp_min, sp_max)

        burst_mul = max(1.0, self.burst_mul)
        spd = (spd * (1.0 + (burst_mul - 1.0) * gate * 0.25)).clamp(sp_min, sp_max)

        new_phase = ctx.phase + self.phase_speed3 * (ctx.k_use > 0).to(torch.float32)
        return ang.to(torch.float32), spd.to(torch.float32), new_phase
