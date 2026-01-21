from __future__ import annotations

from torch_pattern_base import BulletPattern
from torch_pattern_p0 import Pattern0WobbleRing
from torch_pattern_p1 import Pattern1RotatingRing
from torch_pattern_p2 import Pattern2RosetteVolley
from torch_pattern_p3 import Pattern3ChaoticLissajousGate


def make_pattern(cfg) -> BulletPattern:
    pid = int(cfg.pattern_id)

    if pid == 0:
        return Pattern0WobbleRing(phase_speed0=cfg.phase_speed0)
    if pid == 1:
        return Pattern1RotatingRing(phase_speed1=cfg.phase_speed1)
    if pid == 2:
        return Pattern2RosetteVolley(
            phase_speed2=cfg.phase_speed2,
            petals2=cfg.petals2,
            twist2=cfg.twist2,
            speed_mod2=cfg.speed_mod2,
        )
    # pid == 3 default
    return Pattern3ChaoticLissajousGate(
        phase_speed3=cfg.phase_speed3,
        p3_aim_mix3=cfg.p3_aim_mix3,
        p3_sweep_amp3=cfg.p3_sweep_amp3,
        p3_f1_3=cfg.p3_f1_3,
        p3_f2_3=cfg.p3_f2_3,
        p3_gate_hz3=cfg.p3_gate_hz3,
        p3_gate_duty3=cfg.p3_gate_duty3,
        p3_burst_mul3=cfg.p3_burst_mul3,
        p3_speed_mod3=cfg.p3_speed_mod3,
        p3_spread3=cfg.p3_spread3,
    )
