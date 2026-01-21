from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    # world
    world_size: float = 1.0
    dt: float = 1.0 / 60.0
    max_steps: int = 60 * 60

    # player
    player_speed: float = 0.3
    player_radius: float = 0.007

    # bullets (capacity)
    max_bullets: int = 2048
    bullet_radius: float = 0.01

    bullet_speed_min: float = 0.1
    bullet_speed_max: float = 0.2

    aim_noise: float = 0.0

    spawn_rate: float = 300.0
    obs_k: int = 32

    # reward shaping
    alive_reward: float = 0.01
    hit_penalty: float = 3.0
    move_penalty: float = 0.0005
    wall_penalty: float = 0.02

    near_miss_enabled: bool = True
    near_miss_margin: float = 0.03
    near_miss_coef: float = 0.015

    clear_bonus: float = 0.25

    # Curriculum (spawn_rate only)
    curriculum: bool = True
    curriculum_episodes: int = 800
    spawn_rate_start: float = 3.0
    spawn_rate_step: float = 0.5

    bullet_speed_start_factor: float = 1.0
    aim_noise_start_factor: float = 1.0

    # Boss params
    boss_y: float = 0.92
    boss_x_amp: float = 0.38
    boss_move_hz: float = 0.05

    # pattern
    # 0: wobble ring
    # 1: rotating ring
    # 2: rosette volley (8-way unit)
    # 3: chaotic lissajous gate
    pattern_id: int = 3

    phase_speed0: float = 0.55
    phase_speed1: float = 0.12
    phase_speed2: float = 0.99
    phase_speed3: float = 0.45

    petals2: int = 2
    twist2: float = 0.9
    speed_mod2: float = 0.0

    p3_aim_mix3: float = 0.1
    p3_sweep_amp3: float = 1.40
    p3_f1_3: float = 0.73
    p3_f2_3: float = 1.19
    p3_gate_hz3: float = 0.45
    p3_gate_duty3: float = 0.10
    p3_burst_mul3: float = 2.2
    p3_speed_mod3: float = 0.45
    p3_spread3: float = 0.18

    # Rendering (tkinter)
    render_size: int = 720
    render_fps: float = 60
    render_draw_trails: bool = False

    seed: int = 0
