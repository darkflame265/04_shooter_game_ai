## 설명문

**04_shooter_game_ai**는 *동방 프로젝트(Touhou)*의 탄막 패턴을 참고하여, **보스 몬스터(패턴)를 가정한 AI 회피 학습 시뮬레이터**입니다.  
단순히 사방에서 날아오는 랜덤 탄막을 피하는 것이 아니라, 보스가 특정 규칙/패턴으로 발사하는 탄막을 기반으로 **패턴 적응형 회피 전략**을 학습하는 데 초점을 둡니다.

이 프로젝트의 목표는 다음과 같습니다.

- 보스 패턴 기반 탄막 환경에서의 **강화학습 회피 정책(policy) 학습**
- 패턴/난이도/보상 설계를 통한 **학습 안정화 및 성장 과정 실험**
- 실제 게임 환경에 적용하기 전 단계로써 **시뮬레이터 구조 및 학습 파이프라인 검증**

---

## 프로젝트 환경 설정

1. 가상환경 설치
```powershell
python -m venv venv
```

2. 가상환경 진입
```powershell
venv/scripts/activate
```

3. 가성환경 내부에서 pip 버전 최신으로 업그레이드.
```powershell
python -m pip install --upgrade pip
```

4. 프로젝트에 필요한 라이브러리 설치
```powershell
pip install -r requirements.txt
```

## Requirements 내부 라이브러리.

```txt
filelock==3.20.3
fsspec==2026.1.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.1
pillow==12.0.0
setuptools==80.9.0
sympy==1.14.0
torch==2.9.1+cu130
torchaudio==2.9.1+cu130
torchvision==0.24.1+cu130
typing_extensions==4.15.0
```

## 실행 방법

### render 모드로 ai가 회피하는 모습 관찰하기.(학습X)
```powershell
python train.py --episodes 500000
```

## no-render 모드로 ai의 학습 진행하기.
```powershell
python train.py --episodes 500000 --no-render
```

학습 파일은 checkpoints폴더에 저장됨.

```
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

    bullet_speed_min: float = 0.2
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
    phase_speed2: float = 0.1
    phase_speed3: float = 0.45

    petals2: int = 8
    twist2: float = 0.9
    speed_mod2: float = 0.5

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
```
torch_config.py에서 pattern_id 값을 변경해, 보스의 탄막 패턴을 바꿀 수 있습니다.

<img width="715" height="786" alt="스크린샷 2026-01-22 104659" src="https://github.com/user-attachments/assets/76647b0e-d88b-400c-a3dc-5476902f91e7" />
<img width="720" height="782" alt="스크린샷 2026-01-22 104725" src="https://github.com/user-attachments/assets/8159bdff-7aec-49b0-b6a2-23da8a8aecca" />

