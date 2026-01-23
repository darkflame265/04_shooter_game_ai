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

<img width="715" height="786" alt="스크린샷 2026-01-22 104659" src="https://github.com/user-attachments/assets/76647b0e-d88b-400c-a3dc-5476902f91e7" />
<img width="720" height="782" alt="스크린샷 2026-01-22 104725" src="https://github.com/user-attachments/assets/8159bdff-7aec-49b0-b6a2-23da8a8aecca" />

