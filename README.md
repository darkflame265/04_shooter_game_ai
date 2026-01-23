## 프로젝트 환경 설정

1. 가상환경 설치
python -m venv venv

2. 가상환경 진입
venv/scripts/activate

3. 가성환경 내부에서 pip 버전 최신으로 업그레이드.
python -m pip install --upgrade pip

4. 프로젝트에 필요한 라이브러리 설치
pip install -r requirements.txt

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
python train.py --episodes 500000

## no-render 모드로 ai의 학습 진행하기.
python train.py --episodes 500000 --no-render

학습 파일은 checkpoints폴더에 저장됨.


<img width="715" height="786" alt="스크린샷 2026-01-22 104659" src="https://github.com/user-attachments/assets/b3e5068e-b4e4-44f0-9945-30a142ff3919" />

<img width="720" height="782" alt="스크린샷 2026-01-22 104725" src="https://github.com/user-attachments/assets/fbce2a91-b29c-4e02-b369-0013a2e7da4d" />
