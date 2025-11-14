# ch03 w2v 실습 흐름

## 저장소 클론
```bash
git clone https://github.com/WooYoungSeok/deep-learning-from-scratch-2.git
```

## 프로젝트 디렉토리로 이동
```bash
cd deep-learning-from-scratch-2
```

## VS Code로 열기 (or open folder)
```bash
code .
```

## 가상환경 및 패키지 준비

### Python 가상환경 생성
```bash
python -m venv venv
```

### 가상환경 활성화
#### Windows
```bash
venv\Scripts\activate
```

#### Git Bash on Windows
```bash
source venv/Scripts/activate
```

### 파이썬 버전 확인
```bash
python --version
```

### 필요 패키지 다운로드
```bash
pip install numpy matplotlib scipy
```

### 환경 체크
```bash
python cbow_predict.py  # 성능이 엉망
```

## cbow 학습
`train.py` 파일을 확인하여 학습 진행.