# ch03 w2v 실습 흐름

## 1. VScode 환경 준비
### 저장소 클론
```bash
git clone https://github.com/WooYoungSeok/deep-learning-from-scratch-2.git
```

#### 프로젝트 디렉토리로 이동
```bash
cd deep-learning-from-scratch-2
```

#### VS Code로 열기 (or open folder)
```bash
code .
```

### 가상환경 및 패키지 준비

#### Python 가상환경 생성
```bash
python -m venv venv
```

#### 가상환경 활성화
##### Windows
```bash
venv\Scripts\activate
```

#### 파이썬 버전 확인
```bash
python --version
```

#### 필요 패키지 다운로드
```bash
pip install numpy matplotlib scipy
```

#### 환경 체크
```bash
python cbow_predict.py
```

## 2. cbow 학습
### train.py
`train.py` 파일을 확인하여 학습 과정 확인.

### simple_cbow.py
`simple_cbow` 파일을 확인하여 모델 구조 확인.

### 학습 시작
`train.py`를 실행 및 loss값 plot 확인.

### 예측값 확인
`cbow_predict.py`를 실행하고 terminal 창 확인.
#### 반복(iteration)과 에폭(epoch)의 의미

##### 반복(iteration)의 의미
```python
batch_size = 3
```

###### 전체 데이터 크기 확인
```python
text = 'You say goodbye and I say hello.'
# 전처리 후 단어: you, say, goodbye, and, i, say, hello
# contexts-target 쌍: 6개 (**example_context_target에 text 넣고 돌려보면 확인 가능**)
```

###### 1 에폭당 반복 횟수
```python
# 전체 데이터 수: contexts-target 쌍
반복 횟수 = ceil(전체 데이터 수 / batch_size)
          = ceil(6 / 3)
          = 2
```

따라서 **"반복 1 / 2"**는:
- 전체 데이터를 `batch_size=3`으로 나누면 **2번의 배치**가 필요

---

##### 에폭(epoch) vs 반복(iteration)
- **에폭(Epoch)**: 전체 데이터를 **한 번** 모두 학습
- **반복(Iteration/Step)**: **하나의 배치**를 학습

###### 예시:
```python
데이터 6개, batch_size=3

1 에폭 = 2 반복
- 반복 1: 데이터 [0, 1, 2] 처리
- 반복 2: 데이터 [3, 4, 5] 처리

1000 에폭 = 2000 반복 (총)
```

## 3. skip-gram