# ch04 w2v 개선 흐름
## 0. 기존 github와 비교
https://github.com/WegraLee/deep-learning-from-scratch-2

## 1. VScode 환경 준비(필요시, README_ch03.md 참고)
### 가상환경 및 패키지 준비(필요시, README_ch03.md 참고)

#### 이번주차 디렉토리로 이동
```bash
cd ch04
```

## train.py 사용 방법

### 1. 기본 사용 (CBOW 모델, 기본 하이퍼파라미터)
```bash
python train.py # cbow
```

### 2. Skip-gram 모델 학습
```bash
python train.py skipgram
```
**⏱️ 주의:** 기본 하이퍼파라미터로 학습할 경우 1에포크가 약 **30분** 소요됩니다. 즉, 총 5시간 정도 걸립니다..

### 3. 하이퍼파라미터 커스터마이징
```bash
python train.py skipgram --window_size 5 --hidden_size 100 --batch_size 100 --max_epoch 10
```

## 커맨드 라인 옵션

| 옵션 | 설명 | 기본값 | 선택 가능 값 |
|------|------|--------|--------------|
| `model` | 학습할 모델 종류 | `cbow` | `cbow`, `skipgram` |
| `--window_size` | 윈도우 크기 | `5` | 양의 정수 |
| `--hidden_size` | 은닉층 크기 (임베딩 차원) | `100` | 양의 정수 |
| `--batch_size` | 배치 크기 | `100` | 양의 정수 |
| `--max_epoch` | 최대 에폭 수 | `10` | 양의 정수 |

## 출력 파일
학습이 완료되면 다음 파일이 생성됩니다:
- CBOW 모델: `cbow_params.pkl`
- Skip-gram 모델: `skipgram_params.pkl`

이 파일들은 학습된 단어 벡터와 단어-ID 매핑 정보를 포함합니다.

## 학습된 모델 평가
학습이 완료된 후 `eval.py`를 실행하여 모델을 평가할 수 있습니다:

```python
# eval.py에서 사용할 파일 선택
pkl_file = 'cbow_params.pkl'     # CBOW 모델 평가
# pkl_file = 'skipgram_params.pkl'  # Skip-gram 모델 평가
```

```bash
python eval.py
```