## train.py 사용 방법

### 1. 기본 사용 (CBOW 모델, 기본 하이퍼파라미터)
```bash
python train.py
```

### 2. Skip-gram 모델 학습
```bash
python train.py --model skipgram
```

### 3. 하이퍼파라미터 커스터마이징
```bash
# CBOW 모델, 윈도우 크기 3, 은닉층 150
python train.py --model cbow --window_size 3 --hidden_size 150

# Skip-gram 모델, 배치 크기 128, 에폭 20
python train.py --model skipgram --batch_size 128 --max_epoch 20
```

### 4. 모든 옵션 사용
```bash
python train.py --model skipgram --window_size 5 --hidden_size 100 --batch_size 100 --max_epoch 10
```

## 커맨드 라인 옵션

| 옵션 | 설명 | 기본값 | 선택 가능 값 |
|------|------|--------|--------------|
| `--model` | 학습할 모델 종류 | `cbow` | `cbow`, `skipgram` |
| `--window_size` | 윈도우 크기 | `5` | 양의 정수 |
| `--hidden_size` | 은닉층 크기 (임베딩 차원) | `100` | 양의 정수 |
| `--batch_size` | 배치 크기 | `100` | 양의 정수 |
| `--max_epoch` | 최대 에폭 수 | `10` | 양의 정수 |

## 도움말 보기
```bash
python train.py --help
```

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