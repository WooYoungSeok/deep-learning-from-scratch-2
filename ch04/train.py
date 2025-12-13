# coding: utf-8
import sys
sys.path.insert(0, '..')  # 부모 디렉터리를 최우선으로 import
import numpy as np
from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요). => CUDA 버전에 맞는 CuPy wheel 설치(터미널에 nvidia-smi로 버전 확인 후 https://docs.cupy.dev/en/stable/install.html 참고)
# ===============================================
#  config.GPU = True
# ===============================================
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb
import argparse

# 커맨드 라인 인자 파싱
parser = argparse.ArgumentParser(description='Word2Vec 학습 (CBOW 또는 Skip-gram)')
parser.add_argument('model', type=str, nargs='?', default='cbow', choices=['cbow', 'skipgram'],
                    help='학습할 모델 선택: cbow 또는 skipgram (기본값: cbow)')
parser.add_argument('--window_size', type=int, default=5,
                    help='윈도우 크기 (기본값: 5)')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='은닉층 크기 (기본값: 100)')
parser.add_argument('--batch_size', type=int, default=100,
                    help='배치 크기 (기본값: 100)')
parser.add_argument('--max_epoch', type=int, default=10,
                    help='최대 에폭 수 (기본값: 10)')
args = parser.parse_args()

# 하이퍼파라미터 설정
window_size = args.window_size
hidden_size = args.hidden_size
batch_size = args.batch_size
max_epoch = args.max_epoch
model_type = args.model

print(f'=== 학습 설정 ===')
print(f'모델: {model_type.upper()}')
print(f'윈도우 크기: {window_size}')
print(f'은닉층 크기: {hidden_size}')
print(f'배치 크기: {batch_size}')
print(f'최대 에폭: {max_epoch}')
print('=' * 20)

# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델 등 생성
if model_type == 'cbow':
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    # 데이터셋과 일자에 따라 파일 이름 설정
    # ===============================================
    pkl_file = 'cbow_params.pkl' # 데이터셋과 일자에 따라 파일 이름 설정
    # ===============================================
else:  # skipgram
    model = SkipGram(vocab_size, hidden_size, window_size, corpus)
    # 데이터셋과 일자에 따라 파일 이름 설정
    # ===============================================
    pkl_file = 'skipgram_params.pkl' 
    # ===============================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs # 단어 표현 벡터(임베딩 값으로 사용하기도 함)
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
print(f'\n학습 완료! 파라미터를 {pkl_file}에 저장합니다.')
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
