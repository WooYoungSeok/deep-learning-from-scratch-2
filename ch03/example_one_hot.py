import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target

def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환

    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1 # 인덱스에 해당하는 값만 1로 변경

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1 # 인덱스에 해당하는 값만 1로 변경

    return one_hot


# 예시 사용법
text = 'what are you doing?' # 샘플 텍스트 데이터 => 대체 가능
corpus, word_to_id, id_to_word = preprocess(text)

# contexts, target: 반복해서 등장하는 단어는 앞서서 등장한 index로 저장
contexts, target = create_contexts_target(corpus, window_size=1)
print('맥락 데이터:\n', contexts)
print('타겟 데이터:\n', target)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

print('원핫인코딩 맥락 데이터:\n', contexts)
print('원핫인코딩 타겟 데이터:\n', target)