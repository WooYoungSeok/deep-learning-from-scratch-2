import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess

def create_contexts_target(corpus, window_size=1):
    '''맥락과 타깃 생성
    
    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    '''
    target = corpus[window_size:-window_size] # corpus의 앞 단어, 뒷 단어
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1): # 앞 단어부터 뒷 단어까지 반복
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

# 예시 사용법
text = 'what are you doing?' # 샘플 텍스트 데이터 => 대체 가능
corpus, word_to_id, id_to_word = preprocess(text)

# contexts, target: 반복해서 등장하는 단어는 앞서서 등장한 index로 저장
contexts, target = create_contexts_target(corpus, window_size=1)
print('맥락 데이터:\n', contexts)
print('타겟 데이터:\n', target)
