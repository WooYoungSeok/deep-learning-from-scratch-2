# coding: utf-8
import sys
sys.path.insert(0, '..')  # 부모 디렉터리를 최우선으로 import
from common.util import most_similar, analogy
import pickle

# 파라미터 파일명을 직접 넣어서 로드
# 커맨드라인: 학습 파일처럼 하이퍼파라미터가 많으면 용이
# 코드 파일내 수정: eval 파일처럼 파라미터 파일 하나 로드하는 경우 용이
# pkl_file = 'cbow_params.pkl'
pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# 가장 비슷한(most similar) 단어 뽑기
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 유추(analogy) 작업
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)
