# coding: utf-8
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np

########## 한글 폰트 설정 ##########
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
########## ########## ##########

from common.trainer import Trainer
from common.optimizer import Adam
from simple_skip_gram import SimpleSkipGram
from common.util import preprocess, create_contexts_target, convert_one_hot

######## 하이퍼파라미터 설정 ##########
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000
######### ############# ###########

######### 샘플 텍스트 데이터 => 대체 가능 #########
text = 'I say hello and you say goodbye.' 
######### ######################## #########

# 텍스트 기반 파일명 생성 (첫 3단어 사용)
def create_filename_from_text(text, max_words=3):
    """텍스트에서 파일명 생성"""
    import re
    # 알파벳과 공백만 남기고 소문자로 변환
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = cleaned.split()[:max_words]
    return '_'.join(words) if words else 'untitled'

file_id = create_filename_from_text(text)
print(f'학습 텍스트: {text}')
print(f'파일 식별자: {file_id}\n')

corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
# 맥락(context)과 타겟(target) 생성
contexts, target = create_contexts_target(corpus, window_size)

# Skip-gram: target을 입력으로, contexts를 출력(정답)으로 사용
# 원-핫 인코딩: 숫자를 벡터로 변환
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# 모델과 트레이너 생성
model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# Skip-gram 학습: target을 입력, contexts를 정답으로
# CBOW와 달리 forward(contexts, target)가 아니라 forward(target, contexts) 순서
print('Skip-gram 모델 학습 시작...')
print(f'어휘 크기: {vocab_size}')
print(f'은닉층 크기: {hidden_size}')
print(f'윈도우 크기: {window_size}')
print(f'배치 크기: {batch_size}')
print(f'에포크: {max_epoch}\n')

# 학습 시작 - 정의해둔 Skip-gram은 target, context 순으로 입력받음
trainer.fit(target, contexts, max_epoch, batch_size)
trainer.plot()

# 학습된 단어의 분산 표현 출력
word_vecs = model.word_vecs
print('\n학습된 단어 벡터:')
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

# 학습된 파라미터 저장
print('\n학습된 파라미터를 저장합니다...')
np.save(f'skipgram_params_W_in_{file_id}.npy', model.word_vecs)
np.save(f'skipgram_params_W_out_{file_id}.npy', model.out_layer.params[0])

# 단어 사전 저장
np.save(f'word_to_id_{file_id}.npy', word_to_id)
np.save(f'id_to_word_{file_id}.npy', id_to_word)

# 원본 텍스트도 함께 저장
with open(f'training_text_skipgram_{file_id}.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print('저장 완료!')
print(f'- skipgram_params_W_in_{file_id}.npy')
print(f'- skipgram_params_W_out_{file_id}.npy')
print(f'- word_to_id_{file_id}.npy')
print(f'- id_to_word_{file_id}.npy')
print(f'- training_text_skipgram_{file_id}.txt')

# .gitignore에 패턴 추가
def update_gitignore():
    """부모 디렉토리의 .gitignore에 Skip-gram 파일 패턴 추가"""
    gitignore_path = '../.gitignore'
    
    patterns = [
        '# Skip-gram 학습 파일',
        'ch03/skipgram_params_W_in_*.npy',
        'ch03/skipgram_params_W_out_*.npy',
        'ch03/word_to_id_*.npy',
        'ch03/id_to_word_*.npy',
        'ch03/training_text_skipgram_*.txt'
    ]
    
    try:
        # 기존 .gitignore 읽기
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        except FileNotFoundError:
            existing_content = ''
        
        # 이미 추가되어 있는지 확인
        if 'Skip-gram 학습 파일' in existing_content:
            return
        
        # .gitignore에 패턴 추가
        with open(gitignore_path, 'a', encoding='utf-8') as f:
            if existing_content and not existing_content.endswith('\n'):
                f.write('\n')
            f.write('\n'.join(patterns) + '\n')
        
        print('\n.gitignore 업데이트 완료!')
    except Exception as e:
        print(f'\n.gitignore 업데이트 실패: {e}')

update_gitignore()