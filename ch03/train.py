# coding: utf-8
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.' # 샘플 텍스트 데이터 => 대체 가능
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
# 맥락(context)과 타겟(target) 생성 => example.py 참고
contexts, target = create_contexts_target(corpus, window_size)
# 원-핫 인코딩: 숫자를 벡터로 변환 => example.py 참고
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# 모델과 트레이너 생성
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 학습된 단어의 분산 표현 출력
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
