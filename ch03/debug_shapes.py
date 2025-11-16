# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'I say hello and you say goodbye.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
window_size = 1

print(f"텍스트: {text}")
print(f"어휘 크기: {vocab_size}")
print(f"corpus: {corpus}")
print(f"word_to_id: {word_to_id}")
print()

# 맥락과 타겟 생성
contexts, target = create_contexts_target(corpus, window_size)

print("=" * 60)
print("변환 전 (인덱스)")
print("=" * 60)
print(f"contexts shape: {contexts.shape}")
print(f"target shape: {target.shape}")
print(f"contexts:\n{contexts}")
print(f"target:\n{target}")
print()

# 원-핫 인코딩
target_onehot = convert_one_hot(target, vocab_size)
contexts_onehot = convert_one_hot(contexts, vocab_size)

print("=" * 60)
print("원-핫 인코딩 후")
print("=" * 60)
print(f"target_onehot shape: {target_onehot.shape}")
print(f"contexts_onehot shape: {contexts_onehot.shape}")
print(f"contexts_onehot[:, 0] shape: {contexts_onehot[:, 0].shape}")
print(f"contexts_onehot[:, 1] shape: {contexts_onehot[:, 1].shape}")
print()

print("=" * 60)
print("CBOW vs Skip-gram 입력 비교")
print("=" * 60)
print("CBOW:")
print(f"  입력(contexts): {contexts_onehot.shape}")
print(f"  정답(target): {target_onehot.shape}")
print()
print("Skip-gram:")
print(f"  입력(target): {target_onehot.shape}")
print(f"  정답(contexts): {contexts_onehot.shape}")