# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in) # 첫 번째 맥락 단어에 대한 입력 계층
        self.in_layer1 = MatMul(W_in) # 두 번째 맥락 단어에 대한 입력 계층
        self.out_layer = MatMul(W_out) # 출력 계층
        self.loss_layer = SoftmaxWithLoss() # 소프트맥스와 손실 함수 계층

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        '''
        **과정:**
        1. 좌우 맥락 단어를 각각 은닉층으로 변환
        2. 두 벡터의 **평균**을 계산 (CBOW의 핵심)
        3. 평균 벡터로 타겟 단어 예측
        4. 손실(loss) 계산
        '''
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    # 역전파: 순전파의 역순으로 기울기 계산
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
