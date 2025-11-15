# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul


class CBOWInference:
    """학습된 CBOW 모델을 사용한 추론 클래스"""
    
    def __init__(self, W_in, W_out, word_to_id, id_to_word):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = len(word_to_id)
        
        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
    
    def predict(self, context_words):
        """
        맥락 단어들로부터 중심 단어를 예측
        
        Args:
            context_words: 맥락 단어 리스트 (예: ['you', 'goodbye'])
        
        Returns:
            predicted_word: 예측된 단어
            probabilities: 각 단어에 대한 확률 딕셔너리
        """
        # 단어를 ID로 변환
        context_ids = [self.word_to_id[w.lower()] for w in context_words]
        
        # 원-핫 인코딩
        c0 = np.zeros((1, self.vocab_size))
        c1 = np.zeros((1, self.vocab_size))
        c0[0, context_ids[0]] = 1
        c1[0, context_ids[1]] = 1
        
        # 순전파
        h0 = self.in_layer0.forward(c0)
        h1 = self.in_layer1.forward(c1)
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        
        # Softmax로 확률 계산
        exp_score = np.exp(score - np.max(score))
        probs = exp_score / np.sum(exp_score)
        
        # 가장 높은 확률의 단어 찾기
        predicted_id = np.argmax(probs[0])
        predicted_word = self.id_to_word[predicted_id]
        
        # 모든 단어에 대한 확률 딕셔너리 생성
        probabilities = {self.id_to_word[i]: float(probs[0, i]) 
                        for i in range(self.vocab_size)}
        
        return predicted_word, probabilities

def find_saved_models():
    """저장된 모델 목록 찾기"""
    import os
    import glob
    
    # training_text_*.txt 파일들을 찾아서 모델 목록 생성
    text_files = glob.glob('training_text_*.txt')
    
    models = []
    for text_file in text_files:
        # 파일명에서 file_id 추출
        file_id = text_file.replace('training_text_', '').replace('.txt', '')
        
        # 텍스트 내용 읽기
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            models.append((file_id, text_content))
        except:
            continue
    
    return models


def select_model():
    """사용자가 모델 선택"""
    models = find_saved_models()
    
    if not models:
        print("저장된 모델이 없습니다.")
        print("먼저 train_with_save.py를 실행하여 모델을 학습하세요.")
        return None
    
    print("저장된 모델 목록:\n")
    for idx, (file_id, text_content) in enumerate(models, 1):
        # 텍스트가 너무 길면 앞부분만 표시
        display_text = text_content if len(text_content) <= 50 else text_content[:47] + '...'
        print(f"{idx}. {file_id}")
        print(f"   → {display_text}\n")
    
    while True:
        try:
            choice = input(f"모델 선택 (1~{len(models)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(models):
                return models[choice_idx]
            else:
                print(f"1부터 {len(models)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return None


def main():
    """추론 예제"""
    print("=" * 60)
    print("CBOW 모델 추론")
    print("=" * 60)
    print()
    
    # 모델 선택
    selected = select_model()
    if selected is None:
        return
    
    file_id, training_text = selected
    print(f"\n선택된 모델: {file_id}")
    print(f"학습 텍스트: {training_text}\n")
    
    # 학습된 파라미터 로드
    try:
        W_in = np.load(f'cbow_params_W_in_{file_id}.npy')
        W_out = np.load(f'cbow_params_W_out_{file_id}.npy')
        word_to_id = np.load(f'word_to_id_{file_id}.npy', allow_pickle=True).item()
        id_to_word = np.load(f'id_to_word_{file_id}.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"오류: 파라미터 파일을 찾을 수 없습니다. ({e})")
        return
    
    # 추론 모델 생성
    model = CBOWInference(W_in, W_out, word_to_id, id_to_word)
    
    print(f"어휘 크기: {len(word_to_id)}")
    print(f"학습된 단어: {list(word_to_id.keys())}\n")
    
    # 예측 테스트
    print("=" * 60)
    print("예측 테스트")
    print("=" * 60)
    
    ############### 테스트 케이스 ###############
    # i say hello and you say goodbye.
    test_cases = [
        ['you', 'goodbye'],
        ['say', 'hello'],
        ['hello', 'you']
    ]
    
    ############### ######### ###############

    for context in test_cases:
        try:
            predicted, probs = model.predict(context)
            print(f"맥락: {context} → 예측: {predicted} (확률: {probs[predicted]:.4f})")
        except KeyError:
            print(f"맥락: {context} → 오류: 학습되지 않은 단어가 포함되어 있습니다.")


if __name__ == '__main__':
    main()