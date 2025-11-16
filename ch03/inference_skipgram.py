# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul


class SkipGramInference:
    """학습된 Skip-gram 모델을 사용한 추론 클래스"""
    
    def __init__(self, W_in, W_out, word_to_id, id_to_word):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = len(word_to_id)
        
        # 계층 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
    
    def predict(self, center_word):
        """
        중심 단어로부터 맥락 단어들을 예측
        
        Args:
            center_word: 중심 단어 (예: 'say')
        
        Returns:
            predictions: 예측된 맥락 단어 리스트 [왼쪽 단어, 오른쪽 단어]
            all_probabilities: 각 단어에 대한 확률 딕셔너리
        """
        # 단어를 ID로 변환
        center_id = self.word_to_id[center_word.lower()]
        
        # 원-핫 인코딩
        target = np.zeros((1, self.vocab_size))
        target[0, center_id] = 1
        
        # 순전파
        h = self.in_layer.forward(target)
        score = self.out_layer.forward(h)
        
        # Softmax로 확률 계산
        exp_score = np.exp(score - np.max(score))
        probs = exp_score / np.sum(exp_score)
        
        # 가장 높은 확률의 단어 2개 찾기 (왼쪽, 오른쪽 맥락)
        top2_indices = np.argsort(probs[0])[::-1][:2]
        predictions = [self.id_to_word[idx] for idx in top2_indices]
        
        # 모든 단어에 대한 확률 딕셔너리 생성
        all_probabilities = {self.id_to_word[i]: float(probs[0, i]) 
                            for i in range(self.vocab_size)}
        
        return predictions, all_probabilities


def find_saved_models():
    """저장된 Skip-gram 모델 목록 찾기"""
    import os
    import glob
    
    # training_text_skipgram_*.txt 파일들을 찾아서 모델 목록 생성
    text_files = glob.glob('training_text_skipgram_*.txt')
    
    models = []
    for text_file in text_files:
        # 파일명에서 file_id 추출
        file_id = text_file.replace('training_text_skipgram_', '').replace('.txt', '')
        
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
        print("저장된 Skip-gram 모델이 없습니다.")
        print("먼저 train_skipgram_with_save.py를 실행하여 모델을 학습하세요.")
        return None
    
    print("저장된 Skip-gram 모델 목록:\n")
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
    print("Skip-gram 모델 추론")
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
        W_in = np.load(f'skipgram_params_W_in_{file_id}.npy')
        W_out = np.load(f'skipgram_params_W_out_{file_id}.npy')
        word_to_id = np.load(f'word_to_id_{file_id}.npy', allow_pickle=True).item()
        id_to_word = np.load(f'id_to_word_{file_id}.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"오류: 파라미터 파일을 찾을 수 없습니다. ({e})")
        return
    
    # 추론 모델 생성
    model = SkipGramInference(W_in, W_out, word_to_id, id_to_word)
    
    print(f"어휘 크기: {len(word_to_id)}")
    print(f"학습된 단어: {list(word_to_id.keys())}\n")
    
    # 예측 테스트
    print("=" * 60)
    print("예측 테스트 (중심 단어 → 맥락 단어들 예측)")
    print("=" * 60)
    
    ############### 테스트 케이스 ###############
    # Skip-gram: 중심 단어를 입력으로 사용
    # i say hello and you say goodbye.
    test_cases = [
        'say',      # say → [i, hello] 또는 [you, goodbye]
        'hello',    # hello → [say, and]
        'you',      # you → [and, say]
        'and',      # and → [hello, you]
        'i',        # i → [say]
        'goodbye'   # goodbye → [say, .]
    ]
    
    ############### ######### ###############

    for center_word in test_cases:
        try:
            predictions, probs = model.predict(center_word)
            print(f"\n중심 단어: '{center_word}'")
            print(f"  예측된 맥락 단어: {predictions}")
            print(f"  확률: [{probs[predictions[0]]:.4f}, {probs[predictions[1]]:.4f}]")
            
            # 상위 5개 단어와 확률 출력
            sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  전체 확률 Top 5:")
            for word, prob in sorted_words:
                print(f"    {word}: {prob:.4f}")
                
        except KeyError:
            print(f"중심 단어: '{center_word}' → 오류: 학습되지 않은 단어입니다.")


if __name__ == '__main__':
    main()