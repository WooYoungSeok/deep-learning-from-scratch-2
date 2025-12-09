# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import glob


def cosine_similarity(vec1, vec2):
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        vec1, vec2: numpy 배열
    
    Returns:
        float: -1 ~ 1 사이의 유사도 값 (1에 가까울수록 유사)
    """
    # 벡터의 내적
    dot_product = np.dot(vec1, vec2)
    
    # 각 벡터의 노름(크기)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 0으로 나누는 것 방지
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 코사인 유사도 = 내적 / (노름1 * 노름2)
    return dot_product / (norm1 * norm2)


def most_similar(word, word_to_id, id_to_word, word_vecs, top_n=5):
    """
    주어진 단어와 가장 유사한 단어들을 찾기
    
    Args:
        word: 기준 단어
        word_to_id: 단어 → ID 딕셔너리
        id_to_word: ID → 단어 딕셔너리
        word_vecs: 단어 벡터 행렬 (vocab_size, hidden_size)
        top_n: 반환할 유사 단어 개수
    
    Returns:
        list: [(단어, 유사도), ...] 형태의 리스트
    """
    # 단어가 어휘에 없으면 오류
    if word.lower() not in word_to_id:
        return None
    
    # 기준 단어의 벡터
    word_id = word_to_id[word.lower()]
    word_vec = word_vecs[word_id]
    
    # 모든 단어와의 유사도 계산
    similarities = []
    for i in range(len(word_vecs)):
        if i == word_id:  # 자기 자신은 제외
            continue
        
        other_vec = word_vecs[i]
        sim = cosine_similarity(word_vec, other_vec)
        similarities.append((id_to_word[i], sim))
    
    # 유사도 기준 내림차순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


def analogy(word_a, word_b, word_c, word_to_id, id_to_word, word_vecs, top_n=5):
    """
    단어 유추 (analogy): A:B = C:?
    
    예시:
    - big:biggest = small:smallest
    - man:king = woman:queen (king에서 man을 빼면 왕위, 거기에 woman을 더하면 queen)
    - Seoul:Korea = Paris:France
    
    벡터 연산: vec(B) - vec(A) + vec(C) ≈ vec(?)
    해석: "A에서 B로의 관계"를 C에 적용
    
    Args:
        word_a: 관계의 시작 단어 (예: 'big')
        word_b: 관계의 끝 단어 (예: 'biggest')
        word_c: 적용할 단어 (예: 'small')
        word_to_id, id_to_word, word_vecs: 모델 파라미터
        top_n: 반환할 후보 개수
    
    Returns:
        list: [(단어, 유사도), ...] 형태의 리스트
        예: [('smallest', 0.95), ...]
    """
    # 단어들이 모두 어휘에 있는지 확인
    for w in [word_a, word_b, word_c]:
        if w.lower() not in word_to_id:
            print(f"오류: '{w}'는 어휘에 없습니다.")
            return None
    
    # 각 단어의 벡터
    vec_a = word_vecs[word_to_id[word_a.lower()]]
    vec_b = word_vecs[word_to_id[word_b.lower()]]
    vec_c = word_vecs[word_to_id[word_c.lower()]]
    
    # 유추 벡터: vec(B) - vec(A) + vec(C)
    target_vec = vec_b - vec_a + vec_c
    
    # 모든 단어와의 유사도 계산
    similarities = []
    exclude_ids = [word_to_id[w.lower()] for w in [word_a, word_b, word_c]]
    
    for i in range(len(word_vecs)):
        if i in exclude_ids:  # 입력 단어들은 제외
            continue
        
        sim = cosine_similarity(target_vec, word_vecs[i])
        similarities.append((id_to_word[i], sim))
    
    # 유사도 기준 내림차순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


def find_saved_models():
    """저장된 CBOW 모델 목록 찾기"""
    text_files = glob.glob('training_text_*.txt')
    
    # Skip-gram 모델은 제외
    text_files = [f for f in text_files if 'skipgram' not in f]
    
    models = []
    for text_file in text_files:
        file_id = text_file.replace('training_text_', '').replace('.txt', '')
        
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
        print("저장된 CBOW 모델이 없습니다.")
        print("먼저 train_with_save.py를 실행하여 모델을 학습하세요.")
        return None
    
    print("저장된 CBOW 모델 목록:\n")
    for idx, (file_id, text_content) in enumerate(models, 1):
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
    """단어 유사도 계산 예제"""
    print("=" * 60)
    print("CBOW 단어 벡터 유사도 분석")
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
        word_vecs = np.load(f'cbow_params_W_in_{file_id}.npy')
        word_to_id = np.load(f'word_to_id_{file_id}.npy', allow_pickle=True).item()
        id_to_word = np.load(f'id_to_word_{file_id}.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"오류: 파라미터 파일을 찾을 수 없습니다. ({e})")
        return
    
    vocab_size = len(word_to_id)
    hidden_size = word_vecs.shape[1]
    
    print(f"어휘 크기: {vocab_size}")
    print(f"벡터 차원: {hidden_size}")
    print(f"학습된 단어: {list(word_to_id.keys())}\n")
    
    # 단어 벡터 출력
    print("=" * 60)
    print("학습된 단어 벡터")
    print("=" * 60)
    for word_id, word in id_to_word.items():
        vec = word_vecs[word_id]
        print(f"{word:10s}: {vec}")
    print()
    
    # 유사도 계산 테스트
    print("=" * 60)
    print("단어 간 유사도 계산")
    print("=" * 60)
    
    # 모든 단어 쌍의 유사도 계산
    words = list(word_to_id.keys())
    print("\n코사인 유사도 행렬:\n")
    
    # 헤더 출력
    print(f"{'':10s}", end='')
    for word in words:
        print(f"{word:10s}", end='')
    print()
    
    # 유사도 행렬 출력
    for word1 in words:
        print(f"{word1:10s}", end='')
        vec1 = word_vecs[word_to_id[word1]]
        for word2 in words:
            vec2 = word_vecs[word_to_id[word2]]
            sim = cosine_similarity(vec1, vec2)
            print(f"{sim:10.4f}", end='')
        print()
    print()
    
    # 특정 단어와 가장 유사한 단어 찾기
    print("=" * 60)
    print("가장 유사한 단어 찾기")
    print("=" * 60)
    
    test_words = ['i', 'say', 'hello']  # 테스트할 단어들
    
    for test_word in test_words:
        if test_word not in word_to_id:
            continue
        
        similar_words = most_similar(test_word, word_to_id, id_to_word, word_vecs, top_n=3)
        print(f"\n'{test_word}'와 가장 유사한 단어:")
        if similar_words:
            for word, sim in similar_words:
                print(f"  {word}: {sim:.4f}")
        else:
            print("  (유사한 단어를 찾을 수 없습니다)")
    
    # 단어 유추 (analogy) - 어휘가 충분히 클 때만 의미있음
    if vocab_size >= 10:
        print("\n" + "=" * 60)
        print("단어 유추 (Word Analogy)")
        print("=" * 60)
        print("\n예: 'i':'say' = 'you':?")
        print("해석: 'i에서 say로의 관계'를 'you'에 적용")
        
        result = analogy('i', 'say', 'you', word_to_id, id_to_word, word_vecs, top_n=3)
        if result:
            print("예측 결과:")
            for word, sim in result:
                print(f"  {word}: {sim:.4f}")


if __name__ == '__main__':
    main()