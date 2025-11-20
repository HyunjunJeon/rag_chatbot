# Hybrid Search 전략: 벡터 검색 + BM25

## 개요

의미 기반 검색(Vector)과 키워드 기반 검색(BM25)을 결합한 Hybrid Search 전략입니다.

## 왜 Hybrid Search인가?

### 벡터 검색의 장점과 한계

**장점:**
- ✅ 의미적 유사도 파악 ("GPU 부족" ≈ "메모리 에러")
- ✅ 동의어 처리 가능
- ✅ 문맥 이해

**한계:**
- ❌ 정확한 키워드 매칭 약함 (예: 특정 함수명, 에러 코드)
- ❌ 희귀한 전문 용어 검색 어려움

### BM25 검색의 장점과 한계

**장점:**
- ✅ 정확한 키워드 매칭 (예: "torch.cuda.OutOfMemoryError")
- ✅ 전문 용어 검색 우수
- ✅ 빠른 속도

**한계:**
- ❌ 동의어 처리 안 됨
- ❌ 의미 이해 불가

### Hybrid = 둘의 장점 결합

```
Vector Search: "GPU 메모리 부족" → "CUDA out of memory" 찾기 ✅
BM25 Search:   "OutOfMemoryError" → 정확한 에러명 찾기 ✅
```

## 문서 구조 설계

### 필드 분리 전략

각 문서에 질문과 답변을 **분리된 필드**로 저장합니다.

```python
{
    # === Qdrant Vector (의미 검색용) ===
    "vector": [0.1, 0.2, ...],  # 768차원 임베딩
    
    # === Payload (메타데이터 + 텍스트) ===
    "payload": {
        # 원본 텍스트 (BM25 검색 대상)
        "question_text": "GPU 메모리 부족 문제 해결 방법은?",
        "answer_text": "배치 크기를 줄이거나 gradient checkpointing을 사용하세요.",
        
        # 토큰화된 텍스트 (BM25 인덱싱용)
        "question_tokens": ["GPU", "메모리", "부족", "문제", "해결", "방법"],
        "answer_tokens": ["배치", "크기", "줄이다", "gradient", "checkpointing", "사용"],
        
        # 메타데이터 (필터링용)
        "course": "level2_cv",
        "generation": "4",
        ...
    }
}
```

### 토큰화 전략

**한국어 형태소 분석기 사용:**
- **Kiwi** (추천): 빠르고 정확, 설치 간단
- Mecab: 전통적, 정확도 높음
- KoNLPy: 다양한 분석기 지원

**토큰화 예시:**
```python
from kiwipiepy import Kiwi

kiwi = Kiwi()

text = "GPU 메모리 부족 문제를 해결하는 방법은?"
tokens = [token.form for token in kiwi.tokenize(text)]
# ["GPU", "메모리", "부족", "문제", "를", "해결", "하다", "방법", "은"]
```

## BM25 인덱스 구조

### 별도 인덱스 관리

Qdrant와 별도로 BM25 인덱스를 유지합니다.

```python
from rank_bm25 import BM25Okapi

# 인덱스 구조
bm25_indices = {
    "question": BM25Okapi([
        doc1_question_tokens,
        doc2_question_tokens,
        ...
    ]),
    "answer": BM25Okapi([
        doc1_answer_tokens,
        doc2_answer_tokens,
        ...
    ]),
    "combined": BM25Okapi([
        doc1_question_tokens + doc1_answer_tokens,
        doc2_question_tokens + doc2_answer_tokens,
        ...
    ])
}

# 문서 ID 매핑 (검색 결과 연결용)
doc_id_mapping = [
    "doc_id_1",
    "doc_id_2",
    ...
]
```

### 인덱스 저장 및 로드

```python
import pickle

# 저장
with open("bm25_index.pkl", "wb") as f:
    pickle.dump({
        "indices": bm25_indices,
        "doc_ids": doc_id_mapping
    }, f)

# 로드
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
    bm25_indices = data["indices"]
    doc_id_mapping = data["doc_ids"]
```

## Hybrid Search 프로세스

### 1. 벡터 검색 (Qdrant)

```python
# 쿼리 임베딩
query_vector = embedding_model.encode(query)

# Qdrant 검색
vector_results = qdrant_client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    limit=50,  # 후보 많이 가져오기
    query_filter=course_filter  # 필터 적용
)
```

### 2. BM25 검색

```python
# 쿼리 토큰화
query_tokens = kiwi.tokenize(query)

# BM25 검색 (질문 + 답변 가중치)
question_scores = bm25_indices["question"].get_scores(query_tokens)
answer_scores = bm25_indices["answer"].get_scores(query_tokens)

# 가중치 조합 (질문에 더 높은 가중치)
bm25_scores = 0.7 * question_scores + 0.3 * answer_scores

# 상위 50개 선택
top_indices = np.argsort(bm25_scores)[::-1][:50]
bm25_results = [doc_id_mapping[i] for i in top_indices]
```

### 3. 결과 병합: Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    k: int = 60
) -> list:
    """
    RRF로 두 검색 결과를 병합합니다.
    
    RRF 공식: score(d) = Σ 1 / (k + rank(d))
    """
    scores = {}
    
    # 벡터 검색 결과
    for rank, doc in enumerate(vector_results, 1):
        doc_id = doc.id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    # BM25 검색 결과
    for rank, doc_id in enumerate(bm25_results, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    # 점수 순 정렬
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_docs
```

### 4. 가중치 조절 (Alpha Blending)

```python
def weighted_hybrid_search(
    vector_results: list,
    bm25_results: list,
    alpha: float = 0.7  # 벡터 가중치
) -> list:
    """
    alpha: 벡터 검색 가중치 (0~1)
    1-alpha: BM25 가중치
    
    alpha=0.7: 벡터 70%, BM25 30%
    alpha=0.5: 벡터 50%, BM25 50%
    """
    scores = {}
    
    # 정규화된 점수 계산
    for rank, doc in enumerate(vector_results, 1):
        normalized_score = 1 / rank
        scores[doc.id] = alpha * normalized_score
    
    for rank, doc_id in enumerate(bm25_results, 1):
        normalized_score = 1 / rank
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * normalized_score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## 필드별 BM25 검색 전략

### 전략 1: 질문 우선 검색

사용자의 질문과 비슷한 질문을 찾는 것이 우선입니다.

```python
# 질문 필드에 높은 가중치
bm25_score = 0.8 * question_bm25 + 0.2 * answer_bm25
```

**적합한 경우:**
- 중복 질문 탐지
- "비슷한 질문 있나요?" 기능

### 전략 2: 답변 우선 검색

답변 내용에서 키워드를 찾는 것이 중요합니다.

```python
# 답변 필드에 높은 가중치
bm25_score = 0.2 * question_bm25 + 0.8 * answer_bm25
```

**적합한 경우:**
- 특정 해결 방법 검색 (예: "gradient checkpointing")
- 코드 스니펫 검색

### 전략 3: 균형 검색 (추천)

질문과 답변 모두 고려합니다.

```python
# 균형 가중치
bm25_score = 0.6 * question_bm25 + 0.4 * answer_bm25
```

**적합한 경우:**
- 일반적인 Q&A 검색
- 포괄적인 정보 검색

## 성능 최적화

### 1. 인덱스 사전 로드

```python
class HybridRetriever:
    def __init__(self):
        # 시작 시 인덱스 로드 (느림)
        self.bm25_indices = self._load_bm25_index()
        self.qdrant_client = QdrantClient()
        
    def search(self, query):
        # 검색은 빠름
        pass
```

### 2. 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def hybrid_search_cached(query: str, course: str):
    return hybrid_search(query, course)
```

### 3. 비동기 검색

```python
import asyncio

async def async_hybrid_search(query):
    # 벡터 검색과 BM25 검색을 병렬로
    vector_task = asyncio.create_task(vector_search(query))
    bm25_task = asyncio.create_task(bm25_search(query))
    
    vector_results, bm25_results = await asyncio.gather(
        vector_task,
        bm25_task
    )
    
    return merge_results(vector_results, bm25_results)
```

## 검색 품질 평가

### 메트릭

1. **Precision@K**: 상위 K개 중 관련 문서 비율
2. **Recall@K**: 전체 관련 문서 중 상위 K개에 포함된 비율
3. **MRR (Mean Reciprocal Rank)**: 첫 관련 문서의 순위

### A/B 테스트

```python
# 벡터만
vector_only = search_vector(query)

# BM25만
bm25_only = search_bm25(query)

# 하이브리드
hybrid = hybrid_search(query, alpha=0.7)

# 평가
evaluate_results(vector_only, bm25_only, hybrid, ground_truth)
```

## 실전 사용 예시

### 예시 1: 일반 검색

```python
retriever = HybridRetriever()

results = retriever.search(
    query="GPU 메모리 부족 해결",
    course="level2_cv",
    alpha=0.7,  # 벡터 70%, BM25 30%
    limit=10
)
```

### 예시 2: 키워드 중심 검색

```python
# 특정 에러 코드 검색
results = retriever.search(
    query="torch.cuda.OutOfMemoryError",
    alpha=0.3,  # BM25 비중 증가
    limit=10
)
```

### 예시 3: 의미 중심 검색

```python
# 추상적인 개념 검색
results = retriever.search(
    query="모델 성능 향상 방법",
    alpha=0.9,  # 벡터 비중 증가
    limit=10
)
```

## 장단점 분석

### Hybrid Search의 장점

✅ **포괄적 검색**: 의미와 키워드 모두 커버
✅ **높은 정확도**: 두 방식의 약점 보완
✅ **유연성**: 가중치 조절로 상황별 최적화
✅ **강건성**: 한 방식이 실패해도 다른 방식이 커버

### 고려사항

⚠️ **인덱스 관리**: 두 개의 인덱스 유지 필요
⚠️ **메모리**: BM25 인덱스가 메모리에 상주
⚠️ **복잡도**: 구현 및 디버깅 복잡
⚠️ **파라미터 튜닝**: alpha 값 최적화 필요

## 다음 단계

1. ✅ 전략 설계 완료
2. 🔄 BM25 인덱스 생성 스크립트 작성
3. 🔄 Hybrid Retriever 클래스 구현
4. 🔄 성능 평가 및 튜닝
5. 🔄 프로덕션 배포

## 참고 자료

- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/tutorials/hybrid-search/)
- [rank_bm25 라이브러리](https://github.com/dorianbrown/rank_bm25)

