# Hybrid Search 사용 가이드

## 빠른 시작

### 1. 패키지 설치

```bash
pip install kiwipiepy rank-bm25 qdrant-client sentence-transformers
```

### 2. BM25 인덱스 생성

```bash
cd document_processing
python bm25_indexer.py --test
```

### 3. Qdrant에 벡터 저장

```bash
python ingest_to_vectordb.py --recreate
```

### 4. Hybrid Search 실행

```python
from hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

results = retriever.search(
    query="GPU 메모리 부족",
    course="level2_cv",
    alpha=0.7,  # 벡터 70%, BM25 30%
    limit=10
)

for result in results:
    print(f"질문: {result['question_text']}")
    print(f"답변: {result['answer_text']}")
    print(f"점수: {result['fusion_score']:.4f}")
    print()
```

## 문서화 전략

### Q&A 문서 구조

```python
# Qdrant에 저장되는 구조
{
    "id": "doc_id_123",
    "vector": [...],  # 768차원 Dense embedding
    
    "payload": {
        # 원본 텍스트 (BM25도 사용)
        "question_text": "GPU 메모리 부족 해결 방법은?",
        "answer_text": "배치 크기를 줄이거나...",
        
        # 메타데이터
        "course": "level2_cv",
        "generation": "4",
        "year": 2022,
        ...
    }
}

# BM25 인덱스 (별도 저장)
{
    "question": BM25Okapi([질문 토큰들]),
    "answer": BM25Okapi([답변 토큰들]),
    "combined": BM25Okapi([질문+답변 토큰들])
}
```

### 필드 분리의 장점

1. **질문 우선 검색**: 질문에서만 키워드 매칭
2. **답변 우선 검색**: 답변에서만 키워드 매칭
3. **가중치 조절**: 질문 70% + 답변 30% 등

## 검색 전략

### Alpha 값 조절

```python
# 의미 중심 검색 (추상적 개념)
results = retriever.search(
    query="모델 성능 향상 방법",
    alpha=0.9  # 벡터 90%, BM25 10%
)

# 균형 검색 (일반적)
results = retriever.search(
    query="GPU 메모리 부족",
    alpha=0.7  # 벡터 70%, BM25 30%
)

# 키워드 중심 검색 (정확한 용어)
results = retriever.search(
    query="torch.cuda.OutOfMemoryError",
    alpha=0.3  # 벡터 30%, BM25 70%
)
```

### BM25 필드 가중치

```python
# 질문 우선 (중복 질문 탐지)
results = retriever.search(
    query="GPU 메모리 부족",
    bm25_question_weight=0.8,
    bm25_answer_weight=0.2
)

# 답변 우선 (해결 방법 검색)
results = retriever.search(
    query="gradient checkpointing",
    bm25_question_weight=0.2,
    bm25_answer_weight=0.8
)
```

### 융합 방법 선택

```python
# RRF (Reciprocal Rank Fusion) - 추천
results = retriever.search(
    query="GPU 메모리 부족",
    fusion_method="rrf"
)

# Weighted Fusion (가중치 기반)
results = retriever.search(
    query="GPU 메모리 부족",
    fusion_method="weighted",
    alpha=0.7
)
```

## 실전 사용 예시

### 예시 1: 챗봇 질문 응답

```python
def answer_question(user_question: str, user_course: str):
    """사용자 질문에 답변"""
    retriever = HybridRetriever()
    
    # 같은 과정 + 최근 2년 데이터
    results = retriever.search(
        query=user_question,
        course=user_course,
        year_from=2023,
        alpha=0.7,
        limit=3
    )
    
    # 상위 3개 답변 반환
    for result in results:
        print(f"[유사도 {result['fusion_score']:.2f}]")
        print(f"Q: {result['question_text']}")
        print(f"A: {result['answer_text']}\n")
```

### 예시 2: 중복 질문 감지

```python
def find_duplicate_questions(new_question: str):
    """비슷한 질문이 있는지 확인"""
    retriever = HybridRetriever()
    
    # 질문에 높은 가중치
    results = retriever.search(
        query=new_question,
        alpha=0.8,  # 의미 유사도 중요
        bm25_question_weight=0.9,  # 질문 필드 우선
        bm25_answer_weight=0.1,
        limit=5
    )
    
    # 유사도 0.8 이상만
    similar = [r for r in results if r['fusion_score'] > 0.8]
    
    if similar:
        print("비슷한 질문이 이미 있습니다:")
        for result in similar:
            print(f"- {result['question_text']}")
    else:
        print("새로운 질문입니다.")
```

### 예시 3: 키워드 검색

```python
def search_by_keyword(keyword: str):
    """특정 키워드로 검색"""
    retriever = HybridRetriever()
    
    # BM25 비중 증가
    results = retriever.search(
        query=keyword,
        alpha=0.4,  # BM25 60%
        bm25_answer_weight=0.7,  # 답변에서 검색
        limit=10
    )
    
    return results
```

## 성능 비교

### 각 방법의 강점

```python
# 비교 테스트
retriever = HybridRetriever()

comparison = retriever.compare_search_methods(
    query="GPU 메모리 부족",
    course="level2_cv",
    limit=5
)

print("=== Vector Only ===")
for r in comparison["vector"]:
    print(f"- {r['payload']['question_text'][:80]}...")

print("\n=== BM25 Only ===")
for r in comparison["bm25"]:
    print(f"- {r['document']['question_text'][:80]}...")

print("\n=== Hybrid ===")
for r in comparison["hybrid"]:
    print(f"- {r['question_text'][:80]}...")
```

## 파일 구조

```
document_processing/
├── hybrid_search_strategy.md    # 전략 가이드
├── bm25_indexer.py              # BM25 인덱스 생성
├── hybrid_retriever.py          # Hybrid Search 구현
├── ingest_to_vectordb.py        # Qdrant 벡터 저장
└── HYBRID_SEARCH_USAGE.md       # 이 파일

document_chunks/
└── bm25_index.pkl               # BM25 인덱스 파일
```

## 실행 순서

### 1단계: 인덱스 생성

```bash
# BM25 인덱스 생성 (~1-2분)
python bm25_indexer.py \
    --input-dir document_chunks/slack_qa_merged \
    --output document_chunks/bm25_index.pkl \
    --test
```

### 2단계: Qdrant 저장

```bash
# Qdrant 실행
docker run -p 6333:6333 qdrant/qdrant

# 벡터 저장 (~5-10분)
python ingest_to_vectordb.py --recreate --test
```

### 3단계: Hybrid Search 테스트

```bash
python hybrid_retriever.py
```

## 문제 해결

### Kiwi 설치 오류

```bash
# Kiwi 설치
pip install kiwipiepy

# 초기화 (첫 실행 시 자동)
python -c "from kiwipiepy import Kiwi; Kiwi()"
```

### BM25 인덱스 메모리 부족

```python
# 인덱스를 필요할 때만 로드
indexer = BM25Indexer()
indexer.load_index("bm25_index.pkl")  # 필요시에만

# 사용 후 메모리 해제
del indexer
import gc
gc.collect()
```

### Qdrant 연결 실패

```bash
# Qdrant 실행 확인
docker ps | grep qdrant

# 실행 안되어 있으면
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

## 성능 최적화

### 1. 인덱스 사전 로드

```python
# 서버 시작 시 한 번만 로드
retriever = HybridRetriever()

# 이후 계속 재사용
def search_api(query: str):
    return retriever.search(query)
```

### 2. 결과 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, course: str = None):
    return retriever.search(query, course)
```

### 3. 비동기 처리

```python
import asyncio

async def async_search(query: str):
    # 벡터 + BM25 병렬 실행
    loop = asyncio.get_event_loop()
    
    vector_task = loop.run_in_executor(None, vector_search, query)
    bm25_task = loop.run_in_executor(None, bm25_search, query)
    
    vector_results, bm25_results = await asyncio.gather(
        vector_task, bm25_task
    )
    
    return merge_results(vector_results, bm25_results)
```

## 평가 메트릭

### 검색 품질 측정

```python
def evaluate_search_quality(test_cases):
    """검색 품질 평가"""
    metrics = {
        "precision@5": 0,
        "recall@5": 0,
        "mrr": 0
    }
    
    for query, relevant_docs in test_cases:
        results = retriever.search(query, limit=5)
        retrieved_ids = [r["doc_id"] for r in results]
        
        # Precision@5
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_docs))
        metrics["precision@5"] += relevant_retrieved / 5
        
        # Recall@5
        metrics["recall@5"] += relevant_retrieved / len(relevant_docs)
        
        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                metrics["mrr"] += 1 / rank
                break
    
    # 평균 계산
    n = len(test_cases)
    return {k: v/n for k, v in metrics.items()}
```

## 다음 단계

1. ✅ Hybrid Search 구현 완료
2. 🔄 API 서버 구축
3. 🔄 웹 인터페이스 연동
4. 🔄 성능 모니터링
5. 🔄 프로덕션 배포

## 요약

### ✅ 구현 완료

- **BM25 인덱싱**: 질문/답변 분리, 한국어 형태소 분석
- **Hybrid Search**: RRF + Weighted Fusion
- **유연한 검색**: Alpha, 필드 가중치 조절
- **필터링**: 과정, 기수, 시기별 검색

### 🎯 핵심 장점

1. **높은 정확도**: 의미 + 키워드 검색 결합
2. **유연성**: 상황별 가중치 조절
3. **강건성**: 한 방식 실패해도 다른 방식 커버
4. **한국어 최적화**: Kiwi 형태소 분석기

