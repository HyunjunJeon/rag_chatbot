# VectorDB 저장 및 검색 가이드

## 🎯 전략 요약

### 청킹: 질문-답변 페어링
- 각 질문에 대해 **답변마다 개별 문서** 생성
- 1,273개 Q&A → 약 2,000~3,000개 문서 (답변 개수에 따라)

### 메타데이터: 16개 필드
```python
{
    # 필터링 핵심 (가장 중요)
    "course": "level2_cv",
    "generation": "4",
    "year": 2022,
    
    # 검색 최적화
    "has_bot_answer": false,
    "reaction_count": 15,
    "answer_index": 0
}
```

### 임베딩: 한국어 특화
- 모델: `jhgan/ko-sroberta-multitask` (768차원)
- 텍스트 구성: 과정 + 질문 + 답변 + 작성자

## 🚀 설치 및 설정

### 1. Qdrant 실행

```bash
# Docker로 Qdrant 실행
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 2. Python 패키지 설치

```bash
# 필수 패키지
pip install qdrant-client sentence-transformers tqdm
```

## 💾 데이터 저장

### 기본 사용법

```bash
cd document_processing

# 기본 실행
python ingest_to_vectordb.py

# 컬렉션 재생성 (기존 데이터 삭제)
python ingest_to_vectordb.py --recreate

# 테스트 검색 포함
python ingest_to_vectordb.py --test
```

### 옵션 설정

```bash
# 전체 옵션
python ingest_to_vectordb.py \
    --input-dir /path/to/slack_qa_merged \
    --qdrant-url http://localhost:6333 \
    --collection slack_qa \
    --model jhgan/ko-sroberta-multitask \
    --recreate \
    --test
```

### Python 코드로 실행

```python
from ingest_to_vectordb import QAVectorDBIngestion

# 1. Ingestion 객체 생성
ingestion = QAVectorDBIngestion(
    qdrant_url="http://localhost:6333",
    collection_name="slack_qa",
    embedding_model="jhgan/ko-sroberta-multitask"
)

# 2. 컬렉션 생성
ingestion.create_collection(recreate=False)

# 3. 데이터 저장
stats = ingestion.ingest_from_directory(
    "document_chunks/slack_qa_merged/"
)

# 4. 정보 확인
ingestion.get_collection_info()

# 5. 테스트 검색
ingestion.test_search(
    query="GPU 메모리 부족",
    course="level2_cv",
    limit=5
)
```

## 🔍 검색 방법

### 1. 기본 검색

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 클라이언트 초기화
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 검색
query = "GPU 메모리 부족 문제 해결 방법"
query_vector = model.encode(query).tolist()

results = client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    limit=10
)

for result in results:
    print(f"유사도: {result.score:.3f}")
    print(f"질문: {result.payload['question_text']}")
    print(f"답변: {result.payload['answer_text']}")
    print()
```

### 2. 과정별 필터링

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# level2_cv 과정만 검색
results = client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="course",
                match=MatchValue(value="level2_cv")
            )
        ]
    ),
    limit=10
)
```

### 3. 복합 필터링

```python
from qdrant_client.models import Range

# level2_cv + 2023년 이후 + 봇 답변 제외
results = client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            # 과정 필터
            FieldCondition(
                key="course",
                match=MatchValue(value="level2_cv")
            ),
            # 연도 필터
            FieldCondition(
                key="year",
                range=Range(gte=2023)
            ),
            # 봇 답변 제외
            FieldCondition(
                key="has_bot_answer",
                match=MatchValue(value=False)
            )
        ]
    ),
    limit=10,
    score_threshold=0.7  # 유사도 임계값
)
```

### 4. 인기도 기반 재정렬

```python
# 반응이 많은 답변 우선
results = client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="has_reactions",
                match=MatchValue(value=True)
            )
        ]
    ),
    limit=20  # 더 많이 가져옴
)

# reaction_count로 재정렬
sorted_results = sorted(
    results,
    key=lambda x: x.payload.get("reaction_count", 0),
    reverse=True
)[:10]

for result in sorted_results:
    print(f"반응 수: {result.payload['reaction_count']}")
    print(f"질문: {result.payload['question_text'][:100]}...")
    print()
```

## 🎨 실전 활용 예시

### 예시 1: 챗봇 응답 생성

```python
def get_relevant_qa(user_question: str, user_course: str) -> list[dict]:
    """
    사용자 질문에 관련된 Q&A를 검색합니다.
    
    Args:
        user_question: 사용자의 질문
        user_course: 사용자가 수강하는 과정
        
    Returns:
        관련 Q&A 리스트
    """
    # 벡터 임베딩
    query_vector = model.encode(user_question).tolist()
    
    # 검색 (같은 과정 + 최근 2년)
    results = client.search(
        collection_name="slack_qa",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="course", match=MatchValue(value=user_course)),
                FieldCondition(key="year", range=Range(gte=2023))
            ]
        ),
        limit=5,
        score_threshold=0.75
    )
    
    # 결과 포맷팅
    qa_list = []
    for result in results:
        qa_list.append({
            "question": result.payload["question_text"],
            "answer": result.payload["answer_text"],
            "similarity": result.score,
            "metadata": {
                "generation": result.payload["generation"],
                "date": result.payload["date"],
                "reactions": result.payload["reaction_count"]
            }
        })
    
    return qa_list
```

### 예시 2: 중복 질문 탐지

```python
def find_similar_questions(new_question: str, threshold: float = 0.85) -> list:
    """
    비슷한 질문이 이미 있는지 확인합니다.
    
    Args:
        new_question: 새로운 질문
        threshold: 유사도 임계값 (높을수록 엄격)
        
    Returns:
        유사한 질문 리스트
    """
    query_vector = model.encode(new_question).tolist()
    
    results = client.search(
        collection_name="slack_qa",
        query_vector=query_vector,
        limit=5,
        score_threshold=threshold
    )
    
    similar_questions = []
    for result in results:
        similar_questions.append({
            "question": result.payload["question_text"],
            "similarity": result.score,
            "link": f"thread_{result.payload['thread_id']}"
        })
    
    return similar_questions
```

### 예시 3: 인기 Q&A 추천

```python
def get_popular_qa(course: str, top_k: int = 10) -> list:
    """
    인기 있는 Q&A를 추천합니다.
    
    Args:
        course: 과정명
        top_k: 반환할 개수
        
    Returns:
        인기 Q&A 리스트
    """
    # 해당 과정의 반응이 많은 Q&A 검색
    # (쿼리 벡터 없이 필터만으로 검색)
    results = client.scroll(
        collection_name="slack_qa",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="course", match=MatchValue(value=course)),
                FieldCondition(key="has_reactions", match=MatchValue(value=True))
            ]
        ),
        limit=100  # 많이 가져와서 정렬
    )[0]
    
    # 반응 수로 정렬
    sorted_results = sorted(
        results,
        key=lambda x: x.payload.get("reaction_count", 0),
        reverse=True
    )[:top_k]
    
    popular_qa = []
    for result in sorted_results:
        popular_qa.append({
            "question": result.payload["question_text"],
            "answer": result.payload["answer_text"],
            "reactions": result.payload["reaction_count"],
            "date": result.payload["date"]
        })
    
    return popular_qa
```

## 📊 성능 최적화

### 1. 배치 크기 조정

```python
# 배치 크기를 늘려서 처리 속도 향상
ingestion = QAVectorDBIngestion(
    batch_size=200  # 기본값: 100
)
```

### 2. 캐싱 활용

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, course: str = None) -> list:
    """자주 검색되는 쿼리를 캐싱"""
    query_vector = model.encode(query).tolist()
    # ... 검색 로직
    return results
```

### 3. 리랭킹 (고급)

```python
from sentence_transformers import CrossEncoder

# 1차: 벡터 검색으로 후보 30개
candidates = client.search(
    collection_name="slack_qa",
    query_vector=query_vector,
    limit=30
)

# 2차: Cross-encoder로 정밀 재점수
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [
    [user_question, result.payload["answer_text"]]
    for result in candidates
]
scores = reranker.predict(pairs)

# 상위 10개 반환
top_results = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)[:10]
```

## 🔧 문제 해결

### Qdrant 연결 실패

```bash
# Qdrant가 실행 중인지 확인
docker ps | grep qdrant

# 실행 안 되어 있으면
docker run -p 6333:6333 qdrant/qdrant
```

### 메모리 부족

```python
# 배치 크기 줄이기
ingestion = QAVectorDBIngestion(
    batch_size=50  # 기본값: 100
)
```

### 임베딩 속도 느림

```python
# GPU 사용 (CUDA 설치 필요)
import torch

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
if torch.cuda.is_available():
    model = model.to('cuda')
```

## 📈 다음 단계

1. ✅ VectorDB 저장 완료
2. 🔄 검색 API 서버 구축
3. 🔄 RAG 시스템 통합
4. 🔄 웹 인터페이스 개발
5. 🔄 성능 모니터링

## 🎯 핵심 포인트

### ✅ 최적의 검색을 위한 설계
1. **질문-답변 페어링**: 각 답변을 개별 평가
2. **풍부한 메타데이터**: 16개 필터 필드
3. **한국어 특화 모델**: 높은 검색 정확도
4. **유연한 필터링**: 과정/기수/시기별 검색

### ✅ 실전 활용 가능
1. **챗봇 연동**: 관련 Q&A 자동 추천
2. **중복 탐지**: 비슷한 질문 자동 감지
3. **인기 콘텐츠**: 반응 많은 Q&A 큐레이션
4. **확장 가능**: 새 데이터 증분 추가

### ✅ 성능 최적화
1. **배치 처리**: 빠른 저장 속도
2. **캐싱**: 자주 검색되는 쿼리 최적화
3. **리랭킹**: 정밀한 결과 제공
4. **GPU 지원**: 대규모 처리 가능

