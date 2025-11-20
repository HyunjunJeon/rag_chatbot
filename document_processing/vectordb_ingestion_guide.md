# Slack Q&A VectorDB 저장 전략 가이드

## 개요

Qdrant VectorDB에 Slack Q&A 데이터를 최적으로 저장하기 위한 전략 가이드입니다.

## 핵심 전략

### 1. 청킹 전략: **질문-답변 페어링** (추천)

각 질문에 대해 답변마다 개별 문서를 생성합니다.

**장점:**
- ✅ 질문 컨텍스트와 답변 내용을 함께 임베딩
- ✅ 각 답변을 개별적으로 평가 가능
- ✅ 가장 관련성 높은 답변만 선택 가능
- ✅ 메타데이터로 같은 스레드 그룹화 가능

**예시:**
```
질문: "GPU 메모리 부족 문제 해결 방법은?"
답변1: "배치 크기를 줄여보세요..." → 문서 1
답변2: "gradient checkpointing을 사용하면..." → 문서 2
답변3: "mixed precision training을..." → 문서 3
```

### 2. 메타데이터 설계

```python
payload = {
    # === 필터링 핵심 필드 ===
    "course": "level2_cv",              # 과정명 (가장 중요한 필터)
    "course_level": "level2",           # level2 or level3
    "course_topic": "cv",               # cv, nlp, recsys, common
    "generation": "4",                  # 기수
    
    # === 시간 정보 ===
    "date": "2022-11-15",              # YYYY-MM-DD
    "year": 2022,                      # 연도별 필터
    "year_month": "2022-11",           # 월별 필터
    "timestamp": 1668470400,           # Unix timestamp
    
    # === 문서 타입 ===
    "doc_type": "qa_pair",             # 문서 유형
    "has_bot_answer": false,           # 봇 답변 여부
    "is_accepted": false,              # 채택된 답변 (추후 확장)
    
    # === 품질 지표 ===
    "has_reactions": true,             # 반응 유무
    "reaction_count": 15,              # 총 반응 수 (인기도)
    "answer_count": 3,                 # 해당 질문의 총 답변 수
    "answer_index": 0,                 # 답변 순서 (0=첫 답변)
    
    # === 텍스트 필드 ===
    "question_text": "질문 전문",
    "answer_text": "답변 전문",
    "question_user": "홍길동",
    "answer_user": "김철수",
    
    # === 추적 정보 ===
    "thread_id": "gen4_cv_20221115_001",  # 스레드 고유 ID
    "qa_id": "gen4_cv_20221115_001_a0",   # 문서 고유 ID
    "source_file": "2022-11-15_qa.json"
}
```

### 3. 임베딩 텍스트 구성

**추천 형식:**
```python
embedding_text = f"""과정: {course}
기수: {generation}

질문: {question_text}

답변: {answer_text}

작성자: {answer_user}"""
```

**이유:**
- 과정 정보를 포함하여 도메인 특화 검색
- 질문 컨텍스트 유지
- 답변 작성자 정보로 신뢰도 판단 가능

### 4. Qdrant 컬렉션 설정

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# 컬렉션 생성
client.create_collection(
    collection_name="slack_qa",
    vectors_config=VectorParams(
        size=768,  # 임베딩 차원 (모델에 따라 조정)
        distance=Distance.COSINE  # 코사인 유사도
    )
)

# 인덱스는 자동 생성됨 (payload 필드에 자동 인덱싱)
```

## 🔍 검색 전략

### 기본 검색
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 단순 벡터 검색
results = client.search(
    collection_name="slack_qa",
    query_vector=embedding_model.encode(query),
    limit=10
)
```

### 과정별 필터링
```python
# level2_cv 과정만 검색
results = client.search(
    collection_name="slack_qa",
    query_vector=embedding_model.encode("GPU 메모리 부족"),
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

### 복합 필터링
```python
from qdrant_client.models import Range

# level2_cv + 2023년 이후 + 봇 답변 제외
results = client.search(
    collection_name="slack_qa",
    query_vector=embedding_model.encode("데이터 증강 기법"),
    query_filter=Filter(
        must=[
            FieldCondition(key="course", match=MatchValue(value="level2_cv")),
            FieldCondition(key="year", range=Range(gte=2023)),
            FieldCondition(key="has_bot_answer", match=MatchValue(value=False))
        ]
    ),
    limit=10,
    score_threshold=0.7  # 유사도 임계값
)
```

### 인기도 기반 부스팅
```python
# 반응이 많은 답변 우선 (payload를 활용한 정렬)
results = client.search(
    collection_name="slack_qa",
    query_vector=embedding_model.encode(query),
    query_filter=Filter(
        must=[
            FieldCondition(key="has_reactions", match=MatchValue(value=True))
        ]
    ),
    limit=20  # 더 많이 가져와서 정렬
)

# 결과를 reaction_count로 재정렬
sorted_results = sorted(
    results,
    key=lambda x: x.payload.get("reaction_count", 0),
    reverse=True
)[:10]
```

### 하이브리드 검색 (벡터 + BM25)
```python
# Qdrant의 하이브리드 검색 활용
from qdrant_client.models import SearchRequest, Fusion

# 텍스트 검색과 벡터 검색 결합
results = client.query_batch_points(
    collection_name="slack_qa",
    requests=[
        SearchRequest(
            vector=embedding_model.encode(query),
            limit=20,
            filter=course_filter
        )
    ],
    # 추가적으로 full-text search 결과와 결합 가능
)
```

## 📊 권장 임베딩 모델

### 옵션 1: OpenAI (유료, 고성능)
```python
from openai import OpenAI

client = OpenAI()

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",  # 3072 차원
        input=text
    )
    return response.data[0].embedding
```

### 옵션 2: HuggingFace 한국어 특화 (무료, 추천)
```python
from sentence_transformers import SentenceTransformer

# 한국어 특화 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 768 차원

def embed_text(text: str) -> list[float]:
    return model.encode(text, convert_to_numpy=True).tolist()
```

### 옵션 3: 다국어 모델
```python
# 다국어 지원 (한국어 포함)
model = SentenceTransformer('intfloat/multilingual-e5-large')  # 1024 차원

def embed_text(text: str) -> list[float]:
    # e5 모델은 query에 prefix 추가 권장
    text_with_prefix = f"query: {text}"
    return model.encode(text_with_prefix, convert_to_numpy=True).tolist()
```

## 🔄 데이터 처리 파이프라인

```
1. JSON 로드
   ↓
2. Q&A 페어 생성 (질문 + 각 답변)
   ↓
3. 메타데이터 추출 및 정제
   ↓
4. 임베딩 텍스트 생성
   ↓
5. 벡터 임베딩 수행
   ↓
6. Qdrant에 배치 저장 (100개씩)
   ↓
7. 진행상황 로깅
```

## 💡 검색 최적화 팁

### 1. 과정별 컬렉션 분리 (선택사항)
```python
# 각 과정마다 별도 컬렉션
collections = {
    "slack_qa_cv": "level2_cv + level3_cv 데이터",
    "slack_qa_nlp": "level2_nlp + level3_nlp 데이터",
    "slack_qa_common": "공통 데이터"
}
```

**장점:**
- 검색 속도 향상
- 과정별 최적화 가능

**단점:**
- 관리 복잡도 증가
- 크로스 도메인 검색 불가

**권장:** 초기에는 단일 컬렉션으로 시작, 필요시 분리

### 2. 인덱스 최적화
```python
# Qdrant는 자동으로 payload 필드에 인덱스 생성
# 자주 사용하는 필터 필드:
# - course
# - generation
# - year
# - has_bot_answer
```

### 3. 캐싱 전략
```python
# 자주 검색되는 쿼리 캐싱
from functools import lru_cache

@lru_cache(maxsize=100)
def search_with_cache(query: str, course: str = None):
    # 검색 수행
    pass
```

### 4. 리랭킹 (Re-ranking)
```python
# 1차: 벡터 검색으로 후보 30개 추출
candidates = vector_search(query, limit=30)

# 2차: Cross-encoder로 정밀 점수 계산
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

scores = reranker.predict([
    [query, doc.payload["answer_text"]] 
    for doc in candidates
])

# 상위 10개 반환
top_results = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)[:10]
```

## 📈 확장성 고려사항

### 1. 증분 업데이트
```python
# 새로운 Q&A 추가 시
def add_new_qa(qa_data: dict):
    # 임베딩 생성
    # Qdrant에 추가
    # 기존 데이터는 유지
    pass
```

### 2. 버전 관리
```python
payload = {
    # ...
    "version": "1.0",           # 데이터 버전
    "ingestion_date": "2024-11-20",  # 처리 날짜
}
```

### 3. A/B 테스트
```python
# 다른 임베딩 모델 비교
collections = {
    "slack_qa_v1": "openai embeddings",
    "slack_qa_v2": "korean model embeddings"
}
```

## 🎯 실전 사용 예시

### 시나리오 1: 학생의 질문 검색
```python
query = "GPU 메모리 부족 문제 해결"
course = "level2_cv"

results = search_qa(
    query=query,
    course=course,
    year_from=2023,  # 최근 데이터 우선
    limit=5
)
```

### 시나리오 2: 유사 질문 찾기
```python
# 중복 질문 탐지
similar_questions = search_qa(
    query=new_question,
    doc_type="qa_pair",
    score_threshold=0.85,  # 높은 유사도만
    limit=3
)
```

### 시나리오 3: 인기 답변 찾기
```python
# 많은 반응을 받은 답변
popular_answers = search_qa(
    query=query,
    has_reactions=True,
    sort_by="reaction_count",
    limit=10
)
```

## 🔧 다음 단계

1. ✅ 데이터 전처리 완료
2. 🔄 VectorDB 저장 스크립트 작성 (다음 단계)
3. 🔄 검색 API 구현
4. 🔄 웹 인터페이스 연동
5. 🔄 성능 모니터링 및 최적화

