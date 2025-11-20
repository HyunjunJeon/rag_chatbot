# Retriever 통합 가이드

## 개요

이 문서는 `document_processing`의 검증된 Retriever 컴포넌트를 `naver_connect_chatbot`의 LangChain 기반 아키텍처에 통합하는 과정을 설명합니다.

## 목차

- [아키텍처 개요](#아키텍처-개요)
- [통합된 컴포넌트](#통합된-컴포넌트)
- [OpenRouterEmbeddings 사용법](#openrouterembeddings-사용법)
- [Hybrid Retriever 구성](#hybrid-retriever-구성)
- [테스트 및 검증](#테스트-및-검증)
- [트러블슈팅](#트러블슈팅)
- [향후 개선 사항](#향후-개선-사항)

---

## 아키텍처 개요

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    naver_connect_chatbot                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ OpenRouter    │───▶│   Qdrant     │───▶│   Hybrid     │ │
│  │ Embeddings    │    │  (Dense)     │    │  Retriever   │ │
│  └───────────────┘    └──────────────┘    │    (RRF)     │ │
│                                            │              │ │
│  ┌───────────────┐                        │              │ │
│  │  BM25 Index   │───────────────────────▶│              │ │
│  │  (Sparse)     │                        └──────────────┘ │
│  └───────────────┘                                ▼         │
│                                           ┌──────────────┐  │
│                                           │    Agent     │  │
│                                           │   / Chain    │  │
│                                           └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

1. **사용자 질의** → Agent/Chain
2. **질의 임베딩** → OpenRouterEmbeddings
3. **병렬 검색**:
   - **Dense**: Qdrant 벡터 검색
   - **Sparse**: BM25 키워드 검색
4. **결과 융합** → Reciprocal Rank Fusion (RRF)
5. **최종 결과** → Agent/Chain

---

## 통합된 컴포넌트

### 1. OpenRouterEmbeddings

**위치**: `app/naver_connect_chatbot/config/embedding.py`

**특징**:
- LangChain `Embeddings` 인터페이스 완전 호환
- OpenRouter API를 통한 임베딩 생성
- 동기/비동기 메서드 지원
- 자동 배치 처리 및 재시도 로직

**주요 메서드**:
```python
# 동기 메서드
embed_query(text: str) -> List[float]
embed_documents(texts: List[str]) -> List[List[float]]

# 비동기 메서드
async aembed_query(text: str) -> List[float]
async aembed_documents(texts: List[str]) -> List[List[float]]
```

### 2. BM25 인덱스

**위치**: `sparse_index/bm25_slack_qa.pkl`

**내용**:
- 4,581개 Slack Q&A 문서
- 질문/답변 분리 인덱싱
- Kiwi 형태소 분석 기반

**데이터 구조**:
```python
{
    "bm25_question": BM25Okapi,    # 질문 인덱스
    "bm25_answer": BM25Okapi,      # 답변 인덱스
    "bm25_combined": BM25Okapi,    # 통합 인덱스
    "doc_ids": List[str],          # 문서 ID
    "documents": List[dict],        # 원본 문서
}
```

### 3. Qdrant 컬렉션

**컬렉션 이름**: `slack_qa`

**정보**:
- 벡터 수: 4,581개
- 벡터 차원: 2560 (qwen3-embedding-4b 모델)
- 메타데이터: course, generation, date, question_text, answer_text 등

---

## OpenRouterEmbeddings 사용법

### 기본 사용

```python
from naver_connect_chatbot.config.embedding import OpenRouterEmbeddings
from pydantic import SecretStr
import os

# 초기화
embeddings = OpenRouterEmbeddings(
    model="qwen/qwen3-embedding-4b",
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
)

# 단일 쿼리 임베딩
query_vector = embeddings.embed_query("GPU 메모리 부족 해결 방법")
print(f"벡터 차원: {len(query_vector)}")

# 다중 문서 임베딩
documents = ["문서1", "문서2", "문서3"]
doc_vectors = embeddings.embed_documents(documents)
print(f"생성된 벡터 수: {len(doc_vectors)}")
```

### 환경 설정

`.env` 파일에 다음을 추가:

```bash
# OpenRouter API
OPENROUTER_API_KEY=sk-or-your-openrouter-api-key-here

# Qdrant 설정
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=slack_qa
QDRANT_EMBEDDING_DIMENSIONS=2560
```

### 고급 설정

```python
embeddings = OpenRouterEmbeddings(
    model="qwen/qwen3-embedding-4b",
    api_key=SecretStr(api_key),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,        # API 호출 타임아웃 (초)
    max_retries=3,       # 최대 재시도 횟수
    batch_size=32,       # 배치 크기
)
```

---

## Hybrid Retriever 구성

### 방법 1: retriever_factory 사용 (권장)

```python
from langchain_core.documents import Document
from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid
from naver_connect_chatbot.config.embedding import OpenRouterEmbeddings
from naver_connect_chatbot.config import settings

# 1. BM25용 문서 로드
import pickle
with open("document_chunks/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

documents = [
    Document(
        page_content=f"질문: {doc['question_text']}\n답변: {doc['answer_text']}",
        metadata=doc
    )
    for doc in bm25_data["documents"]
]

# 2. Embeddings 초기화
embeddings = OpenRouterEmbeddings()

# 3. Hybrid Retriever 구성
hybrid_retriever = build_dense_sparse_hybrid(
    documents=documents,
    embedding_model=embeddings,
    qdrant_url=settings.qdrant.url,
    collection_name=settings.qdrant.collection_name,
    weights=[0.5, 0.5],  # [Sparse, Dense] 가중치
    k=10,
    method=HybridMethod.RRF,
    rrf_c=60,
)

# 4. 검색 수행
results = hybrid_retriever.invoke("GPU 메모리 부족 해결 방법")

for doc in results:
    print(f"과정: {doc.metadata['course']}")
    print(f"질문: {doc.metadata['question_text'][:50]}...")
    print(f"답변: {doc.metadata['answer_text'][:50]}...")
    print()
```

### 방법 2: 수동 구성 (커스터마이징 필요 시)

```python
from qdrant_client import QdrantClient
from naver_connect_chatbot.rag.retriever.qdrant_sdk_retriever import QdrantVDBRetriever
from naver_connect_chatbot.rag.retriever.hybrid_retriever import HybridRetriever, HybridMethod

# 1. Dense Retriever (Qdrant)
client = QdrantClient(url="http://localhost:6333")
dense_retriever = QdrantVDBRetriever(
    client=client,
    embedding_model=embeddings,
    collection_name="slack_qa",
    default_k=10,
)

# 2. Sparse Retriever (BM25)
# BM25 인덱스를 사용한 커스텀 retriever 구현 필요

# 3. Hybrid Retriever
hybrid_retriever = HybridRetriever(
    retrievers=[sparse_retriever, dense_retriever],
    weights=[0.5, 0.5],
    method=HybridMethod.RRF,
    c=60,
)
```

---

## 테스트 및 검증

### 통합 테스트 실행

```bash
# 기본 통합 테스트
python tests/test_integration_retriever.py

# End-to-end 테스트
python tests/test_hybrid_retriever_e2e.py

# 비교 분석
python tests/compare_retrieval_results.py
```

### 단위 테스트

```bash
# pytest 설치 (필요 시)
pip install pytest pytest-asyncio

# OpenRouterEmbeddings 테스트
pytest tests/test_openrouter_embeddings.py -v

# 또는 직접 실행
python tests/verify_openrouter_embeddings.py
```

### 테스트 결과 예시

```
✅ OpenRouterEmbeddings 검증
   - 클래스 로드: 성공
   - LangChain 호환성: 확인
   - 메서드 존재: embed_query, embed_documents, aembed_query, aembed_documents

✅ BM25 인덱스
   - 문서 수: 4,581개
   - 검색 테스트: 성공

✅ Qdrant 컬렉션
   - 벡터 수: 4,581개
   - 벡터 차원: 2560
   - 검색 테스트: 성공

✅ Hybrid Retriever
   - RRF 융합: 성공
   - 최종 결과: 정상 반환
```

---

## 트러블슈팅

### 1. Qdrant 연결 실패

**증상**: `Connection refused` 또는 `Qdrant 연결 실패`

**해결**:
```bash
# Qdrant 실행 확인
docker ps | grep qdrant

# Qdrant 시작 (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 또는 docker-compose
docker-compose up -d qdrant
```

### 2. OpenRouter API 키 오류

**증상**: `OpenRouter API 키가 필요합니다`

**해결**:
1. `.env` 파일에 `OPENROUTER_API_KEY` 추가
2. 환경변수 로드 확인:
   ```python
   import os
   print(os.getenv("OPENROUTER_API_KEY"))  # None이 아니어야 함
   ```

### 3. BM25 인덱스 파일 없음

**증상**: `BM25 인덱스가 없습니다`

**해결**:
```bash
# 인덱스 파일 위치 확인
ls -lh sparse_index/bm25_slack_qa.pkl

# 파일이 없으면 기존 document_chunks에서 복사
cp document_chunks/bm25_index.pkl sparse_index/bm25_slack_qa.pkl
```

### 4. Kiwipiepy 모델 파일 오류

**증상**: `Cannot open language model file 'sj.knlm'`

**해결**:
```bash
# Kiwipiepy 재설치
pip uninstall -y kiwipiepy kiwipiepy_model
pip install kiwipiepy

# Python에서 초기화
python -c "from kiwipiepy import Kiwi; k = Kiwi(); print('OK')"
```

### 5. 설정 파일 로드 오류

**증상**: `SlackSettings bot_token Field required`

**해결**:
- 테스트 스크립트는 직접 import 방식을 사용하여 설정 로드를 우회합니다
- `.env` 파일에 필수 환경변수 추가 (특히 Slack 관련)

---

## 향후 개선 사항

### 1. KiwiBM25Retriever 마이그레이션 (향후)

**현재**: 기존 `bm25_index.pkl` 사용  
**목표**: `KiwiBM25Retriever` 사용

**장점**:
- LangChain `BaseRetriever` 인터페이스
- 고급 Kiwi 기능 활용 (오타 교정, 사용자 사전 등)
- 저장/로드 메커니즘
- 일관된 API

**마이그레이션 스크립트**: `document_processing/rebuild_bm25_for_chatbot.py` (준비됨)

### 2. Reranker 추가

```python
from naver_connect_chatbot.rag.rerank import NaverCloudReranker

# Hybrid 결과를 Reranker로 재정렬
reranker = NaverCloudReranker()
reranked_results = reranker.rerank(query, hybrid_results)
```

### 3. MultiQuery Retriever

```python
from naver_connect_chatbot.rag.retriever_factory import build_multi_query_retriever
from naver_connect_chatbot.config.llm import get_llm

# LLM으로 쿼리 확장
llm = get_llm()
multi_query_retriever = build_multi_query_retriever(
    base_retriever=hybrid_retriever,
    llm=llm,
    num_queries=4,
)
```

### 4. Adaptive RAG

자동으로 검색 전략을 선택하는 Adaptive RAG 시스템:

```python
from naver_connect_chatbot.service.agents import AdaptiveRAGAgent

agent = AdaptiveRAGAgent(
    retriever=hybrid_retriever,
    llm=llm,
)

# 쿼리에 따라 최적의 전략 선택
result = agent.invoke("복잡한 질문")
```

---

## 참고 문서

- [document_processing/README.md](../document_processing/README.md)
- [document_processing/HYBRID_SEARCH_USAGE.md](../document_processing/HYBRID_SEARCH_USAGE.md)
- [document_processing/VECTORDB_USAGE.md](../document_processing/VECTORDB_USAGE.md)
- [ADAPTIVE_RAG_USAGE.md](./ADAPTIVE_RAG_USAGE.md)
- [naver-cloud-api-information.md](./naver-cloud-api-information.md)

---

## 성공 기준 체크리스트

- [x] OpenRouterEmbeddings 클래스 구현
- [x] LangChain Embeddings 인터페이스 호환
- [x] 기존 BM25 인덱스 활용
- [x] Qdrant 컬렉션 연결
- [x] 통합 테스트 작성 및 검증
- [x] End-to-end 테스트 성공
- [x] 비교 분석 완료
- [x] 문서화 완료

---

## 문의 및 지원

통합 과정에서 문제가 발생하면:

1. 트러블슈팅 섹션 확인
2. 테스트 스크립트 실행하여 구체적 오류 확인
3. 로그 파일 확인 (`logs/app_*.log`)
4. 이슈 트래커에 보고

**작성일**: 2025-11-20  
**버전**: 1.0.0  
**작성자**: AI Assistant

