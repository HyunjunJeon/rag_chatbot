# Naver Connect - Chatbot

## 목차

- [설치](#설치)
- [환경 설정](#환경-설정)
- [주요 기능](#주요-기능)
  - [LLM 통합](#llm-통합)
  - [Reranker](#reranker)
- [사용 예제](#사용-예제)

## 설치

```bash
# uv를 사용하여 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --group dev
```

## 환경 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 환경변수를 설정하세요:

### 필수 설정

```bash
# Naver Cloud Embeddings
NAVER_CLOUD_EMBEDDINGS_MODEL_URL=https://...
NAVER_CLOUD_EMBEDDINGS_API_KEY=your_api_key

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=your_collection
```

### LLM 설정 (선택적)

#### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=  # 선택적 (None이면 제한 없음)
OPENAI_ENABLED=true
```

#### OpenRouter

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL_NAME=anthropic/claude-3.5-sonnet
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=  # 선택적
OPENROUTER_ENABLED=true
```

#### Naver Cloud (OpenAI 호환 모드)

```bash
NAVER_CLOUD_OPENAI_COMPATIBLE_BASE_URL=https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-003
NAVER_CLOUD_OPENAI_COMPATIBLE_API_KEY=your_clovastudio_api_key
NAVER_CLOUD_OPENAI_COMPATIBLE_API_GATEWAY_KEY=your_gateway_key  # 선택적
NAVER_CLOUD_OPENAI_COMPATIBLE_DEFAULT_MODEL=HCX-003
NAVER_CLOUD_OPENAI_COMPATIBLE_ENABLED=true
```

### Reranker 설정 (선택적)

```bash
# Clova Studio Reranker
NAVER_CLOUD_RERANKER_ENDPOINT=https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/reranker/{reranker-id}
NAVER_CLOUD_RERANKER_API_KEY=your_clovastudio_api_key
NAVER_CLOUD_RERANKER_API_GATEWAY_KEY=your_gateway_key  # 선택적
NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT=30.0
NAVER_CLOUD_RERANKER_DEFAULT_TOP_K=10
NAVER_CLOUD_RERANKER_ENABLED=true
```

## LangFuse Monitoring

This project includes self-hosted LangFuse v3 for observability and tracing.

### Setup

1. **Generate Secrets**
   ```bash
   openssl rand -hex 32  # Run multiple times for each secret
   ```

2. **Configure Environment**
   ```bash
   cp .env.langfuse.example .env.langfuse
   # Edit .env.langfuse and replace all CHANGEME values
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Access LangFuse UI**
   - Navigate to http://localhost:3000
   - Create account and project
   - Generate API keys: Settings → API Keys → Create new key
   - Add keys to `.env.langfuse`:
     ```bash
     LANGFUSE_PUBLIC_KEY=pk-lf-...
     LANGFUSE_SECRET_KEY=sk-lf-...
     ```

5. **Restart Application**
   ```bash
   docker-compose restart app
   ```

### Features

- **Node-Level Tracing**: Every agent (intent classifier, query analyzer, answer generator, etc.) appears as separate span
- **Slack Metadata**: Traces tagged with user_id, channel_id, thread_ts for filtering
- **Cost Tracking**: LLM usage and costs per conversation
- **Graceful Degradation**: Application works normally if LangFuse is unavailable

### Disabling Monitoring

Set in `.env` or `.env.langfuse`:
```bash
LANGFUSE_ENABLED=false
```

### Architecture

```
Slack @mention
  → SlackHandler creates callback with user metadata
  → graph.ainvoke(config={"callbacks": [handler]})
  → Callback auto-propagates to all node LLM calls
  → Traces visible at http://localhost:3000
```

See `docs/plans/2025-11-21-langfuse-monitoring.md` for detailed architecture.

## 주요 기능

### LLM 통합

OpenAI, OpenRouter, Naver Cloud 세 가지 LLM 제공자를 ChatOpenAI를 통해 통합된 인터페이스로 사용할 수 있습니다.

#### 특징

- **통합 인터페이스**: 모든 제공자가 동일한 `ChatOpenAI` 인터페이스 사용
- **팩토리 패턴**: 제공자별 설정을 자동으로 적용
- **설정 기반**: 환경변수를 통한 간편한 설정 관리
- **타입 안전성**: 완전한 타입 힌트 및 Pydantic 검증
- **확장 가능**: 새로운 OpenAI 호환 제공자 추가 용이

#### 지원 제공자

| 제공자 | 설명 | OpenAI 호환 |
|-------|------|-----------|
| **OpenAI** | OpenAI 공식 API | ✅ |
| **OpenRouter** | 다양한 LLM 통합 서비스 | ✅ |
| **Naver Cloud** | Clova Studio (커스텀 헤더 지원) | ✅ |

### Reranker

Clova Studio Reranker를 활용하여 검색된 문서의 관련도를 재평가하고 순위를 재정렬합니다.

#### 특징

- **HTTPX 기반**: 고성능 비동기 HTTP 클라이언트 사용
- **동기/비동기 지원**: `rerank()` 및 `arerank()` 메서드 제공
- **메타데이터 보존**: 재정렬 점수와 순위를 문서 메타데이터에 추가
- **에러 핸들링**: 상세한 로깅 및 예외 처리
- **타입 안전성**: 완전한 타입 힌트 지원

#### 아키텍처

```
BaseReranker (ABC)
    ├── rerank() - 동기 재정렬 메서드
    └── arerank() - 비동기 재정렬 메서드

ClovaStudioReranker
    ├── Clova Studio API 호출
    ├── HTTPX Client 세션 관리
    ├── 요청/응답 직렬화
    └── 메타데이터 병합
```

## 사용 예제

### LLM 사용법

#### 기본 사용

```python
from naver_connect_chatbot.config import get_chat_model, LLMProvider

# OpenAI 사용
openai_model = get_chat_model(LLMProvider.OPENAI)
response = await openai_model.ainvoke("안녕하세요!")
print(response.content)

# OpenRouter 사용
openrouter_model = get_chat_model(LLMProvider.OPENROUTER)
response = await openrouter_model.ainvoke("Hello!")
print(response.content)

# Naver Cloud 사용
naver_model = get_chat_model(LLMProvider.NAVER_CLOUD)
response = await naver_model.ainvoke("반갑습니다!")
print(response.content)
```

#### 설정 오버라이드

```python
# 특정 모델로 오버라이드
custom_model = get_chat_model(
    LLMProvider.OPENAI,
    model="gpt-4o",  # 기본값 대신 gpt-4o 사용
    temperature=0.9,  # 더 창의적인 응답
    max_tokens=2000,  # 더 긴 응답
)

response = await custom_model.ainvoke("복잡한 질문...")
```

#### 동기 호출

```python
# 동기 메서드도 지원
model = get_chat_model(LLMProvider.OPENAI)
response = model.invoke("질문")
print(response.content)
```

#### LangChain과 통합

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유용한 AI 어시스턴트입니다."),
    ("user", "{question}")
])

# LLM 체인
model = get_chat_model(LLMProvider.OPENAI)
chain = prompt | model | StrOutputParser()

# 실행
result = await chain.ainvoke({"question": "LangChain이란?"})
print(result)
```

#### 여러 제공자 비교

```python
async def compare_providers(question: str):
    """여러 LLM 제공자의 응답 비교"""
    providers = [
        LLMProvider.OPENAI,
        LLMProvider.OPENROUTER,
        LLMProvider.NAVER_CLOUD,
    ]
    
    results = {}
    for provider in providers:
        try:
            model = get_chat_model(provider)
            response = await model.ainvoke(question)
            results[provider.value] = response.content
        except ValueError as e:
            results[provider.value] = f"Error: {e}"
    
    return results

# 실행
results = await compare_providers("AI란 무엇인가?")
for provider, answer in results.items():
    print(f"\n{provider}:")
    print(answer)
```

### Reranker 기본 사용법

```python
from langchain_core.documents import Document
from naver_connect_chatbot.config import settings
from naver_connect_chatbot.rag.rerank import ClovaStudioReranker

# Reranker 초기화
reranker = ClovaStudioReranker.from_settings(settings.reranker)

# 또는 직접 초기화
reranker = ClovaStudioReranker(
    endpoint="https://clovastudio.apigw.ntruss.com/.../reranker/...",
    api_key="your-api-key",
    default_top_k=5,
)

# 문서 재정렬
documents = [
    Document(page_content="AI는 인공지능을 의미합니다.", metadata={"source": "doc1"}),
    Document(page_content="기계학습은 AI의 한 분야입니다.", metadata={"source": "doc2"}),
    Document(page_content="딥러닝은 신경망을 사용합니다.", metadata={"source": "doc3"}),
]

query = "인공지능이란 무엇인가?"

# 상위 3개 문서만 반환
reranked = reranker.rerank(query, documents, top_k=3)

for doc in reranked:
    print(f"Score: {doc.metadata['rerank_score']:.3f}")
    print(f"Rank: {doc.metadata['rerank_rank']}")
    print(f"Content: {doc.page_content}\n")
```

### 비동기 사용법

```python
import asyncio

async def rerank_async():
    reranker = ClovaStudioReranker.from_settings(settings.reranker)
    documents = [...]  # 문서 리스트
    
    reranked = await reranker.arerank("질문", documents, top_k=5)
    return reranked

# 실행
results = asyncio.run(rerank_async())
```

### Retriever와 결합

```python
from naver_connect_chatbot.rag.retriever_factory import build_advanced_hybrid_retriever

# Retriever로 초기 검색
retriever = build_advanced_hybrid_retriever(...)
retrieved_docs = retriever.invoke("질문")

# Reranker로 재정렬
reranker = ClovaStudioReranker.from_settings(settings.reranker)
final_docs = reranker.rerank("질문", retrieved_docs, top_k=5)
```

### 커스텀 Reranker 구현

```python
from naver_connect_chatbot.rag.rerank import BaseReranker

class CustomReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        documents: list[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        # 커스텀 재정렬 로직 구현
        # ...
        return sorted_documents
```

## 테스트

### 기본 테스트 (유닛 테스트)

```bash
# 모든 테스트 실행
pytest

# LLM 테스트만 실행 (유닛 테스트)
pytest tests/test_llm.py -k "not integration"

# Reranker 테스트만 실행
pytest tests/test_rerank.py

# 커버리지 포함
pytest --cov=naver_connect_chatbot

# 특정 테스트 실행
pytest tests/test_llm.py::TestGetChatModel::test_openai_chat_model_creation
pytest tests/test_rerank.py::TestClovaStudioRerankerAPI::test_rerank_success
```

### 통합 테스트 (실제 API 호출)

실제 LLM API를 호출하는 통합 테스트는 별도로 실행할 수 있습니다:

```bash
# 모든 통합 테스트 실행
pytest -m integration -v

# LLM 통합 테스트 실행 (비용 발생 가능)
pytest tests/test_llm.py -m integration -v

# LangFuse 통합 테스트 실행 (LangFuse 서버 필요)
pytest tests/test_langfuse_integration.py -m integration -v

# 수동 테스트 스크립트 (모든 제공자 테스트)
python tests/manual_llm_test.py
```

**주의사항:**
- 통합 테스트는 실제 API를 호출하므로 비용이 발생할 수 있습니다
- `.env` 파일에 API 키와 `ENABLED=true` 설정이 필요합니다
- 네트워크 연결이 필요합니다
- LangFuse 통합 테스트는 `docker-compose up -d`로 LangFuse 서버를 먼저 실행해야 합니다
- 자세한 내용은 `tests/INTEGRATION.md` 참조

## API 문서

- [Clova Studio Reranker 공식 문서](https://api.ncloud-docs.com/docs/clovastudio-reranker)