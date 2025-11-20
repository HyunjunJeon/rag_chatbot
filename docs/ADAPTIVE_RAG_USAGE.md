# Adaptive RAG 사용 가이드

이 문서는 새롭게 구현된 Adaptive RAG 시스템의 사용 방법을 설명합니다.

## 개요

Adaptive RAG는 사용자 질문의 의도를 분석하고, 최적의 검색 및 생성 전략을 적용하여 고품질 답변을 제공하는 시스템입니다.

### 주요 특징

- **의도 기반 적응**: 질문 유형에 따라 다른 처리 전략 적용
- **품질 검증**: Hallucination 검출 및 답변 품질 평가
- **자동 교정**: 문제 발견 시 자동으로 개선 시도
- **병렬 처리**: LangGraph Send API를 활용한 고성능 처리
- **확장 가능**: 새로운 에이전트나 전략 추가 용이

## 빠른 시작

### 1. 기본 사용

```python
from langchain_openai import ChatOpenAI
from naver_connect_chatbot.agent.graph.workflow import build_adaptive_rag_graph
from naver_connect_chatbot.rag import get_hybrid_retriever
from langchain_core.documents import Document

# 준비: Retriever와 LLM
documents = [
    Document(page_content="PyTorch는 딥러닝 프레임워크입니다."),
    Document(page_content="TensorFlow와 함께 널리 사용됩니다."),
]

retriever = get_hybrid_retriever(
    documents=documents,
    embedding_model=embeddings,
    qdrant_url="http://localhost:6333",
    collection_name="my_collection"
)

# LLM 설정 (timeout 포함)
llm = ChatOpenAI(
    model="gpt-4o",
    timeout=45,  # 45초 timeout
    max_retries=2,
)
fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=20,  # 20초 timeout
    max_retries=2,
)

# 워크플로우 생성
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=llm,
    fast_llm=fast_llm
)

# 실행
result = await graph.ainvoke({
    "question": "PyTorch란 무엇인가요?",
    "max_retries": 2,
})

print(result["answer"])
```

### 2. 설정 커스터마이징

```python
from naver_connect_chatbot.config import (
    get_adaptive_rag_settings,
    update_adaptive_rag_settings
)

# 현재 설정 확인
settings = get_adaptive_rag_settings()
print(f"Max retries: {settings.max_retrieval_retries}")
print(f"Min quality score: {settings.min_quality_score}")

# 설정 변경
update_adaptive_rag_settings(
    max_retrieval_retries=1,  # 프로덕션에서는 낮게
    max_correction_retries=1,
    min_quality_score=0.8,
    enable_correction=True,
)
```

## 워크플로우 단계

Adaptive RAG는 다음 단계로 구성됩니다:

### 1. Intent Classification (의도 분류)
질문을 다음 카테고리로 분류합니다:
- `SIMPLE_QA`: 단순 사실 확인
- `COMPLEX_REASONING`: 복잡한 추론 필요
- `EXPLORATORY`: 탐색적 질문
- `CLARIFICATION_NEEDED`: 불명확한 질문

```python
# 의도 분류만 별도로 사용
from naver_connect_chatbot.agent.agents import classify_intent

result = classify_intent("PyTorch란?", llm)
print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence}")
```

### 2. Query Analysis (쿼리 분석)
쿼리 품질을 분석하고 개선합니다.

```python
from naver_connect_chatbot.agent.agents import analyze_query

result = analyze_query("이거 어떻게 해요?", "CLARIFICATION_NEEDED", llm)
print(f"Improved queries: {result.improved_queries}")
```

### 3. Retrieval (검색)
Hybrid 검색 (Dense + Sparse)을 수행합니다.

### 4. Document Evaluation (문서 평가)
검색된 문서의 관련성과 충분성을 평가합니다.

```python
from naver_connect_chatbot.agent.agents import evaluate_documents

result = evaluate_documents("PyTorch란?", documents, llm)
print(f"Sufficient: {result.sufficient}")
print(f"Relevant count: {result.relevant_count}")
```

### 5. Answer Generation (답변 생성)
의도에 맞는 전략으로 답변을 생성합니다.

```python
from naver_connect_chatbot.agent.agents import generate_answer

answer = generate_answer("PyTorch란?", documents, "SIMPLE_QA", llm)
print(answer)
```

### 6. Answer Validation (답변 검증)
Hallucination과 품질을 검증합니다.

```python
from naver_connect_chatbot.agent.agents import validate_answer

result = validate_answer("PyTorch란?", documents, answer, llm)
print(f"Has hallucination: {result.has_hallucination}")
print(f"Quality score: {result.quality_score}")
```

### 7. Correction (교정)
문제 발견 시 교정 전략을 결정합니다.

```python
from naver_connect_chatbot.agent.agents import determine_correction_strategy

strategy = determine_correction_strategy(validation_result, answer, llm)
print(f"Action: {strategy.action}")  # REGENERATE, REFINE_QUERY, etc.
```

## 고급 사용법

### 1. 특정 단계만 사용

각 에이전트는 독립적으로 사용할 수 있습니다:

```python
# Intent만 분류
from naver_connect_chatbot.agent.agents.intent_classifier import create_intent_classifier

classifier = create_intent_classifier(llm)
result = await classifier.ainvoke({
    "messages": [{"role": "user", "content": "What is PyTorch?"}]
})
```

### 2. 커스텀 프롬프트

프롬프트를 수정하려면 YAML 파일을 편집합니다:

```yaml
# app/naver_connect_chatbot/prompts/templates/intent_classification.yaml
_type: chat_messages
metadata:
  name: intent_classification
  version: "1.1"

messages:
  - role: system
    content: |
      당신만의 커스텀 프롬프트...
```

### 3. Timeout 설정

각 LLM에 적절한 timeout을 설정합니다:

```python
# 작업별 timeout 설정
intent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=20,  # Intent classification: 20초
)

generation_llm = ChatOpenAI(
    model="gpt-4o",
    timeout=45,  # Answer generation: 45초
)

validation_llm = ChatOpenAI(
    model="gpt-4o",
    timeout=30,  # Validation: 30초
)
```

### 4. 전체 워크플로우 Timeout

```python
import asyncio

async def run_with_timeout(question: str, timeout: int = 120):
    """전체 워크플로우에 timeout 적용"""
    try:
        result = await asyncio.wait_for(
            graph.ainvoke({
                "question": question,
                "max_retries": 1,
            }),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return {
            "question": question,
            "answer": "처리 시간이 초과되었습니다.",
            "error": "timeout"
        }

# 사용
result = await run_with_timeout("복잡한 질문...", timeout=120)
```

## 에러 처리

### API Timeout 처리

```python
from openai import APITimeoutError, APIError

try:
    result = await graph.ainvoke(input_state)
except APITimeoutError:
    print("LLM API timeout occurred")
    # Fallback 처리
except APIError as e:
    print(f"API error: {e}")
    # 재시도 또는 대체 로직
```

### 재시도 제한 초과

워크플로우는 자동으로 재시도 제한을 처리하지만, 최종 결과를 확인해야 합니다:

```python
result = await graph.ainvoke({
    "question": "...",
    "max_retries": 1,
})

if result.get("retry_count", 0) >= 1:
    print("Max retries exceeded, best effort answer provided")
    print(f"Answer quality: {result.get('quality_score', 0)}")
```

## 모니터링

### LangSmith 통합

```python
import os

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "adaptive-rag-prod"

# 이제 모든 실행이 LangSmith에 기록됩니다
result = await graph.ainvoke(input_state)
```

LangSmith에서 확인할 수 있는 정보:
- 각 노드의 실행 시간
- LLM 호출 세부사항 (prompt, response, tokens)
- 에러 및 재시도 기록
- 전체 워크플로우 시각화

### 로깅

```python
from naver_connect_chatbot.config import logger

# 로그 레벨 조정
logger.setLevel("DEBUG")  # 개발 환경

# 프로덕션에서는 INFO
logger.setLevel("INFO")

# 실행
result = await graph.ainvoke(input_state)
# 각 단계의 로그가 출력됩니다
```

## 성능 최적화

### 1. 모델 선택

비용과 성능의 균형을 맞추세요:

```python
# 저비용 구성 (추천하지 않음)
fast_llm = ChatOpenAI(model="gpt-4o-mini", timeout=20)
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=fast_llm,  # 모든 작업에 fast model
)

# 균형 구성 (권장)
powerful_llm = ChatOpenAI(model="gpt-4o", timeout=45)
fast_llm = ChatOpenAI(model="gpt-4o-mini", timeout=20)
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=powerful_llm,  # 중요한 작업
    fast_llm=fast_llm,  # 보조 작업
)

# 고품질 구성
powerful_llm = ChatOpenAI(model="gpt-4o", timeout=60)
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=powerful_llm,  # 모든 작업에 강력한 모델
)
```

### 2. Early Stopping

단순한 질문은 검증을 건너뛰어 빠르게 처리:

```python
update_adaptive_rag_settings(
    enable_answer_validation=False,  # 빠른 처리 (주의: 품질 저하 가능)
)
```

### 3. 재시도 최소화

프로덕션에서는 재시도를 최소화하여 응답 시간 단축:

```python
update_adaptive_rag_settings(
    max_retrieval_retries=1,  # 1회만
    max_correction_retries=0,  # 교정 비활성화
)
```

## 문제 해결

### 답변 품질이 낮은 경우

1. **품질 임계값 조정**:
```python
update_adaptive_rag_settings(min_quality_score=0.9)
```

2. **Correction 활성화**:
```python
update_adaptive_rag_settings(
    enable_correction=True,
    max_correction_retries=1
)
```

3. **더 강력한 모델 사용**:
```python
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
```

### 응답이 느린 경우

1. **Timeout 단축**:
```python
llm = ChatOpenAI(model="gpt-4o", timeout=30)  # 45초 → 30초
```

2. **단계 비활성화** (주의: 품질 저하):
```python
update_adaptive_rag_settings(
    enable_query_analysis=False,
    enable_answer_validation=False,
)
```

3. **재시도 제한**:
```python
update_adaptive_rag_settings(
    max_retrieval_retries=0,  # 재시도 없음
    max_correction_retries=0,
)
```

### Timeout이 자주 발생하는 경우

1. **Timeout 증가**:
```python
llm = ChatOpenAI(
    model="gpt-4o",
    timeout=60,  # 더 여유있게
    request_timeout=60,
)
```

2. **문서 길이 제한**:
```python
# 노드에서 문서 truncation
def truncate_docs(docs, max_len=5000):
    return [
        Document(page_content=doc.page_content[:max_len])
        for doc in docs
    ]
```

## 프로덕션 체크리스트

배포 전 다음 사항을 확인하세요:

- [ ] LLM timeout 설정 (20-45초)
- [ ] 재시도 제한 설정 (1-2회)
- [ ] LangSmith 통합 (모니터링)
- [ ] 에러 처리 (try-except)
- [ ] 로그 레벨 설정 (INFO)
- [ ] 품질 임계값 설정 (0.8 이상)
- [ ] 워크플로우 timeout (120초)
- [ ] 비용 모니터링 설정

## 다음 단계

- [최적화 가이드](./ADAPTIVE_RAG_OPTIMIZATION.md) - Send API 병렬화 등 고급 최적화
- [API 문서](../app/naver_connect_chatbot/agent/) - 상세 API 레퍼런스
- [예제](../tests/test_adaptive_rag.py) - 더 많은 사용 예제

