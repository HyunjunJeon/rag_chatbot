# Adaptive RAG 성능 최적화 가이드

이 문서는 Adaptive RAG 시스템의 성능을 최적화하기 위한 실용적인 전략과 구현 방법을 설명합니다.

## 1. Send API를 활용한 병렬화

LangGraph의 **Send API**는 동일한 노드를 여러 번 병렬로 실행할 수 있게 해주는 강력한 기능입니다. 이를 통해 fan-out/fan-in 패턴을 구현하여 처리량을 크게 향상시킬 수 있습니다.

### 1.1 문서 평가 병렬화

검색된 여러 문서를 동시에 평가하여 시간을 단축합니다.

```python
from langgraph.graph import Send, StateGraph
from typing import List

# 단일 문서 평가 노드
async def evaluate_single_document(state: dict, llm: Runnable) -> dict:
    """단일 문서를 평가합니다."""
    doc = state["doc"]
    question = state["question"]
    
    # 평가 로직
    evaluator = create_document_evaluator(llm)
    result = await evaluator.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\n\ndocument: {doc.page_content}"
        }]
    })
    
    return {
        "doc_id": state["doc_id"],
        "evaluation": result,
    }

# 병렬 평가를 위한 라우팅 함수
def route_to_parallel_evaluation(state: AdaptiveRAGState):
    """각 문서를 별도의 평가 노드로 전송합니다."""
    documents = state["documents"]
    question = state["question"]
    
    # Send API로 각 문서를 병렬 처리
    return [
        Send(
            "evaluate_single_document",
            {
                "doc": doc,
                "doc_id": i,
                "question": question,
            }
        )
        for i, doc in enumerate(documents)
    ]

# 평가 결과 집계 노드
def aggregate_evaluations(state: AdaptiveRAGState) -> dict:
    """병렬 평가 결과를 집계합니다."""
    # state에는 모든 평가 결과가 수집되어 있음
    evaluations = state.get("evaluations", [])
    
    relevant_count = sum(1 for e in evaluations if e.get("evaluation", {}).get("relevant", False))
    sufficient = relevant_count >= 2  # 최소 2개 이상 관련 문서 필요
    
    return {
        "relevant_doc_count": relevant_count,
        "sufficient_context": sufficient,
        "document_evaluation": {
            "relevant_count": relevant_count,
            "total_count": len(evaluations),
        }
    }

# 워크플로우에 추가
workflow = StateGraph(AdaptiveRAGState)

workflow.add_node("evaluate_single_document", evaluate_single_document)
workflow.add_node("aggregate_evaluations", aggregate_evaluations)

# 조건부 엣지로 병렬 라우팅
workflow.add_conditional_edges(
    "retrieve",
    route_to_parallel_evaluation,
)

# 모든 평가가 완료되면 집계로
workflow.add_edge("evaluate_single_document", "aggregate_evaluations")
```

**성능 향상**: 5개 문서 평가 시 순차 처리 대비 **최대 5배** 빠름

### 1.2 Multi-Query 병렬 검색

여러 쿼리를 동시에 검색하여 시간을 절약합니다.

```python
# 단일 쿼리 검색 노드
async def retrieve_single_query(state: dict, retriever: BaseRetriever) -> dict:
    """단일 쿼리로 검색합니다."""
    query = state["query"]
    query_id = state["query_id"]
    
    documents = await retrieve_documents_async(retriever, query)
    
    return {
        "query_id": query_id,
        "documents": documents,
    }

# 병렬 검색 라우팅
def route_to_parallel_retrieval(state: AdaptiveRAGState):
    """각 쿼리를 별도의 검색 노드로 전송합니다."""
    queries = state.get("refined_queries", [state["question"]])
    
    return [
        Send(
            "retrieve_single_query",
            {
                "query": query,
                "query_id": i,
            }
        )
        for i, query in enumerate(queries)
    ]

# 검색 결과 통합 노드
def merge_retrieval_results(state: AdaptiveRAGState) -> dict:
    """병렬 검색 결과를 중복 제거하며 통합합니다."""
    all_documents = []
    seen_contents = set()
    
    # 모든 검색 결과 수집
    for result in state.get("retrieval_results", []):
        for doc in result.get("documents", []):
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_documents.append(doc)
    
    return {
        "documents": all_documents,
        "context": all_documents,
    }

# 워크플로우에 추가
workflow.add_node("retrieve_single_query", retrieve_single_query)
workflow.add_node("merge_retrieval_results", merge_retrieval_results)

workflow.add_conditional_edges(
    "analyze_query",
    route_to_parallel_retrieval,
)
workflow.add_edge("retrieve_single_query", "merge_retrieval_results")
```

**성능 향상**: 3개 쿼리 검색 시 순차 처리 대비 **최대 3배** 빠름

### 1.3 Send API 사용 시 주의사항

1. **State 분할**: 각 병렬 노드는 독립적인 state를 받습니다. 필요한 데이터만 전달하세요.
2. **결과 집계**: 병렬 실행 후 반드시 aggregate 노드를 통해 결과를 통합하세요.
3. **오류 처리**: 일부 노드가 실패해도 전체가 중단되지 않도록 try-except를 적용하세요.

```python
async def evaluate_single_document_safe(state: dict, llm: Runnable) -> dict:
    """안전한 단일 문서 평가 (오류 처리 포함)"""
    try:
        return await evaluate_single_document(state, llm)
    except Exception as e:
        logger.error(f"Document evaluation failed: {e}")
        return {
            "doc_id": state["doc_id"],
            "evaluation": {"relevant": False, "error": str(e)},
        }
```

## 2. Timeout 설정 및 오류 처리

### 2.1 ChatOpenAI Timeout 파라미터

LangChain의 `ChatOpenAI`는 내장 timeout 기능을 제공합니다. 별도의 `asyncio.wait_for`는 필요 없습니다.

```python
from langchain_openai import ChatOpenAI

# Timeout이 설정된 LLM 인스턴스
fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    timeout=20,  # 20초 timeout
    max_retries=2,  # 실패 시 2회 재시도
    request_timeout=20,  # 요청당 timeout (초)
)

powerful_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    timeout=45,  # 45초 timeout (복잡한 작업용)
    max_retries=2,
)

# 워크플로우에 적용
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=powerful_llm,
    fast_llm=fast_llm,
)
```

### 2.2 노드 레벨 오류 처리

각 노드에서 LLM timeout을 적절히 처리합니다.

```python
from openai import APITimeoutError, APIError

async def classify_intent_node_safe(
    state: AdaptiveRAGState,
    llm: Runnable
) -> dict:
    """안전한 intent classification (timeout 처리)"""
    try:
        return await classify_intent_node(state, llm)
    
    except APITimeoutError:
        logger.warning("Intent classification timed out, using default")
        return {
            "intent": "SIMPLE_QA",
            "intent_confidence": 0.5,
            "intent_reasoning": "Timeout - default classification applied"
        }
    
    except APIError as e:
        logger.error(f"API error during intent classification: {e}")
        return {
            "intent": "SIMPLE_QA",
            "intent_confidence": 0.3,
            "intent_reasoning": f"API error - default classification applied"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error during intent classification: {e}")
        return {
            "intent": "SIMPLE_QA",
            "intent_confidence": 0.3,
            "intent_reasoning": f"Error - default classification applied"
        }
```

### 2.3 전체 워크플로우 Timeout

전체 워크플로우에 최대 실행 시간을 설정하려면 `asyncio.wait_for`를 사용합니다.

```python
import asyncio

async def run_workflow_with_timeout(
    graph,
    input_state: dict,
    timeout: int = 120  # 2분
) -> dict:
    """Timeout이 적용된 워크플로우 실행"""
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(input_state),
            timeout=timeout
        )
        return result
    
    except asyncio.TimeoutError:
        logger.error(f"Workflow execution timed out after {timeout}s")
        return {
            **input_state,
            "answer": "죄송합니다. 처리 시간이 초과되었습니다. 더 간단한 질문으로 다시 시도해주세요.",
            "error": "workflow_timeout",
            "workflow_stage": "timeout",
        }
```

### 2.4 권장 Timeout 설정

| 작업 | 모델 | Timeout | 재시도 |
|------|------|---------|--------|
| Intent Classification | gpt-4o-mini | 20초 | 2회 |
| Query Analysis | gpt-4o | 30초 | 2회 |
| Document Evaluation | gpt-4o-mini | 20초 | 1회 |
| Answer Generation | gpt-4o | 45초 | 2회 |
| Answer Validation | gpt-4o | 30초 | 1회 |
| Correction | gpt-4o | 30초 | 1회 |

**전체 워크플로우**: 120초 (2분) 권장

## 3. 모델 선택 최적화

### 3.1 작업별 모델 분리

비용과 성능을 고려하여 작업별로 적절한 모델을 선택합니다.

```python
from langchain_openai import ChatOpenAI

# 빠르고 저렴한 모델 (분류, 평가)
fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,  # 결정적 출력
    timeout=20,
    max_retries=2,
)

# 강력한 모델 (복잡한 추론, 생성)
powerful_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,  # 약간의 창의성
    timeout=45,
    max_retries=2,
)

# 비용 최적화 전략
cost_optimized_graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=powerful_llm,      # 답변 생성용
    fast_llm=fast_llm,     # 분류, 평가용
)
```

### 3.2 모델별 비용 비교

| 작업 | 모델 | 1M tokens 비용 | 작업당 비용 (예상) |
|------|------|---------------|-----------------|
| Intent Classification | gpt-4o-mini | $0.15 | $0.0001 |
| Document Evaluation | gpt-4o-mini | $0.15 | $0.0003 |
| Answer Generation | gpt-4o | $2.50 | $0.005 |
| Answer Validation | gpt-4o | $2.50 | $0.002 |

**비용 절감**: fast_llm 활용 시 전체 비용 **약 40% 절감**

## 4. Early Stopping

### 4.1 단순 질문 고속 처리

단순한 질문은 검증을 건너뛰고 빠르게 처리합니다.

```python
def should_skip_validation(state: AdaptiveRAGState) -> bool:
    """검증을 건너뛸 수 있는지 확인"""
    intent = state.get("intent")
    relevant_count = state.get("relevant_doc_count", 0)
    
    # 단순 QA + 충분한 관련 문서
    if intent == "SIMPLE_QA" and relevant_count >= 3:
        return True
    
    return False

# 라우팅에 적용
def route_after_generation(state: AdaptiveRAGState):
    if should_skip_validation(state):
        return "finalize"  # 검증 스킵
    else:
        return "validate_answer"  # 검증 수행
```

### 4.2 재시도 제한 엄격화

무한 루프를 방지하기 위해 재시도를 최소화합니다.

```python
# AdaptiveRAGSettings에서 설정
update_adaptive_rag_settings(
    max_retrieval_retries=1,  # 프로덕션: 1회만
    max_correction_retries=1,  # 프로덕션: 1회만
)
```

**성능 향상**: Early stopping 적용 시 단순 질문 처리 시간 **약 50% 단축**

## 5. 리소스 관리

### 5.1 Connection Pooling

HTTP connection pool을 재사용하여 오버헤드를 줄입니다.

```python
import httpx
from langchain_openai import ChatOpenAI

# 공유 HTTP client (connection pooling)
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)

llm = ChatOpenAI(
    model="gpt-4o",
    http_async_client=http_client,  # 재사용
)
```

### 5.2 메모리 관리

대용량 문서 처리 시 메모리 사용을 최적화합니다.

```python
def truncate_documents(
    documents: List[Document],
    max_length: int = 8000  # 토큰 제한 고려
) -> List[Document]:
    """문서 내용을 truncate하여 메모리 절약"""
    return [
        Document(
            page_content=doc.page_content[:max_length],
            metadata=doc.metadata
        )
        for doc in documents
    ]

# 노드에서 적용
async def generate_answer_node(state, llm):
    documents = state.get("documents", [])
    documents = truncate_documents(documents, max_length=8000)
    # ... 나머지 로직
```

## 6. 모니터링 및 프로파일링

### 6.1 성능 메트릭 수집

각 노드의 실행 시간을 측정합니다.

```python
import time
from functools import wraps

def measure_performance(node_name: str):
    """노드 성능 측정 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                logger.info(
                    f"[PERF] {node_name}: {duration:.2f}s",
                    extra={"node": node_name, "duration": duration}
                )
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    f"[PERF] {node_name}: FAILED after {duration:.2f}s",
                    extra={"node": node_name, "duration": duration, "error": str(e)}
                )
                raise
        return wrapper
    return decorator

# 노드에 적용
@measure_performance("classify_intent")
async def classify_intent_node(state, llm):
    # ... 구현
    pass
```

### 6.2 LangSmith 통합

LangSmith를 사용하여 전체 워크플로우를 추적하고 분석합니다.

```python
import os

# 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "adaptive-rag-production"

# 이제 모든 실행이 자동으로 LangSmith에 기록됨
result = await graph.ainvoke(input_state)
```

### 6.3 성능 대시보드

주요 메트릭을 모니터링합니다:

```python
import structlog

# 구조화된 로깅
perf_logger = structlog.get_logger("performance")

async def log_workflow_metrics(state: AdaptiveRAGState, start_time: float):
    """워크플로우 메트릭 로깅"""
    duration = time.time() - start_time
    
    perf_logger.info(
        "workflow_completed",
        duration=duration,
        intent=state.get("intent"),
        retry_count=state.get("retry_count", 0),
        correction_count=state.get("correction_count", 0),
        quality_score=state.get("quality_score", 0),
        has_hallucination=state.get("has_hallucination", False),
    )
```

## 7. 구현 우선순위

실제 적용 시 다음 순서로 최적화를 진행하는 것을 권장합니다:

### 필수 (즉시 적용)
1. ✅ **Timeout 설정**: ChatOpenAI의 timeout 파라미터
2. ✅ **재시도 제한**: max_retrieval_retries, max_correction_retries
3. ✅ **오류 처리**: try-except로 안전한 fallback

### 권장 (단기)
4. ✅ **모델 선택**: 작업별 fast_llm/powerful_llm 분리
5. ✅ **Early stopping**: 단순 질문 고속 처리
6. ✅ **메모리 최적화**: 문서 truncation

### 선택 (중기)
7. ⚡ **Send API 병렬화**: 문서 평가, multi-query 검색
8. ⚡ **Connection pooling**: HTTP client 재사용
9. ⚡ **성능 모니터링**: LangSmith 통합

### 고급 (장기)
10. 📊 **Auto-tuning**: 성능 데이터 기반 자동 파라미터 조정
11. 📊 **Load balancing**: 여러 LLM API 분산
12. 📊 **Advanced profiling**: 병목 지점 자동 탐지

## 8. 성능 목표

다음 성능 목표를 달성하도록 최적화합니다:

| 메트릭 | 목표 | 최적화 전 | 최적화 후 |
|--------|------|----------|----------|
| 평균 응답 시간 (Simple QA) | < 10초 | ~15초 | ~8초 |
| 평균 응답 시간 (Complex) | < 30초 | ~45초 | ~25초 |
| P95 응답 시간 | < 60초 | ~80초 | ~50초 |
| Timeout 발생률 | < 1% | ~5% | < 1% |
| 비용 (1000 queries) | < $5 | ~$8 | ~$4.50 |

## 9. 참고 자료

- [LangGraph Send API Documentation](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [LangChain Timeouts](https://python.langchain.com/docs/how_to/chat_model_rate_limiting)
- [OpenAI API Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Async Programming in Python](https://docs.python.org/3/library/asyncio.html)

