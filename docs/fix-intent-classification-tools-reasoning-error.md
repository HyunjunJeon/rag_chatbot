# Intent Classification Tools + Reasoning 에러 해결 계획

## 문제 요약

### 증상
RAG Workflow의 `classify_intent_node`에서 다음 에러가 발생합니다:
```
Invalid parameter: tools, reasoning
```

워크플로는 fallback 로직으로 계속 동작하지만, 실제 Intent Classification이 정상 수행되지 않습니다.

### 영향 범위
- `classify_intent_node` (nodes.py:66)
- `analyze_query_node` (nodes.py:114)
- 두 노드 모두 `create_agent(tools=[...])` 패턴 사용

---

## Root Cause 분석

### 1. 기술적 원인

**CLOVA Studio HCX-007 API 제약:**
- `tools` 파라미터와 `reasoning_effort` 파라미터를 **동시에 사용할 수 없음**
- 이는 CLOVA Studio API의 설계상 제약

**현재 코드 구조:**
```python
# intent_classifier.py
agent = create_agent(
    model=llm,                          # ← LLM이 reasoning 모드면 에러
    tools=[emit_intent_classification], # ← tools 사용
    system_prompt=enhanced_prompt,
    name="intent_classifier",
)
```

**에러 발생 조건:**
1. `get_chat_model(use_reasoning=True)` 또는 `reasoning_effort` 설정된 LLM
2. 이 LLM을 `create_intent_classifier()` 또는 `create_query_analyzer()`에 전달
3. `create_agent()`가 내부적으로 `llm.bind_tools()`를 호출
4. CLOVA API가 `tools` + `reasoning_effort` 조합 거부

### 2. 관련 파일

| 파일 | 역할 | 문제 |
|------|------|------|
| `service/agents/intent_classifier.py` | Intent 분류 에이전트 | `create_agent(tools=[...])` 사용 |
| `service/agents/query_analyzer.py` | Query 분석 에이전트 | `create_agent(tools=[...])` 사용 |
| `service/graph/nodes.py` | 워크플로 노드 | 에러 발생 시 fallback 처리 |
| `service/graph/workflow.py` | 워크플로 구성 | LLM 할당 로직 |
| `config/llm.py` | LLM 팩토리 | reasoning 모드 설정 |

### 3. 현재 동작 (Fallback)

에러 발생 시 fallback이 동작합니다 (`nodes.py:105-111`):
```python
except Exception as e:
    logger.error(f"Intent classification error: {e}")
    return {
        "intent": "SIMPLE_QA",  # 기본값
        "intent_confidence": 0.5,
        "intent_reasoning": f"Error during classification: {str(e)}",
    }
```

**문제점:**
- 모든 질의가 `SIMPLE_QA`로 분류됨
- 복잡한 추론이나 탐색적 질문에 최적화된 처리 불가
- Query Analysis도 동일한 문제로 기본 쿼리만 사용

---

## 해결 방안 옵션

### Option A: LLM 분리 (권장) ⭐

**개념:** Tools를 사용하는 노드에 non-reasoning LLM 전용 할당

**장점:**
- 최소한의 코드 변경
- 기존 에이전트 로직 유지
- 명확한 역할 분리

**변경 사항:**
1. `workflow.py`에서 LLM 할당 명확화
2. Intent/Query 노드에는 `use_reasoning=False` LLM 사용
3. Answer 노드에만 Reasoning LLM 사용 (현재 구조 유지)

```python
# workflow.py 변경
def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,                    # Tools용 (non-reasoning)
    *,
    reasoning_llm: Runnable | None = None,  # 답변 생성용
    ...
):
    # Intent classification과 query analysis에 사용할 LLM (tools 호환)
    classification_llm = llm  # non-reasoning 필수

    # Answer generation 전용 LLM (reasoning 가능)
    answer_llm = reasoning_llm or llm
```

**서버 초기화 변경 (`server.py` 또는 `handler.py`):**
```python
# Tools용 LLM (reasoning 비활성)
tools_llm = get_chat_model(use_reasoning=False)

# 답변 생성용 LLM (reasoning 활성)
reasoning_llm = get_chat_model(use_reasoning=True, reasoning_effort="medium")

graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=tools_llm,
    reasoning_llm=reasoning_llm,
)
```

---

### Option B: Structured Output 리팩토링

**개념:** `create_agent(tools=[...])` 대신 `with_structured_output()` 패턴 사용

**장점:**
- LangChain 권장 패턴
- 더 간단한 코드 구조
- Reasoning LLM과 호환 가능 (단, `reasoning_effort="none"` 필요)

**참고:** 프로젝트의 `multi_query_retriever.py`에서 이미 이 패턴 사용 중

**변경 사항:**

```python
# intent_classifier.py 변경
def create_intent_classifier(llm: Runnable) -> Runnable:
    prompt_template = get_prompt("intent_classification")
    system_prompt = prompt_template.messages[0].prompt.template

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
    ])

    # with_structured_output 사용
    structured_llm = llm.with_structured_output(IntentClassification)

    return prompt | structured_llm
```

**주의사항:**
- `with_structured_output()`도 내부적으로 tool/function calling 사용
- CLOVA에서 사용하려면 `reasoning_effort="none"` 필수
- 결국 Option A와 동일한 LLM 분리 필요

---

### Option C: PydanticOutputParser 사용

**개념:** Tools 없이 JSON 출력 요청 후 파싱

**장점:**
- 어떤 LLM과도 호환
- Tool/Function Calling 미지원 모델에서도 동작

**단점:**
- JSON 파싱 실패 가능성
- 출력 형식 일관성 낮음
- 추가 프롬프트 엔지니어링 필요

**변경 사항:**

```python
from langchain_core.output_parsers import PydanticOutputParser

def create_intent_classifier(llm: Runnable) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=IntentClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n" + parser.get_format_instructions()),
        ("user", "{input}"),
    ])

    return prompt | llm | parser
```

---

## 권장 해결 방안

### Option A 선택 이유

1. **최소 변경:** 기존 에이전트 로직 변경 불필요
2. **명확한 분리:** LLM 역할이 명확해짐
3. **안전성:** 테스트에서 이미 이 패턴 사용 중
4. **호환성:** 향후 CLOVA API 업데이트에도 안전

### 구현 계획

#### Phase 1: LLM 팩토리 함수 명확화

**파일:** `config/llm.py`

```python
def get_tools_llm(settings_obj: "Settings | None" = None, **kwargs) -> ChatClovaX:
    """
    Tools/Function Calling용 LLM을 반환합니다.

    Note: CLOVA HCX-007은 tools와 reasoning을 동시에 지원하지 않으므로,
    이 함수는 항상 reasoning 비활성화된 LLM을 반환합니다.
    """
    # reasoning 관련 파라미터 강제 제거
    kwargs.pop("use_reasoning", None)
    kwargs.pop("reasoning_effort", None)
    kwargs.pop("thinking", None)

    return get_chat_model(settings_obj, **kwargs)


def get_reasoning_llm(
    settings_obj: "Settings | None" = None,
    effort: str = "medium",
    **kwargs
) -> ChatClovaX:
    """
    Reasoning 모드 LLM을 반환합니다.

    Note: Tools/Function Calling과 함께 사용할 수 없습니다.
    """
    return get_chat_model(
        settings_obj,
        use_reasoning=True,
        reasoning_effort=effort,
        **kwargs
    )
```

#### Phase 2: Workflow 초기화 업데이트

**파일:** `slack/handler.py`

```python
def _build_chatbot_graph() -> CompiledStateGraph:
    """챗봇 그래프를 빌드합니다."""
    from naver_connect_chatbot.config.llm import get_tools_llm, get_reasoning_llm

    # Tools용 LLM (Intent Classification, Query Analysis)
    tools_llm = get_tools_llm()

    # Reasoning용 LLM (Answer Generation)
    reasoning_llm = get_reasoning_llm(effort="medium")

    graph = build_adaptive_rag_graph(
        retriever=retriever,
        llm=tools_llm,
        reasoning_llm=reasoning_llm,
        debug=settings.adaptive_rag.debug,
    )
    return graph
```

#### Phase 3: 테스트 fixture 업데이트

**파일:** `tests/test_adaptive_rag_integration.py`

```python
@pytest.fixture
def tools_llm():
    """Tools용 LLM (non-reasoning)"""
    from naver_connect_chatbot.config.llm import get_tools_llm

    try:
        return get_tools_llm()
    except ValueError:
        pytest.skip("LLM 설정 없음")


@pytest.fixture
def reasoning_llm():
    """Reasoning용 LLM"""
    from naver_connect_chatbot.config.llm import get_reasoning_llm

    try:
        return get_reasoning_llm(effort="medium")
    except ValueError:
        pytest.skip("LLM 설정 없음")
```

#### Phase 4: 문서화

1. `CLAUDE.md`에 LLM 사용 가이드라인 추가
2. `config/llm.py`에 docstring 보강

---

## 검증 계획

### 1. 단위 테스트

```python
def test_intent_classifier_with_tools_llm():
    """Tools LLM으로 Intent Classification 성공 확인"""
    llm = get_tools_llm()
    classifier = create_intent_classifier(llm)

    result = classifier.invoke({
        "messages": [{"role": "user", "content": "PyTorch란 무엇인가요?"}]
    })

    assert result is not None
    # 에러가 발생하지 않으면 성공


def test_intent_classifier_with_reasoning_llm_fails():
    """Reasoning LLM으로 Intent Classification 시 에러 확인"""
    llm = get_reasoning_llm()
    classifier = create_intent_classifier(llm)

    with pytest.raises(Exception):  # API 에러 예상
        classifier.invoke({
            "messages": [{"role": "user", "content": "PyTorch란 무엇인가요?"}]
        })
```

### 2. 통합 테스트

```python
@pytest.mark.asyncio
async def test_full_workflow_with_proper_llm_separation(
    hybrid_retriever, tools_llm, reasoning_llm
):
    """LLM 분리된 전체 워크플로 테스트"""
    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=tools_llm,
        reasoning_llm=reasoning_llm,
    )

    result = await graph.ainvoke({
        "question": "CV 강의에서 ResNet 설명해주세요",
    })

    # Intent가 기본값이 아닌 실제 분류 결과인지 확인
    assert result["intent"] != "SIMPLE_QA" or result["intent_confidence"] > 0.5
    assert "error" not in result.get("intent_reasoning", "").lower()
```

---

## 타임라인

| 단계 | 작업 | 예상 소요 |
|------|------|----------|
| 1 | LLM 팩토리 함수 추가 | 30분 |
| 2 | Workflow 초기화 업데이트 | 30분 |
| 3 | 테스트 fixture 업데이트 | 30분 |
| 4 | 단위/통합 테스트 작성 | 1시간 |
| 5 | 문서화 | 30분 |
| **총계** | | **~3시간** |

---

## 위험 요소 및 대응

### 1. 기존 테스트 호환성
- **위험:** 기존 테스트가 `llm()` fixture 사용
- **대응:** `llm()` fixture를 `get_tools_llm()` 호출로 변경

### 2. 프로덕션 환경 설정
- **위험:** `.env`에 `CLOVA_THINKING_EFFORT` 설정된 경우
- **대응:** `get_tools_llm()`에서 강제로 reasoning 비활성화

### 3. 성능 영향
- **위험:** 두 개의 LLM 인스턴스 사용으로 메모리 증가
- **대응:** ChatClovaX는 stateless이므로 영향 미미

---

## 참고 자료

- [LangChain Structured Output](https://python.langchain.com/docs/concepts/structured_outputs/)
- [CLOVA Studio HCX-007 문서](https://www.ncloud.com/product/aiService/clovaStudio)
- 프로젝트 내 참고: `rag/retriever/multi_query_retriever.py` (with_structured_output 패턴)
