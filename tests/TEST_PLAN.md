# Gemini 마이그레이션 통합 테스트 계획

## 배경

ChatClovaX → ChatGoogleGenerativeAI(Gemini 3.1 Pro) 교체 후 검증 계획.
VectorDB(Qdrant)는 테스트 환경에서 사용 불가 → Mock 기반 전략 필수.

---

## Phase A: 사전 수정 (Critical Fixes)

테스트 실행 전 반드시 수정해야 하는 항목:

### A-1. `_extract_text_response()` Gemini thinking block 처리
- **파일**: `service/graph/nodes.py:159-185`
- **문제**: Gemini thinking_level=high일 때 `AIMessage.content`가 리스트 형태
  (`[{"type": "thinking", ...}, {"type": "text", ...}]`)로 반환됨.
  현재 `str(response.content)` 호출 → 깨진 텍스트 출력
- **수정**: 리스트일 때 `type="text"` 블록만 추출하는 로직 추가

### A-2. `response_parser.py` typing 버그
- **파일**: `service/agents/response_parser.py:9`
- **문제**: `from typing import ..., list` — 소문자 `list`는 typing 모듈에서 import 불가
- **수정**: `list` 제거 (Python 3.13에서는 builtin `list` 직접 사용)

### A-3. `test_clova_model.py` collection 에러 방지
- **파일**: `tests/test_clova_model.py`
- **문제**: 모듈 레벨에서 `ChatClovaX()` 인스턴스 생성 → API 키 없으면 pytest collection 실패
- **수정**: `if __name__ == "__main__":` 가드 추가 또는 `conftest.py`에서 ignore 설정

### A-4. Google Search grounding `bind_tools` 패턴 검증
- **파일**: `service/graph/nodes.py:631`
- **문제**: `bind_tools([{"google_search": {}}])` — langchain-google-genai에서 raw dict 허용 여부 불명확
- **수정**: 실제 API 호출로 검증 후, 필요시 공식 패턴으로 교체

---

## Phase B: 순수 단위 테스트 (API/VectorDB 불필요)

### B-1. Config & Settings 테스트 (`tests/test_gemini_config.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| B-1-1 | `GeminiLLMSettings` 기본값 검증 (model, temperature, max_output_tokens) | Critical |
| B-1-2 | `GeminiLLMSettings` 환경변수 로드 (GOOGLE_API_KEY, GEMINI_MODEL) | Critical |
| B-1-3 | `get_chat_model()` → ChatGoogleGenerativeAI 인스턴스 반환 | Critical |
| B-1-4 | `get_chat_model(thinking_level="low")` vs 기본값 설정 차이 | Critical |
| B-1-5 | 하위 호환 kwargs: `reasoning_effort` → `thinking_level` 매핑 | Medium |
| B-1-6 | `use_reasoning=True` 단독 사용 시 동작 | Medium |

### B-2. 워크플로 라우팅 테스트 (`tests/test_workflow_routing.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| B-2-1 | `route_after_intent`: Hard OOD (relevance < 0.2) → `generate_ood_response` | High |
| B-2-2 | `route_after_intent`: Soft OOD (0.2 ≤ relevance < 0.5) → `analyze_query` | High |
| B-2-3 | `route_after_intent`: In-domain (relevance ≥ 0.5) → `analyze_query` | High |
| B-2-4 | `route_after_intent`: intent != OUT_OF_DOMAIN → always `analyze_query` | High |
| B-2-5 | `should_clarify`: 기존 테스트와 일관성 확인 | Medium |

### B-3. 노드 유틸리티 테스트 (`tests/test_node_utilities.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| B-3-1 | `_build_document_label()`: PDF 메타데이터 → `[강의자료: CV 이론/3강]` | Medium |
| B-3-2 | `_build_document_label()`: Slack QA 메타데이터 | Medium |
| B-3-3 | `_build_document_label()`: 메타데이터 없음 → `[문서 N]` 폴백 | Medium |
| B-3-4 | `_format_chat_history()`: 턴 번호 포함, 500자 truncation | Medium |
| B-3-5 | `_format_chat_history()`: 빈 메시지 → 빈 문자열 | Medium |
| B-3-6 | `_format_chat_history()`: max_turns 제한 | Medium |
| B-3-7 | `_extract_text_response()`: string content → 그대로 반환 | High |
| B-3-8 | `_extract_text_response()`: list content (thinking blocks) → text만 추출 | High |
| B-3-9 | `_extract_text_response()`: dict response → output/content 추출 | High |

### B-4. OOD 패턴 매칭 테스트 (`tests/test_ood_patterns.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| B-4-1 | greeting 패턴: "안녕하세요" → OUT_OF_DOMAIN, relevance=0.0 | High |
| B-4-2 | self_intro 패턴: "넌 누구야" → OUT_OF_DOMAIN | High |
| B-4-3 | chitchat 패턴: "심심해" → OUT_OF_DOMAIN | High |
| B-4-4 | off_topic 정밀화: "점심 메뉴" → OOD, "아침 학습" → NOT OOD | High |
| B-4-5 | 기술 질문: "Python 데코레이터" → NOT pattern-matched OOD | High |
| B-4-6 | `generate_ood_response_node`: self_intro → 자기소개 응답 | High |
| B-4-7 | `generate_ood_response_node`: greeting → 인사 응답 | High |

### B-5. Agent Mock 테스트 (`tests/test_agents_mock.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| B-5-1 | `classify_intent()`: Mock LLM → IntentClassification 반환 | Critical |
| B-5-2 | `classify_intent()`: LLM Exception → 안전 fallback (SIMPLE_QA) | Critical |
| B-5-3 | `analyze_query()`: Mock LLM → QueryAnalysis 반환 | Critical |
| B-5-4 | `analyze_query()`: LLM Exception → 안전 fallback | Critical |
| B-5-5 | `aclassify_intent()`: async 버전 동일 동작 확인 | High |
| B-5-6 | `aanalyze_query()`: async 버전 동일 동작 확인 | High |

---

## Phase C: Gemini API 연동 테스트 (VectorDB 불필요, API 키 필요)

### Mock Retriever 필요

```python
class MockRetriever(BaseRetriever):
    """VectorDB 없이 사전 정의된 문서를 반환하는 Mock Retriever."""
    documents: list[Document] = []

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.documents
```

### C-1. Agent Gemini 연동 (`tests/test_agents_gemini.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| C-1-1 | `classify_intent()` + 실제 Gemini: 기술 질문 → SIMPLE_QA | Critical |
| C-1-2 | `classify_intent()` + 실제 Gemini: "오늘 날씨" → OUT_OF_DOMAIN | Critical |
| C-1-3 | `analyze_query()` + 실제 Gemini: 쿼리 확장 + 필터 추출 | Critical |
| C-1-4 | `with_structured_output()` 반환 타입 = Pydantic 인스턴스 확인 | Critical |
| C-1-5 | thinking_level=low vs default 모두 structured output 정상 동작 | High |

### C-2. 워크플로 End-to-End (`tests/test_workflow_gemini.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| C-2-1 | Happy path: In-domain 질문 → 전체 워크플로 → answer 반환 | Critical |
| C-2-2 | Hard OOD path: "오늘 날씨" → OOD 응답 → finalize | High |
| C-2-3 | Soft OOD path: "Python 데코레이터" → analyze_query → 정상 답변 | High |
| C-2-4 | Post-retrieval OOD: 빈 문서 + low relevance → soft decline | High |
| C-2-5 | Multi-turn: 같은 thread에서 후속 질문 → 중복 답변 없음 | High |
| C-2-6 | Google Search grounding: RAG 문서 < 2 → web search 활성화 | High |

### C-3. Handler 초기화 (`tests/test_handler_init.py`)

| TC | 설명 | 우선순위 |
|----|------|---------|
| C-3-1 | `get_agent_app()` + MockRetriever → 그래프 컴파일 성공 | Critical |
| C-3-2 | 두 LLM 인스턴스 (classification/answer) 분리 확인 | High |

---

## Phase D: Evaluation Framework 업데이트

### D-1. LLM Judge 마이그레이션
- `evaluators/llm_judge.py`에서 `HCX-007` → Gemini 교체 검토
- 또는 Gemini를 Judge LLM으로 사용하는 새 evaluator 작성

---

## Teammate 분업 계획

### Teammate 1: "test-fixer" — Phase A (사전 수정)
- A-1: `_extract_text_response()` Gemini thinking block 처리
- A-2: `response_parser.py` typing 버그 수정
- A-3: `test_clova_model.py` collection 에러 방지
- A-4: Google Search grounding 패턴 검증

### Teammate 2: "unit-tester" — Phase B (순수 단위 테스트)
- B-1 ~ B-5 테스트 파일 작성
- Mock fixtures 공통 모듈 (`tests/conftest_gemini.py`) 작성
- 총 ~30개 테스트 케이스

### Teammate 3: "integration-tester" — Phase C (Gemini API 연동)
- MockRetriever 구현
- C-1 ~ C-3 테스트 파일 작성
- `@pytest.mark.integration` 마커 적용
- 총 ~13개 테스트 케이스

---

## 검증 기준

- [ ] Phase A 수정 후 `pytest -k "not integration" --ignore=tests/test_clova_model.py` 전체 통과
- [ ] Phase B 단위 테스트 30개+ 모두 통과 (API 불필요)
- [ ] Phase C 통합 테스트 13개+ 모두 통과 (Gemini API 필요)
- [ ] 기존 91개 단위 테스트 regression 없음
