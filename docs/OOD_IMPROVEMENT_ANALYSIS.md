# Out-of-Domain 처리 개선 분석 및 구현 계획

> **문서 버전**: 1.0
> **작성일**: 2025-12-10
> **기반 데이터**: evaluation_report_v2_20251210_112808.json

---

## Executive Summary

RAG 평가에서 **Out-of-Domain (OOD) 질문의 통과율이 33% (5/15)**로 가장 낮게 나타났습니다.
주요 원인은 **RAG 시스템이 OOD 질문을 감지하지 못하고 답변을 시도**하는 것입니다.

### 핵심 문제

| 문제 | 원인 | 영향 |
|------|------|------|
| OOD 질문에 답변 시도 | Intent Classification이 OOD 감지 안함 | 33% 통과율 |
| 모호한 질문에 임의 답변 | Clarification 로직 미작동 | ambiguous 0% 통과 |
| pass_rate/overall_score null | @property 직렬화 안됨 | 테스트 assertion 실패 |

---

## 1. 문제 상세 분석

### 1.1 OOD 서브카테고리별 실패 분석

| 서브카테고리 | 통과/전체 | 통과율 | 실패 원인 |
|-------------|-----------|--------|-----------|
| **unrelated** | 1/4 | 25% | 관련 없는 질문에 우회 답변 제공 |
| **ambiguous** | 0/4 | **0%** | 명확화 요청 없이 임의 주제 답변 |
| **hallucination_inducing** | 2/4 | 50% | 존재하지 않는 정보 생성 |
| **boundary** | 2/3 | 67% | 경계 질문에서 한계 인정 부족 |

### 1.2 실패 사례 상세

#### Case 1: `ood_unrelated_001` - "오늘 날씨 어때?"

**문제**:
- 기대 행동: `politely_decline`
- 실제 행동: 날씨 API 구현 관련 문서 기반 답변 제공
- Judge 평가: `behavior_correct: false`

**Judge Reasoning**:
> "범위에 속하지 않는 질문에 대해 정중히 거절하지 않고 관련 정보를 제공한 점은 행동 측면에서 부적절하다."

#### Case 2: `ood_ambiguous_001` - "그거 어떻게 해?"

**문제**:
- 기대 행동: `ask_clarification`
- 실제 행동: 여러 시나리오별 임의 답변 제공
- Judge 평가: `behavior_correct: false`

**Judge Reasoning**:
> "명확화 요청('ask_clarification')을 우선하지 않았습니다."

#### Case 3: `ood_hallucination_001` - "부스트캠프 10기 NLP 커리큘럼 알려줘"

**문제**:
- 기대 행동: `acknowledge_no_info`
- 실제 행동: 일반적 NLP 내용으로 답변 생성
- Judge 평가: `behavior_correct: false`

**Judge Reasoning**:
> "10기 NLP 커리큘럼에 대한 구체적인 정보가 없음에도 불구하고 일반적 NLP 내용을 답변"

### 1.3 성공 사례 분석

#### Case: `ood_unrelated_004` - "주식 시장 어때?"

**성공 이유**:
- 시스템이 "주식 시장에 관한 정보가 없음"을 명확히 밝힘
- "문서 분석을 통해 AI/ML 관련 정보만 포함되어 있음"을 설명
- 범위 외임을 정중히 안내

**Judge Reasoning**:
> "범위 외임을 정중히 안내했고, 대체 접근법을 제시하며 도움을 제공하려 했습니다."

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### 2.1 RAG 워크플로우 흐름에서의 문제점

```
User Question: "오늘 날씨 어때?"
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 1. CLASSIFY INTENT                                      │
│    ❌ 문제: OOD 질문임에도 SIMPLE_QA로 분류             │
│    → Intent: SIMPLE_QA (confidence: 0.5)               │
│    → 에러로 인한 fallback 적용                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 2. ANALYZE QUERY                                        │
│    ❌ 문제: OOD 감지 로직 없음                          │
│    → 검색 쿼리 생성됨                                  │
│    → 필터 추출 시도됨                                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 3. RETRIEVE                                             │
│    ⚠️ 문제: 관련 없는 문서 검색됨                       │
│    → "날씨"라는 키워드로 날씨 API 문서 검색             │
│    → 14개 문서 반환 (모두 관련 없음)                   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 4. GENERATE ANSWER                                      │
│    ❌ 문제: OOD 거부 로직 없음                          │
│    → 검색된 문서 기반으로 답변 생성 시도                │
│    → 날씨 API 관련 정보 제공                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 핵심 누락 기능

1. **Intent Classification에서 OOD 감지 부재**
   - 현재: 4가지 의도만 분류 (SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, CLARIFICATION_NEEDED)
   - 필요: `OUT_OF_DOMAIN` 의도 추가

2. **Query Analysis에서 도메인 적합성 평가 부재**
   - 현재: clarity, specificity, searchability만 평가
   - 필요: `domain_relevance` 점수 추가

3. **Answer Generator에서 OOD 거부 로직 부재**
   - 현재: 무조건 답변 생성 시도
   - 필요: OOD 감지 시 정중한 거부 응답

4. **Retrieval 결과 품질 평가 부재**
   - 현재: 검색된 문서 수만 확인
   - 필요: 문서-질문 관련성 점수 기반 OOD 판단

---

## 3. 구현 계획

### 3.1 Option A: Intent Classification 확장 (권장)

**수정 파일**:
- `app/naver_connect_chatbot/service/agents/intent_classifier.py`
- `app/naver_connect_chatbot/prompts/templates/intent_classification.yaml`

**변경 사항**:

```python
# intent_classifier.py
class IntentClassification(BaseModel):
    intent: Literal[
        "SIMPLE_QA",
        "COMPLEX_REASONING",
        "EXPLORATORY",
        "CLARIFICATION_NEEDED",
        "OUT_OF_DOMAIN"  # 새로 추가
    ]
    confidence: float
    reasoning: str
    domain_relevance: float = 1.0  # 새로 추가: 도메인 관련성 (0.0~1.0)
```

**프롬프트 수정** (`intent_classification.yaml`):

```yaml
## Task
Classify the user's question into one of 5 categories for a Naver Boost Camp AI/ML educational chatbot.

## Categories
1. **SIMPLE_QA**: Direct factual questions about AI/ML concepts
2. **COMPLEX_REASONING**: Questions requiring multi-step analysis
3. **EXPLORATORY**: Open-ended learning/guidance questions
4. **CLARIFICATION_NEEDED**: Ambiguous questions needing more context
5. **OUT_OF_DOMAIN**: Questions unrelated to AI/ML education
   - Examples: 날씨, 음식, 여행, 개인적 질문
   - Should be politely declined

## Domain Relevance Scoring
Evaluate how relevant the question is to Naver Boost Camp's AI/ML curriculum:
- 1.0: Directly about AI/ML concepts, code, or course content
- 0.7-0.9: Related to programming/data science but not core curriculum
- 0.3-0.6: Tangentially related or ambiguous
- 0.0-0.2: Completely unrelated (OUT_OF_DOMAIN)

## Output Format
{
  "intent": "...",
  "confidence": 0.0-1.0,
  "domain_relevance": 0.0-1.0,
  "reasoning": "..."
}
```

**워크플로우 라우팅 변경** (`workflow.py`):

```python
def route_after_intent(state: AdaptiveRAGState) -> str:
    """Intent 분류 후 라우팅."""
    intent = state.get("intent", "SIMPLE_QA")
    domain_relevance = state.get("domain_relevance", 1.0)

    # OUT_OF_DOMAIN 처리
    if intent == "OUT_OF_DOMAIN" or domain_relevance < 0.3:
        return "generate_ood_response"  # 새 노드

    # 기존 로직
    if intent == "CLARIFICATION_NEEDED":
        return "clarify"

    return "analyze_query"
```

**새 노드 추가** (`nodes.py`):

```python
async def generate_ood_response_node(state: AdaptiveRAGState) -> dict:
    """Out-of-Domain 질문에 대한 정중한 거부 응답 생성."""
    question = state.get("question", "")

    response = (
        f"죄송합니다. '{question}'에 대해서는 답변드리기 어렵습니다.\n\n"
        "저는 네이버 부스트캠프 AI 교육 과정과 관련된 질문에 답변드릴 수 있습니다. "
        "예를 들어:\n"
        "- AI/ML 개념 설명 (Transformer, CNN, RecSys 등)\n"
        "- 코드 구현 방법\n"
        "- 강의 내용 관련 질문\n"
        "- 실습/과제 관련 질문\n\n"
        "관련된 질문이 있으시면 언제든 도움드리겠습니다!"
    )

    return {
        "answer": response,
        "generation_strategy": "ood_decline",
        "workflow_stage": "completed",
    }
```

### 3.2 Option B: Retrieval 결과 기반 OOD 감지

**수정 파일**:
- `app/naver_connect_chatbot/service/graph/nodes.py`

**변경 사항**:

```python
async def retrieve_node(state: AdaptiveRAGState, retriever) -> dict:
    """문서 검색 + OOD 감지."""
    # 기존 검색 로직
    docs = await retriever.ainvoke(query)

    # OOD 감지: 검색된 문서의 관련성 점수 평균 확인
    avg_score = sum(d.metadata.get("score", 0) for d in docs) / len(docs) if docs else 0

    # 관련성이 낮으면 OOD로 판단
    if avg_score < 0.5 and len(docs) < 3:
        return {
            "documents": [],
            "is_out_of_domain": True,
            "ood_reason": "검색된 문서의 관련성이 낮습니다.",
        }

    return {"documents": docs, "is_out_of_domain": False}
```

### 3.3 Option C: Answer Generator에서 OOD 처리

**수정 파일**:
- `app/naver_connect_chatbot/prompts/templates/answer_generation_*.yaml`

**프롬프트 추가**:

```yaml
## Out-of-Domain Detection
Before generating an answer, assess if the question is within scope:

1. **Check topic relevance**: Is this about AI/ML, programming, or Boost Camp content?
2. **Check context availability**: Do the provided documents contain relevant information?

If the question is OUT OF SCOPE:
- DO NOT attempt to answer using tangentially related information
- Respond with: "죄송합니다. [질문 주제]에 대해서는 답변드리기 어렵습니다..."
- Suggest what types of questions you CAN help with

If the question is AMBIGUOUS:
- Ask for clarification before providing an answer
- Example: "질문을 좀 더 구체적으로 해주시면 도움드리겠습니다. 어떤 부분에 대해 알고 싶으신가요?"
```

---

## 4. pass_rate/overall_score null 문제 해결

### 4.1 원인

Pydantic v2의 `model_dump()` 메서드는 **기본적으로 @property를 직렬화하지 않습니다**.

```python
# 현재 코드 (schemas.py)
class EvaluationReport(BaseModel):
    @property
    def pass_rate(self) -> float:
        return self.passed_questions / self.total_questions

    @property
    def overall_score(self) -> float:
        scores = [r.get("judge", {}).get("overall_score", 0) for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

# JSON 직렬화 시
report.model_dump()  # pass_rate, overall_score 포함 안됨!
```

### 4.2 해결 방안

**Option A: computed field 사용 (Pydantic v2 권장)**

```python
from pydantic import computed_field

class EvaluationReport(BaseModel):
    # 기존 필드들...

    @computed_field
    @property
    def pass_rate(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.passed_questions / self.total_questions

    @computed_field
    @property
    def overall_score(self) -> float:
        scores = [
            r.get("judge", {}).get("overall_score", 0)
            for r in self.results
            if r.get("judge")
        ]
        return sum(scores) / len(scores) if scores else 0.0
```

**Option B: 저장 시 명시적 계산**

```python
# test_rag_evaluation_v2.py의 리포트 저장 부분 수정
report_dict = report.model_dump()
report_dict["pass_rate"] = report.pass_rate
report_dict["overall_score"] = report.overall_score

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)
```

---

## 5. 구현 우선순위

| 순위 | 작업 | 예상 공수 | 영향도 |
|------|------|-----------|--------|
| 1 | pass_rate/overall_score 직렬화 수정 | 30분 | 🔴 높음 (테스트 통과) |
| 2 | Intent Classification에 OUT_OF_DOMAIN 추가 | 2시간 | 🔴 높음 (OOD 통과율) |
| 3 | OOD 응답 생성 노드 추가 | 1시간 | 🔴 높음 |
| 4 | Answer Generator에 OOD 프롬프트 추가 | 1시간 | 🟡 중간 |
| 5 | Retrieval 결과 기반 OOD 감지 | 1시간 | 🟡 중간 |

---

## 6. 테스트 계획

### 6.1 OOD 감지 테스트

```python
# tests/test_ood_detection.py
@pytest.mark.parametrize("question,expected_intent", [
    ("오늘 날씨 어때?", "OUT_OF_DOMAIN"),
    ("맛있는 점심 메뉴 추천해줘", "OUT_OF_DOMAIN"),
    ("Transformer의 Self-Attention이 뭐야?", "SIMPLE_QA"),
    ("그거 어떻게 해?", "CLARIFICATION_NEEDED"),
])
async def test_intent_classification_ood(question, expected_intent):
    result = await intent_classifier.aclassify_intent(question)
    assert result.intent == expected_intent
```

### 6.2 OOD 응답 품질 테스트

```python
@pytest.mark.parametrize("question", [
    "오늘 날씨 어때?",
    "주식 투자 어떻게 해?",
])
async def test_ood_response_quality(question, rag_graph):
    result = await rag_graph.ainvoke({"question": question})

    # OOD 응답에는 특정 문구가 포함되어야 함
    assert "답변드리기 어렵습니다" in result["answer"]
    assert "AI 교육" in result["answer"] or "부스트캠프" in result["answer"]
```

---

## 7. 예상 결과

### 개선 전 (현재)

| 카테고리 | 통과율 |
|----------|--------|
| In-Domain | 80% |
| **Out-of-Domain** | **33%** |
| Edge Case | 87% |

### 개선 후 (목표)

| 카테고리 | 목표 통과율 |
|----------|-------------|
| In-Domain | 85% |
| **Out-of-Domain** | **80%+** |
| Edge Case | 90% |

---

## 8. 참고: OOD 질문 유형별 기대 행동

| 유형 | 예시 | 기대 행동 |
|------|------|-----------|
| `unrelated` | "오늘 날씨 어때?" | `politely_decline` |
| `ambiguous` | "그거 어떻게 해?" | `ask_clarification` |
| `hallucination_inducing` | "10기 NLP 커리큘럼 알려줘" | `acknowledge_no_info` |
| `boundary` | "GPT-5에 대해 설명해줘" | `acknowledge_limitations` |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-10 | 1.0 | 초기 문서 작성 |
