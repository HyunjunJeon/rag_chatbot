# RAG 시스템 개선 작업 진행 상황

> 마지막 업데이트: 2025-12-11

---

## 완료된 작업

### Phase 1: CLOVA API 호환성 및 OOD 감지 (2025-12-11)

#### 1.1 CLOVA API `parallel_tool_calls` 에러 해결 ✅

**문제 상황**:
- `with_structured_output()` 호출 시 CLOVA HCX-007 API가 `parallel_tool_calls` 파라미터를 지원하지 않아 에러 발생
- 모든 Intent Classification 및 Query Analysis가 실패하고 fallback으로 `SIMPLE_QA` 반환
- **결과**: OOD 질문이 감지되지 않고 모두 일반 질문으로 처리됨

**해결책**:
- `_get_structured_llm()` 함수를 **PydanticOutputParser 기반**으로 전환
- LLM 응답에서 JSON을 추출하고 파싱하는 로직 구현
- 다양한 JSON 형식 (코드펜스 포함, 중첩 객체) 처리 지원

**수정된 파일**:
```
app/naver_connect_chatbot/service/agents/
├── intent_classifier.py   # _get_structured_llm() 재구현
└── query_analyzer.py      # _get_structured_llm() 재구현
```

**테스트 결과**:
```
OOD Detection Test: 4/5 passed
✅ "오늘 날씨 어때?" → OUT_OF_DOMAIN (domain_relevance: 0.00)
✅ "맛있는 점심 메뉴 추천해줘" → OUT_OF_DOMAIN (domain_relevance: 0.00)
⚠️ "그거 어떻게 해?" → OUT_OF_DOMAIN (안전한 처리)
✅ "Transformer의 Self-Attention이란?" → SIMPLE_QA (domain_relevance: 0.95)
✅ "PyTorch에서 DataLoader 사용법" → SIMPLE_QA (domain_relevance: 0.95)
```

#### 1.2 OOD (Out-of-Domain) 감지 시스템 (기구현 확인) ✅

이미 구현되어 있던 기능들:
- `intent_classification.yaml` v3.0: OUT_OF_DOMAIN 카테고리 + domain_relevance 점수
- `intent_classifier.py`: OUT_OF_DOMAIN 의도 + domain_relevance 필드
- `nodes.py`: `generate_ood_response_node` 노드
- `workflow.py`: `route_after_intent` OOD 라우팅

---

### Phase 2: 프롬프트 개선 (2025-12-11)

#### 2.1 Multi-Query 출력 형식 유연화 ✅

**수정 파일**: `multi_query_generation.yaml` (v1.0 → v1.1)

**변경 내용**:
- 엄격한 형식 요구 제거 ("No numbering, bullet points")
- 다양한 출력 형식 허용 (줄바꿈, 번호, 불릿)
- Self-Validation 섹션 추가
- 한국어 예시로 변경

```yaml
### 허용되는 형식 (아래 중 하나 선택):
1. **줄바꿈 구분** (권장)
2. **번호 형식**: 1. 쿼리, 2. 쿼리
3. **불릿 형식**: - 쿼리, - 쿼리
```

#### 2.2 환각 방지 강화 (기구현 확인) ✅

이미 v2.1에서 구현된 내용:
- 증거 분류 (Evidence Classification) - 4가지 유형
- 특화 규칙 (Simple/Complex/Exploratory별)
- 자가 검증 체크리스트
- 모순 정보 처리 (complex)

#### 2.3 Few-Shot 예시 추가 ✅

**수정 파일**: 3개 프롬프트 (v2.1 → v2.2)

| 프롬프트 | 예시 내용 |
|----------|----------|
| `answer_generation_simple.yaml` | PyTorch GPU 사용법 |
| `answer_generation_complex.yaml` | Self-Attention vs Cross-Attention 비교 |
| `answer_generation_exploratory.yaml` | 딥러닝 프로젝트 시작 가이드 |

#### 2.4 Edge Case Handling 추가 ✅

모든 answer_generation 프롬프트에 추가:

```yaml
## Edge Case Handling
### 1. 빈 컨텍스트 (Empty Context)
### 2. 무관한 컨텍스트 (Irrelevant Context)
### 3. 부분 관련 컨텍스트 (Partial Relevance)
### 4. 모순된 정보 (Complex만)
### 5. 매우 긴 컨텍스트 (Complex만)
```

---

## 수정된 파일 전체 목록

| 파일 | 버전 | 변경 내용 |
|------|------|----------|
| `service/agents/intent_classifier.py` | - | PydanticOutputParser로 CLOVA API 호환 |
| `service/agents/query_analyzer.py` | - | PydanticOutputParser로 CLOVA API 호환 |
| `prompts/templates/multi_query_generation.yaml` | v1.0→v1.1 | 유연한 출력 형식 + Self-Validation |
| `prompts/templates/answer_generation_simple.yaml` | v2.1→v2.2 | Few-Shot 예시 + Edge Case Handling |
| `prompts/templates/answer_generation_complex.yaml` | v2.1→v2.2 | Few-Shot 예시 + Edge Case Handling |
| `prompts/templates/answer_generation_exploratory.yaml` | v2.1→v2.2 | Few-Shot 예시 + Edge Case Handling |

---

## 다음 작업 (Phase 3)

`PROMPT_IMPROVEMENT_PLAN.md` P3 항목:

| 항목 | 설명 | 예상 공수 |
|------|------|-----------|
| Query Analysis 프롬프트 분리 | 품질 평가와 쿼리 확장 분리 | 3-4시간 |
| Self-Validation 추가 | 답변 전 자가 검증 강화 | 1-2시간 |
| 한국어 동의어 확장 | 검색 쿼리 확장 | 2-3시간 |

---

## CLOVA HCX-007 API 제한사항 참고

테스트 결과 발견된 CLOVA API 제한사항:

| 기능 | 지원 여부 | 대안 |
|------|----------|------|
| `with_structured_output(method='function_calling')` | ❌ `parallel_tool_calls` 에러 | PydanticOutputParser |
| `with_structured_output(method='json_mode')` | ❌ `response_format.type` 에러 | PydanticOutputParser |
| `with_structured_output(method='json_schema')` | ✅ 작동 | 사용 가능 |
| `tools` + `reasoning` 동시 사용 | ❌ 미지원 | 별도 호출 |

**권장 패턴**: 프롬프트에서 JSON 형식 요청 + PydanticOutputParser로 파싱

---

## 참고 문서

- [PROMPT_IMPROVEMENT_PLAN.md](./PROMPT_IMPROVEMENT_PLAN.md) - 프롬프트 개선 계획서
- [OOD_IMPROVEMENT_ANALYSIS.md](./OOD_IMPROVEMENT_ANALYSIS.md) - OOD 처리 분석
- [tests/evaluation/README.md](../tests/evaluation/README.md) - 평가 데이터셋 구조

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-11 | 2.0 | Phase 1 & 2 완료 - CLOVA API 호환, 프롬프트 개선 |
| 2025-12-09 | 1.0 | 초기 Pre-Retriever 데이터 소스 선택 기능 구현 |
