# Q&A 평가 데이터셋 확장 설계서

**작성일**: 2025-12-09
**버전**: 2.0.0
**상태**: 승인됨

---

## 1. 개요

### 1.1 목적
Naver Connect Boost Camp RAG 챗봇의 Q&A 평가 데이터셋을 27개에서 80개로 확장하고, LLM-as-Judge 기반 자동 평가 시스템을 구축합니다.

### 1.2 목표
1. **커버리지 확대**: 누락된 과정들(level3_*, Object Detection, Generative AI 등) 포함
2. **깊이 강화**: 기존 카테고리 내 다양한 난이도/유형 추가
3. **Edge Case 중심**: RAG 약점 테스트 (multi-hop, cross-course 질문 등)

### 1.3 주요 결정사항
| 항목 | 결정 |
|------|------|
| 목표 규모 | 80개 질문 |
| 새 카테고리 | 5개 (multi_hop, temporal, negation, code_execution, meta_question) |
| 평가 방식 | Full LLM-as-Judge (HyperClovaX HCX-007) |

---

## 2. 질문 분배 계획

### 2.1 카테고리별 분배 (총 80개)

#### In-Domain (50개)
| 서브카테고리 | 현재 | 목표 | 추가 |
|-------------|------|------|------|
| concept | 4 | 10 | +6 |
| implementation | 4 | 10 | +6 |
| troubleshooting | 2 | 6 | +4 |
| comparison | 2 | 6 | +4 |
| course_specific | 3 | 10 | +7 |
| source_specific | 3 | 8 | +5 |

#### Out-of-Domain (15개)
| 서브카테고리 | 현재 | 목표 | 추가 |
|-------------|------|------|------|
| unrelated | 2 | 4 | +2 |
| ambiguous | 3 | 4 | +1 |
| hallucination_inducing | 2 | 4 | +2 |
| boundary | 2 | 3 | +1 |

#### Edge Case (15개) - 신규
| 서브카테고리 | 목표 | 설명 |
|-------------|------|------|
| multi_hop | 4 | 여러 문서 조합 필요 |
| temporal | 3 | 순서/시간 관련 |
| negation | 3 | 부정/제외 조건 |
| code_execution | 2 | 코드 결과 질문 |
| meta_question | 3 | 자료 메타데이터 질문 |

### 2.2 과정별 커버리지

#### Tier 1: 고빈도 과정 (문서 500+개) → 각 5-6개 질문
| 과정 | 문서 수 | 질문 목표 |
|------|--------|----------|
| level2_cv | 1,163 | 6 |
| level3_common | 768 | 5 |
| level2_nlp | 484 | 5 |
| core_common | 469 | 4 |

#### Tier 2: 중빈도 과정 (100-500개) → 각 3-4개 질문
| 과정 | 문서 수 | 질문 목표 |
|------|--------|----------|
| level3_product_serving | 427 | 4 |
| level2_recsys | 253 | 4 |
| NLP 강의 | 191 | 3 |
| MLforRecSys | 122 | 3 |
| AI Math | 109 | 3 |
| RecSys 이론 | 102 | 3 |

#### Tier 3: 저빈도 과정 (100개 미만) → 각 1-2개 질문
| 과정 | 문서 수 | 질문 목표 |
|------|--------|----------|
| Generative AI | 72 | 2 |
| MRC | 67+58 | 2 |
| Object Detection | 39+22 | 2 |
| level3_model_optimization | 36 | 2 |
| Segmentation | 23 | 1 |
| level2_klue | 20 | 1 |
| PyTorch | 24 | 2 |

---

## 3. LLM-as-Judge 평가 시스템

### 3.1 모델 설정
```python
from naver_connect_chatbot.config import get_chat_model

judge_llm = get_chat_model()  # HyperClovaX HCX-007
```

### 3.2 평가 차원

#### Faithfulness (충실성) - 1~5점
검색된 문서에 기반한 답변인지 평가
- 5점: 답변의 모든 정보가 검색 문서에서 직접 도출됨
- 4점: 대부분의 정보가 문서 기반이며, 일부 합리적 추론 포함
- 3점: 핵심 정보는 문서 기반이나, 일부 불확실한 내용 포함
- 2점: 문서와 관련 있으나 상당 부분이 문서에 없는 내용
- 1점: 문서와 무관하거나 명백한 환각 포함

#### Relevance (관련성) - 1~5점
사용자 질문에 적절히 답변했는지 평가
- 5점: 질문의 핵심을 정확히 파악하고 완벽히 답변
- 4점: 질문에 잘 답변했으나 약간의 부가 정보 부족
- 3점: 질문과 관련된 답변이나 핵심을 놓침
- 2점: 질문과 부분적으로만 관련된 답변
- 1점: 질문과 무관한 답변 또는 질문 오해

#### Completeness (완전성) - 1~5점
답변이 충분히 상세한지 평가
- 5점: 필요한 모든 정보를 포함한 완전한 답변
- 4점: 대부분의 정보 포함, 사소한 세부사항 누락
- 3점: 기본적인 답변이나 중요한 세부사항 누락
- 2점: 불완전하여 추가 질문이 필요한 답변
- 1점: 매우 불완전하거나 답변 거부

#### Hallucination Detection (환각 탐지)
다음 포함 시 환각으로 판정:
- 검색 문서에 없는 구체적인 수치, 날짜, 이름
- 존재하지 않는 강의, 마스터, 커리큘럼 언급
- 문서와 모순되는 정보
- 근거 없는 단정적 주장

### 3.3 특수 케이스 평가 기준

#### Out-of-Domain 질문 처리
- `politely_decline`: 범위 외 질문임을 정중히 안내해야 함
- `ask_clarification`: 모호한 질문에 명확화 요청해야 함
- `acknowledge_no_info`: 정보 없음을 솔직히 인정해야 함
- `acknowledge_limitation`: 한계를 인정하며 가능한 정보 제공

#### Edge Case 질문 처리
- `multi_doc_synthesis`: 여러 문서 정보를 종합해야 함
- `temporal_reasoning`: 시간/순서 관계를 올바르게 설명해야 함
- `negation_handling`: 제외 조건을 올바르게 처리해야 함
- `code_explanation`: 코드 동작을 정확히 설명해야 함
- `meta_info_retrieval`: 자료 메타정보를 정확히 제공해야 함

---

## 4. 데이터셋 스키마 v2.0

### 4.1 전체 구조
```json
{
  "version": "2.0.0",
  "created_at": "2025-12-09",
  "updated_at": "2025-12-09",
  "description": "Naver Connect Boost Camp RAG 평가 데이터셋 v2",
  "statistics": {
    "total_questions": 80,
    "by_category": {
      "in_domain": 50,
      "out_of_domain": 15,
      "edge_case": 15
    }
  },
  "categories": {...},
  "questions": [...]
}
```

### 4.2 개별 질문 스키마
```json
{
  "id": "edge_multi_001",
  "category": "edge_case",
  "subcategory": "multi_hop",
  "question": "CV에서 배운 ResNet의 skip connection 개념이 NLP의 Transformer에서는 어떻게 활용되나요?",

  "expected_filters": {
    "doc_type": ["lecture_transcript", "pdf"],
    "course": ["level2_cv", "level2_nlp", "NLP", "Computer Vision"]
  },

  "expected_topics": ["resnet", "skip connection", "transformer", "residual"],

  "ground_truth": {
    "answer_keywords": ["residual connection", "gradient flow", "layer normalization"],
    "should_have_context": true,
    "min_docs_required": 2,
    "expected_behavior": "multi_doc_synthesis"
  },

  "evaluation": {
    "requires_multi_doc": true,
    "cross_course": true,
    "reasoning_type": "comparative"
  },

  "metadata": {
    "difficulty": "hard",
    "requires_reasoning": true,
    "coverage": ["CV", "NLP"],
    "added_in_version": "2.0.0"
  }
}
```

---

## 5. 파일 구조

```
tests/evaluation/
├── README.md                      # 문서 (업데이트)
├── qa_dataset.json                # 기존 데이터셋 (v1 - 백업용)
├── qa_dataset_v2.json             # 확장 데이터셋 (v2 - 80개)
│
├── config/
│   └── evaluation_config.yaml     # 평가 설정
│
├── prompts/
│   ├── judge_system.yaml          # Judge 시스템 프롬프트
│   └── judge_user.yaml            # Judge 사용자 프롬프트 템플릿
│
├── evaluators/
│   ├── __init__.py
│   ├── base.py                    # BaseEvaluator 추상 클래스
│   ├── llm_judge.py               # LLM-as-Judge 구현
│   └── schemas.py                 # Pydantic 평가 스키마
│
├── test_rag_evaluation.py         # 기존 테스트 (v1 호환)
├── test_rag_evaluation_v2.py      # 확장 테스트 (v2)
│
└── reports/
    ├── .gitkeep
    └── evaluation_report_template.json
```

---

## 6. 구현 계획

### Phase 1: 기반 구조
1. 디렉토리 및 파일 구조 생성
2. 평가 스키마 정의 (Pydantic)
3. LLM-as-Judge 프롬프트 작성

### Phase 2: 평가기 구현
1. BaseEvaluator 추상 클래스
2. LLMJudgeEvaluator 구현
3. 프롬프트 로딩 유틸리티

### Phase 3: 데이터셋 확장
1. 기존 27개 질문 마이그레이션
2. 신규 53개 질문 생성
3. 과정별/카테고리별 균형 검증

### Phase 4: 테스트 코드
1. test_rag_evaluation_v2.py 작성
2. 카테고리별 테스트 함수
3. 리포트 생성 기능

### Phase 5: 검증
1. 샘플 질문으로 평가 파이프라인 테스트
2. 전체 데이터셋 실행
3. 결과 분석 및 문서화

---

## 7. 성공 기준

| 지표 | 목표 |
|------|------|
| In-Domain 평균 점수 | ≥ 0.65 |
| Edge Case behavior_correct | ≥ 80% |
| OOD 적절한 거부율 | ≥ 90% |
| 환각 탐지 정확도 | ≥ 85% |

---

## 부록: Judge 프롬프트 전문

### 시스템 프롬프트
```
당신은 네이버 부스트캠프 AI Tech 교육 챗봇의 RAG 시스템 평가 전문가입니다.

## 평가 기준

### 1. Faithfulness (충실성) - 1~5점
검색된 문서에 기반한 답변인지 평가합니다.
- 5점: 답변의 모든 정보가 검색 문서에서 직접 도출됨
- 4점: 대부분의 정보가 문서 기반이며, 일부 합리적 추론 포함
- 3점: 핵심 정보는 문서 기반이나, 일부 불확실한 내용 포함
- 2점: 문서와 관련 있으나 상당 부분이 문서에 없는 내용
- 1점: 문서와 무관하거나 명백한 환각(hallucination) 포함

### 2. Relevance (관련성) - 1~5점
사용자 질문에 적절히 답변했는지 평가합니다.
- 5점: 질문의 핵심을 정확히 파악하고 완벽히 답변
- 4점: 질문에 잘 답변했으나 약간의 부가 정보 부족
- 3점: 질문과 관련된 답변이나 핵심을 놓침
- 2점: 질문과 부분적으로만 관련된 답변
- 1점: 질문과 무관한 답변 또는 질문 오해

### 3. Completeness (완전성) - 1~5점
답변이 충분히 상세한지 평가합니다.
- 5점: 필요한 모든 정보를 포함한 완전한 답변
- 4점: 대부분의 정보 포함, 사소한 세부사항 누락
- 3점: 기본적인 답변이나 중요한 세부사항 누락
- 2점: 불완전하여 추가 질문이 필요한 답변
- 1점: 매우 불완전하거나 답변 거부

### 4. Hallucination Detection (환각 탐지)
답변에 다음이 포함되면 환각으로 판정합니다:
- 검색 문서에 없는 구체적인 수치, 날짜, 이름
- 존재하지 않는 강의, 마스터, 커리큘럼 언급
- 문서와 모순되는 정보
- 근거 없는 단정적 주장

## 특수 케이스 평가 기준

### Out-of-Domain 질문 처리
- "politely_decline": 범위 외 질문임을 정중히 안내해야 함
- "ask_clarification": 모호한 질문에 명확화 요청해야 함
- "acknowledge_no_info": 정보 없음을 솔직히 인정해야 함
- "acknowledge_limitation": 한계를 인정하며 가능한 정보 제공

### Edge Case 질문 처리
- "multi_doc_synthesis": 여러 문서 정보를 종합해야 함
- "temporal_reasoning": 시간/순서 관계를 올바르게 설명해야 함
- "negation_handling": 제외 조건을 올바르게 처리해야 함
- "code_explanation": 코드 동작을 정확히 설명해야 함
- "meta_info_retrieval": 자료 메타정보를 정확히 제공해야 함

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요:
{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "hallucination_detected": <true/false>,
  "behavior_correct": <true/false>,
  "reasoning": "<평가 근거를 2-3문장으로 설명>"
}
```
