# Slack QA LLM 품질 평가 시스템 설계

**작성일**: 2024-11-30
**목적**: RAG 시스템 품질 향상을 위한 Slack Q&A 데이터 필터링
**상태**: 설계 완료, 구현 대기

---

## 1. 문제 정의

### 1.1 현재 상황
- Slack Q&A 데이터 규모: ~953개 JSON 파일, ~4,300개 Q&A 쌍
- 기존 규칙 기반 필터링으로 명백한 저품질 제거 중
- 하지만 여전히 의미없는 데이터가 RAG 품질에 영향

### 1.2 해결해야 할 문제
| 문제 유형 | 설명 |
|----------|------|
| 중복/유사 질문 | 비슷한 질문이 반복되어 벡터DB에서 노이즈 |
| 낮은 품질 답변 | 현재 필터 통과하지만 실질적 도움 안 되는 답변 |
| 맥락 없는 대화 | 코드/에러 없이 추상적인 질문-답변 쌍 |
| 오래된/부정확한 정보 | 예전 기수 데이터로 현재와 맞지 않는 내용 |

### 1.3 선택한 접근 방식
**LLM 기반 다차원 품질 평가** → 3가지 기준으로 평가 후 점수화

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        처리 파이프라인                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1단계: 기존 파이프라인 - 규칙 기반 필터링]                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ original_documents/qa_dataset_from_slack/                    │   │
│  │        │                                                     │   │
│  │        ▼                                                     │   │
│  │ slack_qa_loader.py  → 시스템 메시지 제거                       │   │
│  │ (channel_join, channel_leave, pinned_item 등)                │   │
│  │        │                                                     │   │
│  │        ▼                                                     │   │
│  │ filter_qa_data.py   → 규칙 기반 필터링                        │   │
│  │ (필러 패턴, 단순 응답, 공지/모집, 짧은 감사 등)                   │   │
│  │        │                                                     │   │
│  │        ▼                                                     │   │
│  │ merge_qa_by_course.py → 과정별 병합                           │   │
│  │        │                                                     │   │
│  │        ▼                                                     │   │
│  │ document_chunks/slack_qa_merged/*.json (~3,000 Q&A 예상)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  [2단계: LLM 품질 평가 - NEW]                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  llm_quality_evaluator.py                                    │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │   │
│  │  │ 텍스트 추출   │ →  │ 배치 생성     │ →  │ LLM 평가      │   │   │
│  │  │ (메타 제거)   │    │ (10개씩)     │    │ (async)       │   │   │
│  │  └─────────────┘    └──────────────┘    └───────┬───────┘   │   │
│  │                                                 │           │   │
│  │                     ┌───────────────────────────┘           │   │
│  │                     ▼                                       │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ 체크포인트 저장 (50개마다)                             │   │   │
│  │  │ - 중단 후 재개 가능                                   │   │   │
│  │  │ - 진행률 표시                                         │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  [3단계: 결과 처리]                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ • overall_quality == "remove" → 제거                          │   │
│  │ • 나머지 → 점수를 메타데이터에 병합                              │   │
│  │ • 통계 리포트 생성                                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                    document_chunks/slack_qa_scored/*.json           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 평가 기준 (3가지 차원)

### 3.1 답변 완전성 (Completeness)

질문의 모든 부분에 충분히 답변했는가?

| 점수 | 기준 | 예시 |
|-----|------|------|
| 5 | 질문의 모든 부분에 완벽히 답변 + 추가 유용한 정보 제공 | "~하시면 됩니다. 추가로 ~도 고려해보세요" |
| 4 | 질문의 모든 부분에 충분히 답변 | 핵심 질문에 대한 명확한 답변 |
| 3 | 핵심 질문에는 답변했으나 일부 세부사항 누락 | 주요 답변은 있지만 부가 질문 미응답 |
| 2 | 부분적으로만 답변, 중요한 내용 누락 | 질문의 일부만 다룸 |
| 1 | 거의 답변이 없거나 질문과 무관 | "저도 궁금해요", 단순 공감 |

**특수 케이스:**
- 질문이 여러 개인 경우: 모든 질문에 답변했는지 확인
- "~해보세요"만 있고 구체적 방법 없음 → 최대 3점
- 링크만 제공: 링크의 관련성에 따라 2-4점

### 3.2 맥락 독립성 (Context Independence)

이 Q&A만 봐도 내용을 이해할 수 있는가? (외부 맥락 없이)

| 점수 | 기준 | 예시 |
|-----|------|------|
| 5 | 완전히 독립적, 배경 설명 포함 | 질문에 상황 설명 + 답변에 충분한 컨텍스트 |
| 4 | 대부분 이해 가능, 약간의 추론 필요 | 일반적인 지식으로 이해 가능 |
| 3 | 일부 외부 맥락 필요하지만 핵심은 이해 가능 | 특정 과제/강의 언급되나 일반화 가능 |
| 2 | 상당한 외부 맥락 필요 | "그 부분", "아까 말한 것" 등 지시어 많음 |
| 1 | 맥락 없이는 이해 불가능 | 이전 대화 참조, 특정 상황 전제 |

**특수 케이스:**
- "그거", "저기", "아까" 등 불명확한 지시어 → 감점
- 특정 과제명 언급: 내용이 일반화 가능하면 OK, 아니면 감점
- 이미지/파일 참조 ("첨부 참고"): 내용 없으면 1-2점
- 코드 에러 질문: 에러 메시지가 포함되어 있으면 OK

### 3.3 기술적 정확성 (Technical Accuracy)

코드, 개념, 설명이 기술적으로 정확한가?

| 점수 | 기준 | 예시 |
|-----|------|------|
| 5 | 완벽히 정확 + 모범 사례 반영 | 정확한 코드 + best practice 언급 |
| 4 | 정확함, 작동하는 솔루션 | 문법 오류 없는 코드, 정확한 개념 |
| 3 | 대체로 정확, 사소한 오류 있을 수 있음 | 작동하지만 개선 여지 있음 |
| 2 | 부분적으로 부정확, 오해의 소지 | 잘못된 설명이 섞여 있음 |
| 1 | 심각하게 부정확, 잘못된 정보 | 틀린 코드, 잘못된 개념 설명 |

**특수 케이스:**
- 코드가 없는 개념 질문: 개념의 정확성만 평가
- "~인 것 같아요" 추측성 답변: 확신도에 따라 감점
- 버전/환경 의존적 답변: 맞는 정보면 OK, 버전 명시 없으면 -1점
- 공식 문서 링크 제공: 링크가 유효하고 관련 있으면 4-5점

---

## 4. 종합 등급 판정

| 등급 | 조건 | 처리 |
|-----|------|------|
| **high** | 평균 ≥ 4.0 AND 최저점 ≥ 3 | 최우선 보존, 검색 가중치 ↑ |
| **medium** | 평균 ≥ 3.0 AND 최저점 ≥ 2 | 보존, 기본 가중치 |
| **low** | 평균 ≥ 2.0 OR 최저점 = 1이지만 가치 있음 | 보존, 검색 가중치 ↓ |
| **remove** | 평균 < 2.0 OR 2개 이상 차원이 1점 | 제거 대상 |

---

## 5. Pydantic 스키마

```python
from pydantic import BaseModel, Field
from typing import Literal

class DimensionScore(BaseModel):
    """개별 평가 차원의 점수와 근거"""
    score: Literal[1, 2, 3, 4, 5] = Field(description="1-5 점수")
    reasoning: str = Field(description="이 점수를 준 구체적인 근거 (1-2문장)")

class QualityEvaluation(BaseModel):
    """Q&A 품질 평가 결과"""

    completeness: DimensionScore = Field(
        description="답변 완전성: 질문의 모든 부분에 충분히 답변했는가"
    )
    context_independence: DimensionScore = Field(
        description="맥락 독립성: Q&A만 봐도 내용을 이해할 수 있는가"
    )
    technical_accuracy: DimensionScore = Field(
        description="기술적 정확성: 코드/개념/설명이 정확한가"
    )

    overall_quality: Literal["high", "medium", "low", "remove"] = Field(
        description="종합 품질 등급"
    )
    improvement_suggestion: str | None = Field(
        default=None,
        description="품질 개선을 위한 제안 (optional)"
    )
```

---

## 6. 입력 데이터 변환

### 6.1 변환 목적

LLM에 전송할 때 불필요한 메타데이터를 제거하여:
- 토큰 사용량 절감 (~70% 감소)
- 평가에 집중할 수 있는 깔끔한 입력 제공

### 6.2 변환 전/후 비교

**변환 전 (원본 JSON, ~285 토큰):**
```json
{
  "question": {
    "text": "transformer에서 positional encoding은 왜 필요한가요?",
    "user": "U123",
    "user_name": "학생A",
    "timestamp": "1699123456.789",
    "is_bot": false,
    "metadata": {"edited": null, "reactions": [{"name": "eyes", "count": 2}], "reply_count": 2}
  },
  "answers": [
    {"text": "RNN과 달리 transformer는 순서 정보가 없어서...", "user": "U456", ...},
    {"text": "추가로 sin/cos 함수를 쓰는 이유는...", "user": "U789", ...}
  ]
}
```

**변환 후 (LLM 입력, ~80 토큰):**
```markdown
## 질문
transformer에서 positional encoding은 왜 필요한가요?

## 답변들
[답변 1]
RNN과 달리 transformer는 순서 정보가 없어서...

[답변 2]
추가로 sin/cos 함수를 쓰는 이유는...
```

### 6.3 변환 코드

```python
from dataclasses import dataclass

@dataclass
class EvaluationInput:
    """LLM 평가를 위한 최소화된 입력"""
    question: str
    answers: list[str]
    original_id: str  # timestamp 기반 고유 ID (결과 매핑용)
    source_file: str  # 원본 파일명

    def to_prompt_format(self) -> str:
        """LLM 프롬프트에 삽입할 포맷으로 변환"""
        answers_formatted = "\n\n".join(
            f"[답변 {i+1}]\n{answer}"
            for i, answer in enumerate(self.answers)
        )

        return f"""## 질문
{self.question}

## 답변들
{answers_formatted}"""


def extract_for_evaluation(qa_pair: dict) -> EvaluationInput:
    """원본 QA 데이터에서 평가에 필요한 부분만 추출"""
    question_text = qa_pair["question"]["text"].strip()
    answer_texts = [
        answer["text"].strip()
        for answer in qa_pair["answers"]
        if answer["text"].strip()
    ]
    original_id = qa_pair["question"]["timestamp"]

    return EvaluationInput(
        question=question_text,
        answers=answer_texts,
        original_id=original_id,
        source_file=""
    )
```

---

## 7. 배치 처리 전략

### 7.1 설정

```python
@dataclass
class BatchConfig:
    """배치 처리 설정"""
    batch_size: int = 10           # 동시 처리 개수
    max_retries: int = 3           # API 실패 시 재시도 횟수
    retry_delay: float = 1.0       # 재시도 대기 시간 (초)
    checkpoint_interval: int = 50  # 체크포인트 저장 간격
```

### 7.2 체크포인트 (중단 후 재개)

```python
@dataclass
class ProcessingState:
    """처리 상태"""
    processed_ids: set[str]
    results: dict[str, dict]
    errors: list[dict]

    def save(self, path: Path) -> None:
        """체크포인트 저장"""
        ...

    @classmethod
    def load(cls, path: Path) -> 'ProcessingState':
        """체크포인트 로드"""
        ...
```

### 7.3 비동기 배치 처리

```python
async def process_batch(
    qa_items: list[EvaluationInput],
    evaluator: 'QualityEvaluator',
    config: BatchConfig,
) -> list[tuple[str, QualityEvaluation | None]]:
    """배치 단위로 평가 실행"""

    async def evaluate_with_retry(item: EvaluationInput):
        for attempt in range(config.max_retries):
            try:
                result = await evaluator.evaluate(item)
                return (item.original_id, result)
            except Exception as e:
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(config.retry_delay * (attempt + 1))
                else:
                    return (item.original_id, None)

    tasks = [evaluate_with_retry(item) for item in qa_items]
    return await asyncio.gather(*tasks)
```

---

## 8. 비용 추정

### 8.1 데이터 규모

| 항목 | 값 |
|-----|-----|
| 원본 JSON 파일 수 | 953개 |
| 추정 원본 Q&A 쌍 | ~4,300개 |
| 규칙 필터링 후 (예상) | ~3,000개 |
| 데이터 용량 | 26MB |

### 8.2 토큰 사용량 추정

```
• Q&A당 평균 토큰: ~200 (input) + ~150 (output)
• 시스템 프롬프트: ~1,500 토큰 (1회만)

입력 토큰: 3,000 × 200 = 600,000 tokens
출력 토큰: 3,000 × 150 = 450,000 tokens
```

### 8.3 비용 비교

| 모델 | 입력 비용 | 출력 비용 | 총 비용 | 비고 |
|------|----------|----------|---------|------|
| GPT-4o-mini | $0.09 | $0.27 | **~$0.36** | 추천 (가성비) |
| GPT-4o | $1.50 | $4.50 | ~$6.00 | 더 정확 |
| Claude 3.5 Haiku | $0.15 | $0.30 | ~$0.45 | 대안 |

**권장**: GPT-4o-mini로 시작, 결과 확인 후 필요시 GPT-4o로 재평가

---

## 9. 출력 파일 구조

```json
{
  "course": "level2_nlp",
  "metadata": {
    "total_qa_pairs": 450,
    "quality_filtered": true,
    "evaluation_model": "gpt-4o-mini",
    "evaluation_date": "2024-11-30",
    "quality_stats": {
      "high": 120,
      "medium": 280,
      "low": 50,
      "removed": 85
    },
    "avg_scores": {
      "completeness": 3.8,
      "context_independence": 3.5,
      "technical_accuracy": 4.1
    }
  },
  "qa_pairs": [
    {
      "generation": "7",
      "date": "2024-10-15",
      "question": {...},
      "answers": [...],
      "quality_score": {
        "completeness": {"score": 4, "reasoning": "..."},
        "context_independence": {"score": 3, "reasoning": "..."},
        "technical_accuracy": {"score": 5, "reasoning": "..."},
        "overall_quality": "high",
        "avg_score": 4.0
      }
    }
  ]
}
```

---

## 10. 구현 체크리스트

- [ ] `EvaluationInput` 데이터클래스 구현
- [ ] `QualityEvaluation` Pydantic 모델 구현
- [ ] LLM 평가 프롬프트 템플릿 작성 (Jinja2)
- [ ] `QualityEvaluator` 클래스 구현 (with_structured_output)
- [ ] 배치 처리 로직 구현 (asyncio)
- [ ] 체크포인트 저장/로드 기능
- [ ] 결과 병합 및 통계 생성
- [ ] CLI 스크립트 작성
- [ ] 테스트 (샘플 데이터로 검증)

---

## 11. 향후 확장 가능성

1. **중복/유사 질문 클러스터링**: 임베딩 기반 유사도로 대표 Q&A 선정
2. **날짜 기반 가중치**: 최신 기수 데이터에 더 높은 가중치
3. **검색 시 품질 점수 활용**: Qdrant 메타데이터 필터링 or 가중치
4. **자동 답변 개선**: 낮은 품질 답변을 LLM으로 보완

---

*설계 완료: 2024-11-30*
