# Slack QA LLM 품질 평가 시스템 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** LLM 기반으로 Slack Q&A 데이터의 품질을 평가하고, 저품질 데이터를 필터링하는 시스템 구축

**Architecture:** 기존 규칙 기반 필터링 파이프라인 이후에 LLM 평가 단계 추가. 3가지 차원(완전성, 맥락독립성, 기술정확성)으로 평가하고, 점수를 메타데이터로 저장하거나 낮은 점수는 제거.

**Tech Stack:** LangChain, langchain_naver (ChatClovaX HCX-007), Pydantic, asyncio

**LLM 사용 패턴:**
- `get_chat_model()` → Clova X HCX-007 모델 생성
- `create_structured_agent()` → Tool 기반 structured output
- `parse_agent_response()` → 응답 파싱 (fallback 지원)

**설계 문서:** `docs/plans/2024-11-30-slack-qa-llm-quality-evaluation-design.md`

---

## Task 1: Pydantic 스키마 정의

**Files:**
- Create: `document_processing/slack_qa/quality_schemas.py`
- Test: `tests/document_processing/test_quality_schemas.py`

**Step 1: 테스트 디렉토리 생성**

```bash
mkdir -p tests/document_processing
touch tests/document_processing/__init__.py
```

**Step 2: 실패하는 테스트 작성**

```python
# tests/document_processing/test_quality_schemas.py
"""품질 평가 스키마 테스트."""

import pytest
from document_processing.slack_qa.quality_schemas import (
    DimensionScore,
    QualityEvaluation,
    EvaluationInput,
)


class TestDimensionScore:
    """DimensionScore 모델 테스트."""

    def test_valid_score(self):
        """유효한 점수 생성."""
        score = DimensionScore(score=4, reasoning="Good answer with examples")
        assert score.score == 4
        assert score.reasoning == "Good answer with examples"

    def test_invalid_score_too_high(self):
        """6점은 유효하지 않음."""
        with pytest.raises(ValueError):
            DimensionScore(score=6, reasoning="Invalid")

    def test_invalid_score_too_low(self):
        """0점은 유효하지 않음."""
        with pytest.raises(ValueError):
            DimensionScore(score=0, reasoning="Invalid")


class TestQualityEvaluation:
    """QualityEvaluation 모델 테스트."""

    def test_valid_evaluation(self):
        """유효한 평가 결과 생성."""
        eval_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Complete"),
            context_independence=DimensionScore(score=3, reasoning="Some context needed"),
            technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
            overall_quality="high",
        )
        assert eval_result.overall_quality == "high"
        assert eval_result.avg_score == 4.0

    def test_invalid_overall_quality(self):
        """유효하지 않은 overall_quality."""
        with pytest.raises(ValueError):
            QualityEvaluation(
                completeness=DimensionScore(score=4, reasoning="Complete"),
                context_independence=DimensionScore(score=3, reasoning="Ok"),
                technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
                overall_quality="excellent",  # Invalid
            )


class TestEvaluationInput:
    """EvaluationInput 데이터클래스 테스트."""

    def test_to_prompt_format(self):
        """프롬프트 포맷 변환."""
        input_data = EvaluationInput(
            question="How to use PyTorch?",
            answers=["Use pip install pytorch", "Check the docs"],
            original_id="1234567890.123456",
            source_file="test.json",
        )
        result = input_data.to_prompt_format()

        assert "## 질문" in result
        assert "How to use PyTorch?" in result
        assert "[답변 1]" in result
        assert "[답변 2]" in result

    def test_empty_answers(self):
        """빈 답변 리스트."""
        input_data = EvaluationInput(
            question="Question?",
            answers=[],
            original_id="123",
            source_file="test.json",
        )
        result = input_data.to_prompt_format()
        assert "## 답변들" in result
```

**Step 3: 테스트 실행하여 실패 확인**

```bash
pytest tests/document_processing/test_quality_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'document_processing.slack_qa.quality_schemas'`

**Step 4: 스키마 구현**

```python
# document_processing/slack_qa/quality_schemas.py
"""Slack Q&A 품질 평가를 위한 Pydantic 스키마 및 데이터 클래스."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class DimensionScore(BaseModel):
    """개별 평가 차원의 점수와 근거."""

    score: int = Field(ge=1, le=5, description="1-5 점수")
    reasoning: str = Field(min_length=1, description="이 점수를 준 구체적인 근거")

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: int) -> int:
        """점수가 1-5 범위인지 검증."""
        if not 1 <= v <= 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v


class QualityEvaluation(BaseModel):
    """Q&A 품질 평가 결과."""

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
        description="품질 개선을 위한 제안 (optional)",
    )

    @computed_field
    @property
    def avg_score(self) -> float:
        """평균 점수 계산."""
        return round(
            (
                self.completeness.score
                + self.context_independence.score
                + self.technical_accuracy.score
            )
            / 3,
            2,
        )


@dataclass
class EvaluationInput:
    """LLM 평가를 위한 최소화된 입력."""

    question: str
    answers: list[str]
    original_id: str
    source_file: str

    def to_prompt_format(self) -> str:
        """LLM 프롬프트에 삽입할 포맷으로 변환."""
        if self.answers:
            answers_formatted = "\n\n".join(
                f"[답변 {i + 1}]\n{answer}" for i, answer in enumerate(self.answers)
            )
        else:
            answers_formatted = "(답변 없음)"

        return f"""## 질문
{self.question}

## 답변들
{answers_formatted}"""
```

**Step 5: 테스트 실행하여 통과 확인**

```bash
pytest tests/document_processing/test_quality_schemas.py -v
```

Expected: PASS (6 tests)

**Step 6: 커밋**

```bash
git add document_processing/slack_qa/quality_schemas.py tests/document_processing/
git commit -m "feat(slack_qa): add quality evaluation Pydantic schemas

- DimensionScore for individual dimension scoring (1-5)
- QualityEvaluation for complete evaluation result
- EvaluationInput dataclass for LLM input preparation"
```

---

## Task 2: 원본 Q&A에서 평가 입력 추출 함수

**Files:**
- Modify: `document_processing/slack_qa/quality_schemas.py`
- Modify: `tests/document_processing/test_quality_schemas.py`

**Step 1: 실패하는 테스트 추가**

```python
# tests/document_processing/test_quality_schemas.py (추가)

from document_processing.slack_qa.quality_schemas import extract_for_evaluation


class TestExtractForEvaluation:
    """extract_for_evaluation 함수 테스트."""

    def test_extract_from_qa_pair(self):
        """표준 Q&A 쌍에서 추출."""
        qa_pair = {
            "question": {
                "text": "PyTorch에서 GPU 사용법은?",
                "user": "U123",
                "timestamp": "1699123456.789",
                "is_bot": False,
                "metadata": {"reactions": []},
            },
            "answers": [
                {
                    "text": "torch.cuda.is_available()로 확인하세요.",
                    "user": "U456",
                    "timestamp": "1699123457.000",
                },
                {
                    "text": "model.to('cuda')로 이동합니다.",
                    "user": "U789",
                    "timestamp": "1699123458.000",
                },
            ],
        }

        result = extract_for_evaluation(qa_pair, source_file="test.json")

        assert result.question == "PyTorch에서 GPU 사용법은?"
        assert len(result.answers) == 2
        assert result.original_id == "1699123456.789"
        assert result.source_file == "test.json"

    def test_extract_filters_empty_answers(self):
        """빈 텍스트 답변 필터링."""
        qa_pair = {
            "question": {"text": "Question?", "timestamp": "123"},
            "answers": [
                {"text": "Valid answer"},
                {"text": "   "},  # 공백만
                {"text": ""},  # 빈 문자열
            ],
        }

        result = extract_for_evaluation(qa_pair)
        assert len(result.answers) == 1
        assert result.answers[0] == "Valid answer"
```

**Step 2: 테스트 실행하여 실패 확인**

```bash
pytest tests/document_processing/test_quality_schemas.py::TestExtractForEvaluation -v
```

Expected: FAIL with `ImportError: cannot import name 'extract_for_evaluation'`

**Step 3: 함수 구현**

```python
# document_processing/slack_qa/quality_schemas.py (끝에 추가)

def extract_for_evaluation(
    qa_pair: dict,
    source_file: str = "",
) -> EvaluationInput:
    """
    원본 QA 데이터에서 평가에 필요한 부분만 추출.

    Args:
        qa_pair: 원본 Q&A 쌍 딕셔너리
        source_file: 원본 파일명

    Returns:
        EvaluationInput: 평가용 입력 데이터
    """
    question_data = qa_pair.get("question", {})
    question_text = question_data.get("text", "").strip()
    original_id = question_data.get("timestamp", "")

    answers_data = qa_pair.get("answers", [])
    answer_texts = [
        answer.get("text", "").strip()
        for answer in answers_data
        if answer.get("text", "").strip()
    ]

    return EvaluationInput(
        question=question_text,
        answers=answer_texts,
        original_id=original_id,
        source_file=source_file,
    )
```

**Step 4: 테스트 실행하여 통과 확인**

```bash
pytest tests/document_processing/test_quality_schemas.py::TestExtractForEvaluation -v
```

Expected: PASS

**Step 5: 커밋**

```bash
git add document_processing/slack_qa/quality_schemas.py tests/document_processing/test_quality_schemas.py
git commit -m "feat(slack_qa): add extract_for_evaluation function

Extracts question and answers text from raw Q&A pair,
filtering out empty answers and preserving original_id for mapping"
```

---

## Task 3: LLM 평가 프롬프트 템플릿 작성

**Files:**
- Create: `document_processing/slack_qa/prompts/qa_quality_evaluation.yaml`

**Step 1: 프롬프트 디렉토리 생성**

```bash
mkdir -p document_processing/slack_qa/prompts
```

**Step 2: 프롬프트 템플릿 작성**

```yaml
# document_processing/slack_qa/prompts/qa_quality_evaluation.yaml
_type: chat_messages
metadata:
  name: qa_quality_evaluation
  description: Slack Q&A 데이터 품질 평가 프롬프트
  version: "1.0"
  author: Quality Evaluation System
  last_updated: "2024-11-30"

messages:
  - role: system
    content: |
      # 역할
      당신은 교육용 Q&A 데이터의 품질을 평가하는 전문가입니다.
      네이버 부스트캠프 AI 교육 과정의 Slack Q&A 데이터를 평가합니다.

      # 평가 목적
      이 평가의 목적은 RAG(검색 증강 생성) 시스템에서 사용할 고품질 Q&A를 선별하는 것입니다.
      좋은 Q&A는: 나중에 비슷한 질문을 하는 학생에게 유용한 답변을 제공할 수 있어야 합니다.

      # 평가 기준 (3가지)

      ## 1. 답변 완전성 (Completeness)
      | 점수 | 기준 |
      |-----|------|
      | 5 | 질문의 모든 부분에 완벽히 답변 + 추가 유용한 정보 제공 |
      | 4 | 질문의 모든 부분에 충분히 답변 |
      | 3 | 핵심 질문에는 답변했으나 일부 세부사항 누락 |
      | 2 | 부분적으로만 답변, 중요한 내용 누락 |
      | 1 | 거의 답변이 없거나 질문과 무관 |

      특수 케이스:
      - "~해보세요"만 있고 구체적 방법 없음 → 최대 3점
      - 링크만 제공: 관련성에 따라 2-4점

      ## 2. 맥락 독립성 (Context Independence)
      | 점수 | 기준 |
      |-----|------|
      | 5 | 완전히 독립적, 배경 설명 포함 |
      | 4 | 대부분 이해 가능, 약간의 추론 필요 |
      | 3 | 일부 외부 맥락 필요하지만 핵심은 이해 가능 |
      | 2 | 상당한 외부 맥락 필요 ("그거", "아까" 등) |
      | 1 | 맥락 없이는 이해 불가능 |

      특수 케이스:
      - "그거", "저기" 등 불명확한 지시어 → 감점
      - 이미지/파일 참조 ("첨부 참고"): 내용 없으면 1-2점

      ## 3. 기술적 정확성 (Technical Accuracy)
      | 점수 | 기준 |
      |-----|------|
      | 5 | 완벽히 정확 + 모범 사례 반영 |
      | 4 | 정확함, 작동하는 솔루션 |
      | 3 | 대체로 정확, 사소한 오류 있을 수 있음 |
      | 2 | 부분적으로 부정확, 오해의 소지 |
      | 1 | 심각하게 부정확, 잘못된 정보 |

      특수 케이스:
      - 코드가 없는 개념 질문: 개념의 정확성만 평가
      - "~인 것 같아요" 추측성 답변: 확신도에 따라 감점

      # 종합 등급 판정
      | 등급 | 조건 |
      |-----|------|
      | high | 평균 ≥ 4.0 AND 최저점 ≥ 3 |
      | medium | 평균 ≥ 3.0 AND 최저점 ≥ 2 |
      | low | 평균 ≥ 2.0 OR 최저점 = 1이지만 가치 있음 |
      | remove | 평균 < 2.0 OR 2개 이상 차원이 1점 |

      # 중요 지침
      1. 엄격하게 평가하세요: RAG 품질을 위해 기준을 높게 유지합니다.
      2. reasoning은 반드시 구체적으로 작성하세요.
      3. 한국어 특성 고려: 비격식체, 이모지 사용은 감점 요소 아님.

  - role: human
    content: |
      다음 Q&A를 평가해주세요.

      {qa_content}

      위 Q&A를 3가지 기준으로 평가하고 JSON으로 반환하세요.

input_variables:
  - qa_content
```

**Step 3: 커밋**

```bash
git add document_processing/slack_qa/prompts/
git commit -m "feat(slack_qa): add QA quality evaluation prompt template

Detailed rubric for 3 dimensions: completeness, context_independence, technical_accuracy
With special cases and overall quality grading criteria"
```

---

## Task 4: QualityEvaluator 클래스 구현 (Clova X HCX-007)

**Files:**
- Create: `document_processing/slack_qa/quality_evaluator.py`
- Create: `tests/document_processing/test_quality_evaluator.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/document_processing/test_quality_evaluator.py
"""품질 평가기 테스트."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from document_processing.slack_qa.quality_evaluator import (
    QualityEvaluator,
    emit_quality_evaluation,
)
from document_processing.slack_qa.quality_schemas import (
    EvaluationInput,
    QualityEvaluation,
    DimensionScore,
)


class TestQualityEvaluator:
    """QualityEvaluator 클래스 테스트."""

    def test_init_with_default_model(self):
        """기본 모델(Clova X HCX-007)로 초기화."""
        with patch("document_processing.slack_qa.quality_evaluator.get_chat_model") as mock_get:
            mock_llm = MagicMock()
            mock_get.return_value = mock_llm

            evaluator = QualityEvaluator()
            mock_get.assert_called_once_with(temperature=0.0)
            assert evaluator.llm is not None

    def test_init_with_custom_llm(self):
        """커스텀 LLM으로 초기화."""
        mock_llm = MagicMock()
        evaluator = QualityEvaluator(llm=mock_llm)
        assert evaluator.llm == mock_llm


class TestEmitQualityEvaluation:
    """emit_quality_evaluation Tool 함수 테스트."""

    def test_emit_creates_valid_model(self):
        """Tool 함수가 QualityEvaluation 모델 생성."""
        result = emit_quality_evaluation(
            completeness_score=4,
            completeness_reasoning="Complete answer",
            context_independence_score=3,
            context_independence_reasoning="Some context needed",
            technical_accuracy_score=5,
            technical_accuracy_reasoning="Accurate",
            overall_quality="high",
        )

        assert isinstance(result, QualityEvaluation)
        assert result.completeness.score == 4
        assert result.overall_quality == "high"


class TestQualityEvaluatorEvaluate:
    """QualityEvaluator.evaluate 메서드 테스트."""

    @pytest.fixture
    def mock_agent_response(self):
        """모킹된 Agent 응답."""
        return {
            "messages": [
                MagicMock(content=QualityEvaluation(
                    completeness=DimensionScore(score=4, reasoning="Good"),
                    context_independence=DimensionScore(score=3, reasoning="OK"),
                    technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
                    overall_quality="high",
                ))
            ]
        }

    @pytest.fixture
    def sample_input(self):
        """테스트용 입력 데이터."""
        return EvaluationInput(
            question="PyTorch에서 GPU 사용법은?",
            answers=["torch.cuda.is_available()로 확인", "model.to('cuda')"],
            original_id="123",
            source_file="test.json",
        )

    async def test_evaluate_returns_quality_evaluation(self, sample_input):
        """evaluate가 QualityEvaluation을 반환."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()

        # Agent 응답 모킹 (parse_agent_response가 처리할 형태)
        mock_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Good"),
            context_independence=DimensionScore(score=3, reasoning="OK"),
            technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
            overall_quality="high",
        )
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})

        with patch("document_processing.slack_qa.quality_evaluator.create_structured_agent", return_value=mock_agent):
            with patch("document_processing.slack_qa.quality_evaluator.parse_agent_response", return_value=mock_result):
                evaluator = QualityEvaluator(llm=mock_llm)
                result = await evaluator.evaluate(sample_input)

                assert isinstance(result, QualityEvaluation)
                assert result.overall_quality == "high"
```

**Step 2: 테스트 실행하여 실패 확인**

```bash
pytest tests/document_processing/test_quality_evaluator.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: 평가기 클래스 구현 (Clova X + Tool 기반)**

```python
# document_processing/slack_qa/quality_evaluator.py
"""
LLM 기반 Q&A 품질 평가기.

Clova X HCX-007 모델과 Tool 기반 structured output을 사용합니다.
프로젝트의 표준 패턴(create_structured_agent, parse_agent_response)을 따릅니다.
"""

from typing import Literal

from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from naver_connect_chatbot.config import get_chat_model, logger
from naver_connect_chatbot.service.agents.response_parser import parse_agent_response

from .quality_schemas import EvaluationInput, QualityEvaluation, DimensionScore

# 프롬프트 (YAML 대신 직접 정의 - document_processing은 app 외부이므로)
SYSTEM_PROMPT = '''# 역할
당신은 교육용 Q&A 데이터의 품질을 평가하는 전문가입니다.
네이버 부스트캠프 AI 교육 과정의 Slack Q&A 데이터를 평가합니다.

# 평가 목적
RAG 시스템에서 사용할 고품질 Q&A를 선별합니다.
좋은 Q&A: 비슷한 질문을 하는 학생에게 유용한 답변을 제공할 수 있어야 합니다.

# 평가 기준 (3가지)

## 1. 답변 완전성 (completeness)
| 점수 | 기준 |
|-----|------|
| 5 | 모든 부분에 완벽히 답변 + 추가 유용한 정보 |
| 4 | 모든 부분에 충분히 답변 |
| 3 | 핵심에는 답변, 일부 세부사항 누락 |
| 2 | 부분적으로만 답변, 중요 내용 누락 |
| 1 | 거의 답변 없거나 질문과 무관 |

## 2. 맥락 독립성 (context_independence)
| 점수 | 기준 |
|-----|------|
| 5 | 완전히 독립적, 배경 설명 포함 |
| 4 | 대부분 이해 가능 |
| 3 | 일부 외부 맥락 필요 |
| 2 | 상당한 외부 맥락 필요 ("그거", "아까") |
| 1 | 맥락 없이 이해 불가 |

## 3. 기술적 정확성 (technical_accuracy)
| 점수 | 기준 |
|-----|------|
| 5 | 완벽히 정확 + 모범 사례 |
| 4 | 정확함, 작동하는 솔루션 |
| 3 | 대체로 정확, 사소한 오류 가능 |
| 2 | 부분적으로 부정확 |
| 1 | 심각하게 부정확 |

# 종합 등급 (overall_quality)
- high: 평균 ≥ 4.0 AND 최저점 ≥ 3
- medium: 평균 ≥ 3.0 AND 최저점 ≥ 2
- low: 평균 ≥ 2.0 OR 최저점=1이지만 가치 있음
- remove: 평균 < 2.0 OR 2개 이상 차원이 1점

# 중요
1. 엄격하게 평가하세요
2. reasoning은 구체적으로 (1-2문장)
3. 한국어 비격식체는 감점 아님

IMPORTANT: 평가 완료 후 반드시 emit_quality_evaluation 도구를 호출하세요.
'''


def emit_quality_evaluation(
    completeness_score: int,
    completeness_reasoning: str,
    context_independence_score: int,
    context_independence_reasoning: str,
    technical_accuracy_score: int,
    technical_accuracy_reasoning: str,
    overall_quality: Literal["high", "medium", "low", "remove"],
    improvement_suggestion: str | None = None,
) -> QualityEvaluation:
    """
    Q&A 품질 평가 결과를 구조화된 형식으로 반환합니다.

    이 도구는 Q&A 평가 완료 후 반드시 호출해야 합니다.
    모든 파라미터를 정확히 채워서 호출하세요.

    Args:
        completeness_score: 답변 완전성 점수 (1-5)
        completeness_reasoning: 완전성 점수의 근거
        context_independence_score: 맥락 독립성 점수 (1-5)
        context_independence_reasoning: 맥락 독립성 점수의 근거
        technical_accuracy_score: 기술적 정확성 점수 (1-5)
        technical_accuracy_reasoning: 기술적 정확성 점수의 근거
        overall_quality: 종합 등급 (high/medium/low/remove)
        improvement_suggestion: 품질 개선 제안 (선택사항)

    Returns:
        QualityEvaluation: 구조화된 평가 결과
    """
    return QualityEvaluation(
        completeness=DimensionScore(
            score=completeness_score,
            reasoning=completeness_reasoning,
        ),
        context_independence=DimensionScore(
            score=context_independence_score,
            reasoning=context_independence_reasoning,
        ),
        technical_accuracy=DimensionScore(
            score=technical_accuracy_score,
            reasoning=technical_accuracy_reasoning,
        ),
        overall_quality=overall_quality,
        improvement_suggestion=improvement_suggestion,
    )


def _create_evaluation_agent(llm: Runnable) -> Runnable:
    """품질 평가 Agent 생성."""
    emit_tool = StructuredTool.from_function(
        func=emit_quality_evaluation,
        name="emit_quality_evaluation",
        description="Q&A 품질 평가 결과를 구조화된 형식으로 반환",
    )

    agent = create_agent(
        model=llm,
        tools=[emit_tool],
        system_prompt=SYSTEM_PROMPT,
        name="qa_quality_evaluator",
    )

    return agent


class QualityEvaluator:
    """
    LLM 기반 Q&A 품질 평가기.

    Clova X HCX-007 모델을 사용하며, Tool 기반 structured output 패턴을 따릅니다.

    사용 예시:
        evaluator = QualityEvaluator()
        result = await evaluator.evaluate(evaluation_input)
    """

    # 평가 실패 시 사용할 기본값
    DEFAULT_FALLBACK = QualityEvaluation(
        completeness=DimensionScore(score=1, reasoning="평가 실패"),
        context_independence=DimensionScore(score=1, reasoning="평가 실패"),
        technical_accuracy=DimensionScore(score=1, reasoning="평가 실패"),
        overall_quality="remove",
        improvement_suggestion="LLM 평가 중 오류 발생",
    )

    def __init__(
        self,
        llm: Runnable | None = None,
        temperature: float = 0.0,
    ):
        """
        평가기 초기화.

        Args:
            llm: 사용할 LLM 인스턴스 (None이면 Clova X HCX-007 사용)
            temperature: 생성 온도 (일관성을 위해 0 권장)
        """
        if llm is None:
            self.llm = get_chat_model(temperature=temperature)
        else:
            self.llm = llm

        self._agent = None
        logger.info("QualityEvaluator initialized with Clova X")

    @property
    def agent(self) -> Runnable:
        """평가 Agent (lazy initialization)."""
        if self._agent is None:
            self._agent = _create_evaluation_agent(self.llm)
        return self._agent

    async def evaluate(
        self,
        input_data: EvaluationInput,
        fallback: QualityEvaluation | None = None,
    ) -> QualityEvaluation:
        """
        단일 Q&A 쌍 평가.

        Args:
            input_data: 평가할 Q&A 입력
            fallback: 평가 실패 시 반환할 기본값 (None이면 DEFAULT_FALLBACK)

        Returns:
            QualityEvaluation: 평가 결과
        """
        if fallback is None:
            fallback = self.DEFAULT_FALLBACK

        try:
            qa_content = input_data.to_prompt_format()
            user_message = f"다음 Q&A를 평가해주세요.\n\n{qa_content}"

            response = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            result = parse_agent_response(
                response,
                QualityEvaluation,
                fallback=fallback,
            )

            logger.debug(
                f"Evaluated Q&A {input_data.original_id}: {result.overall_quality}"
            )
            return result

        except Exception as e:
            logger.error(f"Evaluation failed for {input_data.original_id}: {e}")
            return fallback

    def evaluate_sync(
        self,
        input_data: EvaluationInput,
        fallback: QualityEvaluation | None = None,
    ) -> QualityEvaluation:
        """
        동기 방식 평가 (테스트/디버깅용).

        Args:
            input_data: 평가할 Q&A 입력
            fallback: 평가 실패 시 반환할 기본값

        Returns:
            QualityEvaluation: 평가 결과
        """
        if fallback is None:
            fallback = self.DEFAULT_FALLBACK

        try:
            qa_content = input_data.to_prompt_format()
            user_message = f"다음 Q&A를 평가해주세요.\n\n{qa_content}"

            response = self.agent.invoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            return parse_agent_response(
                response,
                QualityEvaluation,
                fallback=fallback,
            )

        except Exception as e:
            logger.error(f"Sync evaluation failed: {e}")
            return fallback
```

**Step 4: 테스트 실행하여 통과 확인**

```bash
pytest tests/document_processing/test_quality_evaluator.py -v
```

Expected: PASS

**Step 5: 커밋**

```bash
git add document_processing/slack_qa/quality_evaluator.py tests/document_processing/test_quality_evaluator.py
git commit -m "feat(slack_qa): add QualityEvaluator class (Clova X HCX-007)

LLM-based quality evaluator using project's standard patterns:
- get_chat_model() for Clova X HCX-007
- Tool-based structured output (emit_quality_evaluation)
- parse_agent_response() with fallback support
- Both async and sync evaluation methods"
```

---

## Task 5: 배치 처리 로직 구현

**Files:**
- Create: `document_processing/slack_qa/batch_processor.py`
- Create: `tests/document_processing/test_batch_processor.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/document_processing/test_batch_processor.py
"""배치 처리 로직 테스트."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from document_processing.slack_qa.batch_processor import (
    BatchConfig,
    ProcessingState,
    BatchProcessor,
)
from document_processing.slack_qa.quality_schemas import (
    EvaluationInput,
    QualityEvaluation,
    DimensionScore,
)


class TestBatchConfig:
    """BatchConfig 테스트."""

    def test_default_values(self):
        """기본값 확인."""
        config = BatchConfig()
        assert config.batch_size == 10
        assert config.max_retries == 3
        assert config.checkpoint_interval == 50


class TestProcessingState:
    """ProcessingState 테스트."""

    def test_save_and_load(self, tmp_path):
        """저장 및 로드."""
        state = ProcessingState()
        state.processed_ids.add("123")
        state.results["123"] = {"score": 4.0}

        checkpoint_path = tmp_path / "checkpoint.json"
        state.save(checkpoint_path)

        loaded = ProcessingState.load(checkpoint_path)
        assert "123" in loaded.processed_ids
        assert loaded.results["123"]["score"] == 4.0

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """존재하지 않는 파일은 빈 상태 반환."""
        loaded = ProcessingState.load(tmp_path / "nonexistent.json")
        assert len(loaded.processed_ids) == 0


class TestBatchProcessor:
    """BatchProcessor 테스트."""

    @pytest.fixture
    def mock_evaluator(self):
        """모킹된 평가기."""
        evaluator = MagicMock()
        mock_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Good"),
            context_independence=DimensionScore(score=3, reasoning="OK"),
            technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
            overall_quality="high",
        )
        evaluator.evaluate = AsyncMock(return_value=mock_result)
        return evaluator

    @pytest.fixture
    def sample_inputs(self):
        """테스트용 입력 데이터."""
        return [
            EvaluationInput(
                question=f"Question {i}?",
                answers=[f"Answer {i}"],
                original_id=str(i),
                source_file="test.json",
            )
            for i in range(5)
        ]

    async def test_process_batch(self, mock_evaluator, sample_inputs):
        """배치 처리."""
        processor = BatchProcessor(evaluator=mock_evaluator)
        results = await processor.process_batch(sample_inputs[:3])

        assert len(results) == 3
        assert all(r[1] is not None for r in results)

    async def test_skips_already_processed(self, mock_evaluator, sample_inputs):
        """이미 처리된 항목 스킵."""
        processor = BatchProcessor(evaluator=mock_evaluator)
        processor.state.processed_ids.add("0")
        processor.state.processed_ids.add("1")

        # 5개 중 2개는 이미 처리됨
        results = await processor.process_batch(sample_inputs)
        assert mock_evaluator.evaluate.call_count == 3  # 3개만 실제 처리
```

**Step 2: 테스트 실행하여 실패 확인**

```bash
pytest tests/document_processing/test_batch_processor.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: 배치 프로세서 구현**

```python
# document_processing/slack_qa/batch_processor.py
"""Q&A 품질 평가 배치 처리."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .quality_schemas import EvaluationInput, QualityEvaluation

if TYPE_CHECKING:
    from .quality_evaluator import QualityEvaluator


@dataclass
class BatchConfig:
    """배치 처리 설정."""

    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_interval: int = 50


@dataclass
class ProcessingState:
    """처리 상태 (중단 후 재개 지원)."""

    processed_ids: set[str] = field(default_factory=set)
    results: dict[str, dict] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """체크포인트 저장."""
        data = {
            "processed_ids": list(self.processed_ids),
            "results": self.results,
            "errors": self.errors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProcessingState":
        """체크포인트 로드."""
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            processed_ids=set(data.get("processed_ids", [])),
            results=data.get("results", {}),
            errors=data.get("errors", []),
        )


class BatchProcessor:
    """Q&A 품질 평가 배치 프로세서."""

    def __init__(
        self,
        evaluator: "QualityEvaluator",
        config: BatchConfig | None = None,
        checkpoint_path: Path | None = None,
    ):
        """
        배치 프로세서 초기화.

        Args:
            evaluator: 품질 평가기
            config: 배치 설정
            checkpoint_path: 체크포인트 파일 경로
        """
        self.evaluator = evaluator
        self.config = config or BatchConfig()
        self.checkpoint_path = checkpoint_path

        if checkpoint_path and checkpoint_path.exists():
            self.state = ProcessingState.load(checkpoint_path)
        else:
            self.state = ProcessingState()

    async def _evaluate_with_retry(
        self,
        item: EvaluationInput,
    ) -> tuple[str, QualityEvaluation | None]:
        """재시도 로직을 포함한 단일 항목 평가."""
        for attempt in range(self.config.max_retries):
            try:
                result = await self.evaluator.evaluate(item)
                return (item.original_id, result)
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.state.errors.append({
                        "original_id": item.original_id,
                        "error": str(e),
                    })
                    return (item.original_id, None)

        return (item.original_id, None)

    async def process_batch(
        self,
        items: list[EvaluationInput],
    ) -> list[tuple[str, QualityEvaluation | None]]:
        """배치 단위로 평가 실행."""
        # 이미 처리된 항목 필터링
        to_process = [
            item for item in items
            if item.original_id not in self.state.processed_ids
        ]

        if not to_process:
            return []

        # 동시 실행
        tasks = [self._evaluate_with_retry(item) for item in to_process]
        results = await asyncio.gather(*tasks)

        # 결과 저장
        for original_id, evaluation in results:
            self.state.processed_ids.add(original_id)
            if evaluation:
                self.state.results[original_id] = evaluation.model_dump()

        return results

    async def process_all(
        self,
        items: list[EvaluationInput],
        progress_callback=None,
    ) -> ProcessingState:
        """전체 항목 배치 처리."""
        total = len(items)
        processed = 0

        for i in range(0, total, self.config.batch_size):
            batch = items[i : i + self.config.batch_size]
            await self.process_batch(batch)

            processed += len(batch)

            # 체크포인트 저장
            if (
                self.checkpoint_path
                and processed % self.config.checkpoint_interval == 0
            ):
                self.state.save(self.checkpoint_path)

            if progress_callback:
                progress_callback(processed, total)

        # 최종 체크포인트 저장
        if self.checkpoint_path:
            self.state.save(self.checkpoint_path)

        return self.state
```

**Step 4: 테스트 실행하여 통과 확인**

```bash
pytest tests/document_processing/test_batch_processor.py -v
```

Expected: PASS

**Step 5: 커밋**

```bash
git add document_processing/slack_qa/batch_processor.py tests/document_processing/test_batch_processor.py
git commit -m "feat(slack_qa): add BatchProcessor for quality evaluation

- BatchConfig for processing settings
- ProcessingState with checkpoint save/load
- BatchProcessor with retry logic and progress tracking
- Skips already processed items for resume capability"
```

---

## Task 6: 모듈 __init__.py 업데이트

**Files:**
- Modify: `document_processing/slack_qa/__init__.py`

**Step 1: __init__.py 업데이트**

```python
# document_processing/slack_qa/__init__.py
"""
Slack Q&A 처리 모듈

Slack 채널의 Q&A 데이터를 로드, 처리, 필터링, 품질 평가하는 기능을 제공합니다.
"""

from .slack_qa_loader import SlackQALoader, QAPair, Message
from .filter_qa_data import filter_qa_pairs, should_remove_question, should_remove_answer
from .quality_schemas import (
    DimensionScore,
    QualityEvaluation,
    EvaluationInput,
    extract_for_evaluation,
)
from .quality_evaluator import QualityEvaluator
from .batch_processor import BatchConfig, ProcessingState, BatchProcessor

__all__ = [
    # Loader
    "SlackQALoader",
    "QAPair",
    "Message",
    # Filter
    "filter_qa_pairs",
    "should_remove_question",
    "should_remove_answer",
    # Quality Evaluation
    "DimensionScore",
    "QualityEvaluation",
    "EvaluationInput",
    "extract_for_evaluation",
    "QualityEvaluator",
    "BatchConfig",
    "ProcessingState",
    "BatchProcessor",
]
```

**Step 2: 커밋**

```bash
git add document_processing/slack_qa/__init__.py
git commit -m "chore(slack_qa): export quality evaluation classes in __init__.py"
```

---

## Task 7: CLI 스크립트 구현

**Files:**
- Create: `document_processing/slack_qa/evaluate_quality.py`

**Step 1: CLI 스크립트 작성**

```python
#!/usr/bin/env python3
# document_processing/slack_qa/evaluate_quality.py
"""
Slack Q&A 품질 평가 CLI 스크립트.

사용법:
    python -m document_processing.slack_qa.evaluate_quality \
        --input document_chunks/slack_qa_merged \
        --output document_chunks/slack_qa_scored \
        --checkpoint quality_checkpoint.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .quality_evaluator import QualityEvaluator
from .quality_schemas import extract_for_evaluation
from .batch_processor import BatchConfig, BatchProcessor


def load_merged_qa_file(file_path: Path) -> list[dict[str, Any]]:
    """병합된 Q&A JSON 파일 로드."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("qa_pairs", [])


def save_scored_qa_file(
    file_path: Path,
    original_data: dict,
    results: dict[str, dict],
) -> dict[str, int]:
    """평가된 Q&A 파일 저장."""
    qa_pairs = original_data.get("qa_pairs", [])
    scored_pairs = []
    stats = {"high": 0, "medium": 0, "low": 0, "removed": 0, "error": 0}

    for qa in qa_pairs:
        original_id = qa.get("question", {}).get("timestamp", "")
        result = results.get(original_id)

        if result is None:
            stats["error"] += 1
            continue

        overall = result.get("overall_quality", "remove")
        stats[overall] += 1

        if overall == "remove":
            continue  # 제거 대상은 저장하지 않음

        # 점수를 Q&A에 추가
        qa_with_score = qa.copy()
        qa_with_score["quality_score"] = result
        scored_pairs.append(qa_with_score)

    # 메타데이터 업데이트
    output_data = original_data.copy()
    output_data["qa_pairs"] = scored_pairs
    output_data["metadata"] = output_data.get("metadata", {})
    output_data["metadata"]["quality_filtered"] = True
    output_data["metadata"]["quality_stats"] = stats
    output_data["metadata"]["original_count"] = len(qa_pairs)
    output_data["metadata"]["filtered_count"] = len(scored_pairs)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return stats


async def process_file(
    input_path: Path,
    output_path: Path,
    processor: BatchProcessor,
) -> dict[str, int]:
    """단일 파일 처리."""
    print(f"Processing: {input_path.name}")

    with open(input_path, encoding="utf-8") as f:
        original_data = json.load(f)

    qa_pairs = original_data.get("qa_pairs", [])
    if not qa_pairs:
        print(f"  No Q&A pairs in {input_path.name}")
        return {"skipped": 1}

    # 평가 입력으로 변환
    inputs = [
        extract_for_evaluation(qa, source_file=input_path.name)
        for qa in qa_pairs
    ]

    # 배치 처리
    def progress(current, total):
        print(f"  Progress: {current}/{total}")

    await processor.process_all(inputs, progress_callback=progress)

    # 결과 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = save_scored_qa_file(output_path, original_data, processor.state.results)

    print(f"  Done: {stats}")
    return stats


async def main(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    model_name: str,
    batch_size: int,
):
    """메인 실행 함수."""
    print("=" * 60)
    print("Slack Q&A LLM Quality Evaluation")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {model_name}")
    print()

    # 평가기 및 프로세서 초기화
    evaluator = QualityEvaluator(model_name=model_name)
    config = BatchConfig(batch_size=batch_size)
    processor = BatchProcessor(
        evaluator=evaluator,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    # 입력 파일 목록
    input_files = sorted(input_dir.glob("*.json"))
    input_files = [f for f in input_files if not f.name.startswith("_")]

    total_stats = {"high": 0, "medium": 0, "low": 0, "removed": 0, "error": 0}

    for input_file in input_files:
        output_file = output_dir / input_file.name
        stats = await process_file(input_file, output_file, processor)

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # 최종 통계
    print()
    print("=" * 60)
    print("Final Statistics")
    print("=" * 60)
    for key, value in total_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Slack Q&A quality")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with merged Q&A files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for scored Q&A files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path for resume",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            input_dir=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    )
```

**Step 2: 커밋**

```bash
git add document_processing/slack_qa/evaluate_quality.py
git commit -m "feat(slack_qa): add quality evaluation CLI script

Usage: python -m document_processing.slack_qa.evaluate_quality --input <dir> --output <dir>
- Processes all merged Q&A JSON files
- Saves scored Q&A with quality_score metadata
- Removes 'remove' quality items from output
- Supports checkpoint for resume"
```

---

## Task 8: 통합 테스트 작성

**Files:**
- Create: `tests/document_processing/test_quality_integration.py`

**Step 1: 통합 테스트 작성**

```python
# tests/document_processing/test_quality_integration.py
"""품질 평가 통합 테스트 (실제 LLM 호출)."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from document_processing.slack_qa.quality_schemas import (
    EvaluationInput,
    QualityEvaluation,
    DimensionScore,
    extract_for_evaluation,
)
from document_processing.slack_qa.quality_evaluator import QualityEvaluator
from document_processing.slack_qa.batch_processor import BatchProcessor, BatchConfig


class TestQualityEvaluationIntegration:
    """품질 평가 통합 테스트 (모킹된 LLM)."""

    @pytest.fixture
    def mock_llm(self):
        """모킹된 LLM."""
        llm = MagicMock()
        structured = MagicMock()

        result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Complete answer"),
            context_independence=DimensionScore(score=5, reasoning="Fully independent"),
            technical_accuracy=DimensionScore(score=4, reasoning="Accurate"),
            overall_quality="high",
        )
        structured.ainvoke = AsyncMock(return_value=result)
        structured.invoke = MagicMock(return_value=result)
        llm.with_structured_output.return_value = structured

        return llm

    async def test_full_pipeline(self, mock_llm):
        """전체 파이프라인 테스트."""
        # 1. 원본 Q&A 데이터
        qa_pair = {
            "question": {
                "text": "PyTorch에서 GPU 메모리 부족 에러 해결법은?",
                "timestamp": "1234567890.123456",
            },
            "answers": [
                {"text": "배치 사이즈를 줄여보세요. torch.cuda.empty_cache()도 도움됩니다."},
            ],
        }

        # 2. 평가 입력 추출
        eval_input = extract_for_evaluation(qa_pair, source_file="test.json")
        assert eval_input.question == "PyTorch에서 GPU 메모리 부족 에러 해결법은?"
        assert len(eval_input.answers) == 1

        # 3. 평가기로 평가
        evaluator = QualityEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(eval_input)

        assert isinstance(result, QualityEvaluation)
        assert result.overall_quality == "high"
        assert result.avg_score >= 4.0

    async def test_batch_processing(self, mock_llm):
        """배치 처리 테스트."""
        inputs = [
            EvaluationInput(
                question=f"Question {i}?",
                answers=[f"Answer {i}"],
                original_id=str(i),
                source_file="test.json",
            )
            for i in range(5)
        ]

        evaluator = QualityEvaluator(llm=mock_llm)
        processor = BatchProcessor(
            evaluator=evaluator,
            config=BatchConfig(batch_size=2),
        )

        state = await processor.process_all(inputs)

        assert len(state.processed_ids) == 5
        assert len(state.results) == 5


# 실제 API를 사용하는 통합 테스트 (수동 실행)
@pytest.mark.integration
@pytest.mark.skip(reason="Requires API key and costs money")
class TestRealLLMIntegration:
    """실제 LLM API를 사용하는 통합 테스트."""

    async def test_real_evaluation(self):
        """실제 API 호출 테스트."""
        evaluator = QualityEvaluator(model_name="gpt-4o-mini")

        input_data = EvaluationInput(
            question="Python에서 리스트 컴프리헨션은 어떻게 사용하나요?",
            answers=[
                "[x**2 for x in range(10)]처럼 대괄호 안에 표현식을 넣으면 됩니다. "
                "조건문도 추가할 수 있어요: [x for x in range(10) if x % 2 == 0]"
            ],
            original_id="test123",
            source_file="test.json",
        )

        result = await evaluator.evaluate(input_data)

        assert isinstance(result, QualityEvaluation)
        assert result.overall_quality in ["high", "medium", "low", "remove"]
        print(f"Result: {result.model_dump_json(indent=2)}")
```

**Step 2: 테스트 실행**

```bash
pytest tests/document_processing/test_quality_integration.py -v
```

Expected: PASS (skipping integration tests)

**Step 3: 커밋**

```bash
git add tests/document_processing/test_quality_integration.py
git commit -m "test(slack_qa): add quality evaluation integration tests

- Full pipeline test with mocked LLM
- Batch processing test
- Real LLM integration test (skipped by default)"
```

---

## Task 9: 최종 테스트 및 문서화

**Step 1: 전체 테스트 실행**

```bash
pytest tests/document_processing/ -v
```

Expected: All tests PASS

**Step 2: 타입 체크 (선택)**

```bash
mypy document_processing/slack_qa/quality_*.py document_processing/slack_qa/batch_processor.py
```

**Step 3: 최종 커밋**

```bash
git add -A
git commit -m "feat(slack_qa): complete LLM quality evaluation system

Implementation complete:
- QualityEvaluation schema with 3 dimensions
- QualityEvaluator with structured output
- BatchProcessor with checkpoints and retry
- CLI script for batch processing
- Comprehensive test coverage

See docs/plans/2024-11-30-slack-qa-llm-quality-evaluation-design.md for details"
```

---

## 구현 체크리스트

- [ ] Task 1: Pydantic 스키마 정의
- [ ] Task 2: 평가 입력 추출 함수
- [ ] Task 3: LLM 평가 프롬프트 템플릿
- [ ] Task 4: QualityEvaluator 클래스
- [ ] Task 5: 배치 처리 로직
- [ ] Task 6: 모듈 __init__.py 업데이트
- [ ] Task 7: CLI 스크립트
- [ ] Task 8: 통합 테스트
- [ ] Task 9: 최종 테스트 및 문서화

---

## 실행 방법

구현 완료 후 실행:

```bash
# 1. 기존 파이프라인 실행 (규칙 기반 필터링)
python -m document_processing.slack_qa.process_all_slack_data
python -m document_processing.slack_qa.filter_qa_data
python -m document_processing.slack_qa.merge_qa_by_course

# 2. LLM 품질 평가 실행
python -m document_processing.slack_qa.evaluate_quality \
    --input document_processing/document_chunks/slack_qa_merged \
    --output document_processing/document_chunks/slack_qa_scored \
    --checkpoint quality_checkpoint.json \
    --model gpt-4o-mini \
    --batch-size 10
```

---

*구현 계획 작성 완료: 2024-11-30*
