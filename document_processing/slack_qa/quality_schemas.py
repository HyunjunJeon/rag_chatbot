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

    def to_prompt_format(self, max_answers: int = 2) -> str:
        """LLM 프롬프트에 삽입할 포맷으로 변환.

        Args:
            max_answers: 포함할 최대 답변 개수 (기본값 2).
                         답변이 많을수록 API 실패율이 높아지므로 제한 권장.
        """
        # 답변 개수 제한 (너무 많으면 API 실패율 증가)
        truncated_answers = self.answers[:max_answers]

        if truncated_answers:
            answers_formatted = "\n\n".join(
                f"[Answer {i + 1}]\n{answer}" for i, answer in enumerate(truncated_answers)
            )
            # 답변이 잘렸다면 표시
            if len(self.answers) > max_answers:
                answers_formatted += (
                    f"\n\n(+{len(self.answers) - max_answers} more answers omitted)"
                )
        else:
            answers_formatted = "(No answers)"

        return f"""## Question
{self.question}

## Answers
{answers_formatted}"""


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
        answer.get("text", "").strip() for answer in answers_data if answer.get("text", "").strip()
    ]

    return EvaluationInput(
        question=question_text,
        answers=answer_texts,
        original_id=original_id,
        source_file=source_file,
    )
