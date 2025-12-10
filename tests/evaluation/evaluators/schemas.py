"""
LLM-as-Judge 평가 스키마 정의.

Pydantic v2 모델로 평가 결과의 타입 안전성을 보장합니다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class JudgeEvaluation(BaseModel):
    """LLM-as-Judge 평가 결과.

    HCX-007이 반환하는 구조화된 평가 결과입니다.
    """

    faithfulness: int = Field(
        ge=1, le=5,
        description="충실성 점수 (1-5): 검색 문서 기반 답변 여부"
    )
    relevance: int = Field(
        ge=1, le=5,
        description="관련성 점수 (1-5): 질문에 대한 답변 적절성"
    )
    completeness: int = Field(
        ge=1, le=5,
        description="완전성 점수 (1-5): 답변의 충분성"
    )
    hallucination_detected: bool = Field(
        description="환각 탐지 여부"
    )
    behavior_correct: bool = Field(
        description="기대 행동 수행 여부 (OOD/Edge Case용)"
    )
    reasoning: str = Field(
        description="평가 근거 (2-3문장)"
    )

    @computed_field
    @property
    def overall_score(self) -> float:
        """종합 점수 (0-1 스케일).

        환각 탐지 시 0.2 페널티 적용.
        """
        base = (self.faithfulness + self.relevance + self.completeness) / 15
        penalty = 0.2 if self.hallucination_detected else 0
        return max(0.0, base - penalty)

    @computed_field
    @property
    def is_passing(self) -> bool:
        """통과 여부 (overall_score >= 0.6)"""
        return self.overall_score >= 0.6 and not self.hallucination_detected


class QuestionEvaluation(BaseModel):
    """개별 질문 평가 결과.

    질문 메타데이터와 Judge 평가를 통합합니다.
    """

    question_id: str
    question: str
    category: str
    subcategory: str

    # RAG 결과
    answer: str = ""
    retrieved_docs_count: int = 0
    retrieval_time_ms: float = 0.0
    filter_applied: bool = False

    # Judge 평가
    judge_evaluation: JudgeEvaluation | None = None

    # 최종 결과
    passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        """딕셔너리 변환 (리포트용)"""
        result = {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "subcategory": self.subcategory,
            "answer": self.answer[:500] if self.answer else "",
            "retrieved_docs_count": self.retrieved_docs_count,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "filter_applied": self.filter_applied,
            "passed": self.passed,
            "error": self.error,
        }

        if self.judge_evaluation:
            result["judge"] = {
                "faithfulness": self.judge_evaluation.faithfulness,
                "relevance": self.judge_evaluation.relevance,
                "completeness": self.judge_evaluation.completeness,
                "hallucination_detected": self.judge_evaluation.hallucination_detected,
                "behavior_correct": self.judge_evaluation.behavior_correct,
                "overall_score": round(self.judge_evaluation.overall_score, 3),
                "reasoning": self.judge_evaluation.reasoning,
            }

        return result


class EvaluationReport(BaseModel):
    """전체 평가 리포트.

    카테고리별/과정별 통계를 집계합니다.
    """

    dataset_version: str = "2.0.0"
    timestamp: str = ""
    total_questions: int = 0
    passed_questions: int = 0

    # 카테고리별 통계
    by_category: dict[str, dict[str, int | float]] = Field(default_factory=dict)
    by_subcategory: dict[str, dict[str, int | float]] = Field(default_factory=dict)
    by_course: dict[str, dict[str, int | float]] = Field(default_factory=dict)

    # 세부 결과
    results: list[dict] = Field(default_factory=list)

    # 실패 질문 목록
    failed_questions: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """전체 통과율.

        Note: @computed_field로 model_dump() 시 직렬화됩니다.
        """
        if self.total_questions == 0:
            return 0.0
        return self.passed_questions / self.total_questions

    @computed_field
    @property
    def overall_score(self) -> float:
        """평균 overall_score.

        Note: @computed_field로 model_dump() 시 직렬화됩니다.
        """
        scores = [
            r.get("judge", {}).get("overall_score", 0)
            for r in self.results
            if r.get("judge")
        ]
        return sum(scores) / len(scores) if scores else 0.0
