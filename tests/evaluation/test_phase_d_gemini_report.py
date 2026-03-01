"""
Phase D 릴리스 게이트 테스트.

목표:
- Gemini 기반 Judge 평가가 동작해야 한다.
- 평가 리포트 산출물이 반드시 생성되어야 한다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.documents import Document

from tests.evaluation.evaluators import EvaluationReport, LLMJudgeEvaluator, QuestionEvaluation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase_d_gemini_judge_generates_report_artifact():
    """Gemini Judge로 평가 후 리포트 JSON 산출물을 생성한다."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.fail("GOOGLE_API_KEY is required for release-gate evaluation")

    evaluator = LLMJudgeEvaluator()

    question_data = {
        "id": "phase_d_smoke_001",
        "question": "PyTorch 텐서의 개념을 간단히 설명해줘.",
        "category": "in_domain",
        "subcategory": "concept",
        "ground_truth": {
            "expected_behavior": "provide_answer",
            "answer_keywords": ["텐서", "다차원", "배열"],
        },
        "metadata": {"difficulty": "easy"},
    }

    answer = "PyTorch 텐서는 다차원 배열을 표현하는 기본 데이터 구조입니다."
    docs = [
        Document(
            page_content="PyTorch tensor는 다차원 배열 자료구조이며 GPU 연산을 지원합니다.",
            metadata={"source_file": "smoke_doc_1", "doc_type": "pdf", "course": "AI 기초"},
        )
    ]

    judge_result = await evaluator.evaluate(question_data, answer, docs)

    result = QuestionEvaluation(
        question_id=question_data["id"],
        question=question_data["question"],
        category=question_data["category"],
        subcategory=question_data["subcategory"],
        answer=answer,
        retrieved_docs_count=len(docs),
        passed=judge_result.is_passing,
        judge_evaluation=judge_result,
    )

    report = EvaluationReport(
        dataset_version="phase_d_release_gate",
        timestamp=datetime.now().isoformat(),
        total_questions=1,
        passed_questions=1 if result.passed else 0,
        results=[result.to_dict()],
    )

    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"evaluation_report_phase_d_{datetime.now():%Y%m%d_%H%M%S}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    with open(report_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert report_path.exists()
    assert saved["dataset_version"] == "phase_d_release_gate"
    assert saved["total_questions"] == 1
    assert isinstance(saved.get("results"), list) and len(saved["results"]) == 1
    assert "judge" in saved["results"][0]
