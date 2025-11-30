"""품질 평가 스키마 테스트."""

import pytest
from document_processing.slack_qa.quality_schemas import (
    DimensionScore,
    QualityEvaluation,
    EvaluationInput,
    extract_for_evaluation,
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
