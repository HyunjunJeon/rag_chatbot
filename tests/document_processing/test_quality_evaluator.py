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

        mock_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Good"),
            context_independence=DimensionScore(score=3, reasoning="OK"),
            technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
            overall_quality="high",
        )
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})

        with patch("document_processing.slack_qa.quality_evaluator.create_agent", return_value=mock_agent):
            with patch("document_processing.slack_qa.quality_evaluator.parse_agent_response", return_value=mock_result):
                evaluator = QualityEvaluator(llm=mock_llm)
                result = await evaluator.evaluate(sample_input)

                assert isinstance(result, QualityEvaluation)
                assert result.overall_quality == "high"

    async def test_evaluate_uses_fallback_on_error(self, sample_input):
        """오류 발생 시 fallback 사용."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("API Error"))

        with patch("document_processing.slack_qa.quality_evaluator.create_agent", return_value=mock_agent):
            evaluator = QualityEvaluator(llm=mock_llm)
            result = await evaluator.evaluate(sample_input)

            # 기본 fallback 반환
            assert isinstance(result, QualityEvaluation)
            assert result.overall_quality == "remove"
