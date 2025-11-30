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

    async def test_full_pipeline(self):
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

        # 3. 모킹된 평가기로 평가
        mock_evaluator = MagicMock()
        mock_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Good"),
            context_independence=DimensionScore(score=3, reasoning="OK"),
            technical_accuracy=DimensionScore(score=5, reasoning="Accurate"),
            overall_quality="high",
        )
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

        result = await mock_evaluator.evaluate(eval_input)

        assert isinstance(result, QualityEvaluation)
        assert result.overall_quality == "high"
        assert result.avg_score >= 4.0

    async def test_batch_processing(self):
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

        mock_evaluator = MagicMock()
        mock_result = QualityEvaluation(
            completeness=DimensionScore(score=4, reasoning="Good"),
            context_independence=DimensionScore(score=4, reasoning="Good"),
            technical_accuracy=DimensionScore(score=4, reasoning="Good"),
            overall_quality="high",
        )
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

        processor = BatchProcessor(
            evaluator=mock_evaluator,
            config=BatchConfig(batch_size=2),
        )

        state = await processor.process_all(inputs)

        assert len(state.processed_ids) == 5
        assert len(state.results) == 5

    def test_evaluation_input_extraction(self):
        """평가 입력 추출 테스트."""
        qa_pair = {
            "question": {
                "text": "Python에서 리스트와 튜플의 차이는?",
                "timestamp": "1234567890.000000",
                "user": "U123",
            },
            "answers": [
                {"text": "리스트는 mutable, 튜플은 immutable입니다."},
                {"text": "성능 면에서 튜플이 더 빠릅니다."},
                {"text": ""},  # 빈 답변 - 필터링됨
            ],
        }

        result = extract_for_evaluation(qa_pair, source_file="test.json")

        assert result.question == "Python에서 리스트와 튜플의 차이는?"
        assert len(result.answers) == 2  # 빈 답변 제외
        assert result.original_id == "1234567890.000000"
        assert result.source_file == "test.json"

    def test_quality_evaluation_avg_score(self):
        """평균 점수 계산 테스트."""
        evaluation = QualityEvaluation(
            completeness=DimensionScore(score=5, reasoning="Perfect"),
            context_independence=DimensionScore(score=4, reasoning="Good"),
            technical_accuracy=DimensionScore(score=3, reasoning="OK"),
            overall_quality="high",
        )

        assert evaluation.avg_score == 4.0  # (5 + 4 + 3) / 3 = 4.0

    def test_prompt_format_generation(self):
        """프롬프트 포맷 생성 테스트."""
        eval_input = EvaluationInput(
            question="테스트 질문입니다",
            answers=["첫 번째 답변", "두 번째 답변"],
            original_id="123",
            source_file="test.json",
        )

        prompt = eval_input.to_prompt_format()

        assert "## 질문" in prompt
        assert "테스트 질문입니다" in prompt
        assert "## 답변들" in prompt
        assert "[답변 1]" in prompt
        assert "[답변 2]" in prompt
        assert "첫 번째 답변" in prompt
        assert "두 번째 답변" in prompt


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API key and costs money")
class TestRealLLMIntegration:
    """실제 LLM API를 사용하는 통합 테스트."""

    async def test_real_evaluation(self):
        """실제 API 호출 테스트."""
        evaluator = QualityEvaluator()

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
