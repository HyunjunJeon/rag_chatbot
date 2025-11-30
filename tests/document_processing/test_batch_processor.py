"""배치 처리 로직 테스트."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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

    async def test_process_all_with_checkpoint(self, mock_evaluator, sample_inputs, tmp_path):
        """전체 처리 with 체크포인트."""
        checkpoint_path = tmp_path / "checkpoint.json"
        processor = BatchProcessor(
            evaluator=mock_evaluator,
            config=BatchConfig(batch_size=2, checkpoint_interval=2),
            checkpoint_path=checkpoint_path,
        )

        state = await processor.process_all(sample_inputs)

        assert len(state.processed_ids) == 5
        assert checkpoint_path.exists()
