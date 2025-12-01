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
        assert config.batch_size == 1  # Rate limiting을 위해 순차 처리
        assert config.max_retries == 3
        assert config.checkpoint_interval == 10
        assert config.requests_per_minute == 30

    def test_min_request_interval(self):
        """요청 간 최소 대기 시간 계산."""
        config = BatchConfig(requests_per_minute=60)
        # 60 RPM = 1초 간격, +10% 버퍼 = 1.1초
        assert config.min_request_interval == pytest.approx(1.1)

    def test_custom_rate_limit(self):
        """커스텀 rate limit 설정."""
        config = BatchConfig(requests_per_minute=20)
        # 20 RPM = 3초 간격, +10% 버퍼 = 3.3초
        assert config.min_request_interval == pytest.approx(3.3)


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

    def test_atomic_save(self, tmp_path):
        """atomic write (임시 파일 사용)."""
        state = ProcessingState()
        state.processed_ids.add("123")

        checkpoint_path = tmp_path / "checkpoint.json"
        state.save(checkpoint_path)

        # 임시 파일이 남아있지 않아야 함
        assert not (tmp_path / "checkpoint.tmp").exists()
        assert checkpoint_path.exists()


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
        config = BatchConfig(requests_per_minute=6000)  # 테스트용 빠른 rate limit
        processor = BatchProcessor(evaluator=mock_evaluator, config=config)
        results = await processor.process_batch(sample_inputs[:3])

        assert len(results) == 3
        assert all(r[1] is not None for r in results)

    async def test_skips_already_processed(self, mock_evaluator, sample_inputs):
        """이미 처리된 항목 스킵."""
        config = BatchConfig(requests_per_minute=6000)
        processor = BatchProcessor(evaluator=mock_evaluator, config=config)
        processor.state.processed_ids.add("0")
        processor.state.processed_ids.add("1")

        # 5개 중 2개는 이미 처리됨
        results = await processor.process_batch(sample_inputs)
        assert mock_evaluator.evaluate.call_count == 3  # 3개만 실제 처리

    async def test_process_all_with_checkpoint(self, mock_evaluator, sample_inputs, tmp_path):
        """전체 처리 with 체크포인트."""
        checkpoint_path = tmp_path / "checkpoint.json"
        config = BatchConfig(
            batch_size=1,
            checkpoint_interval=2,
            requests_per_minute=6000,  # 테스트용 빠른 rate limit
        )
        processor = BatchProcessor(
            evaluator=mock_evaluator,
            config=config,
            checkpoint_path=checkpoint_path,
        )

        state = await processor.process_all(sample_inputs)

        assert len(state.processed_ids) == 5
        assert checkpoint_path.exists()

    async def test_resume_from_checkpoint(self, mock_evaluator, sample_inputs, tmp_path):
        """체크포인트에서 재개."""
        checkpoint_path = tmp_path / "checkpoint.json"

        # 먼저 2개 처리
        config = BatchConfig(checkpoint_interval=1, requests_per_minute=6000)
        processor1 = BatchProcessor(
            evaluator=mock_evaluator,
            config=config,
            checkpoint_path=checkpoint_path,
        )
        await processor1.process_batch(sample_inputs[:2])
        processor1.state.save(checkpoint_path)

        # 새 프로세서로 재개
        mock_evaluator.evaluate.reset_mock()
        processor2 = BatchProcessor(
            evaluator=mock_evaluator,
            config=config,
            checkpoint_path=checkpoint_path,
        )

        # 이미 처리된 2개는 스킵
        assert len(processor2.state.processed_ids) == 2

        # 나머지 3개만 처리
        await processor2.process_all(sample_inputs)
        assert mock_evaluator.evaluate.call_count == 3

    async def test_get_stats(self, mock_evaluator, sample_inputs):
        """통계 조회."""
        config = BatchConfig(requests_per_minute=6000)
        processor = BatchProcessor(evaluator=mock_evaluator, config=config)
        await processor.process_batch(sample_inputs[:3])

        stats = processor.get_stats()
        assert stats["processed"] == 3
        assert stats["success"] == 3
        assert stats["errors"] == 0
        assert stats["success_rate"] == 100.0
