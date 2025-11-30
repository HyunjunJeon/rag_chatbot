"""Q&A 품질 평가 배치 처리."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

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

        return list(results)

    async def process_all(
        self,
        items: list[EvaluationInput],
        progress_callback: Callable[[int, int], None] | None = None,
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
