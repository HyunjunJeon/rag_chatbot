"""Q&A 품질 평가 배치 처리 (Rate Limiting 지원)."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .quality_schemas import EvaluationInput, QualityEvaluation

if TYPE_CHECKING:
    from .quality_evaluator import QualityEvaluator


@dataclass
class BatchConfig:
    """배치 처리 설정.

    Naver API Rate Limit 고려:
    - requests_per_minute: 분당 요청 수 (기본 30 - 안전한 값)
    - checkpoint_interval: 체크포인트 저장 주기 (기본 10)
    """

    batch_size: int = 1  # Rate limiting을 위해 순차 처리 권장
    max_retries: int = 3
    retry_delay: float = 2.0  # 재시도 대기 시간
    checkpoint_interval: int = 10  # 더 자주 저장
    requests_per_minute: int = 30  # Naver API safe limit
    rate_limit_buffer: float = 0.1  # 추가 버퍼 (10%)

    @property
    def min_request_interval(self) -> float:
        """요청 간 최소 대기 시간 (초)."""
        base_interval = 60.0 / self.requests_per_minute
        return base_interval * (1 + self.rate_limit_buffer)


@dataclass
class ProcessingState:
    """처리 상태 (중단 후 재개 지원)."""

    processed_ids: set[str] = field(default_factory=set)
    results: dict[str, dict] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    last_request_time: float = 0.0

    def save(self, path: Path) -> None:
        """체크포인트 저장."""
        data = {
            "processed_ids": list(self.processed_ids),
            "results": self.results,
            "errors": self.errors,
            "stats": {
                "total_processed": len(self.processed_ids),
                "total_success": len(self.results),
                "total_errors": len(self.errors),
            },
        }
        # 임시 파일에 먼저 쓰고 이동 (atomic write)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ProcessingState":
        """체크포인트 로드."""
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        state = cls(
            processed_ids=set(data.get("processed_ids", [])),
            results=data.get("results", {}),
            errors=data.get("errors", []),
        )

        print(f"  [Resume] Loaded checkpoint: {len(state.processed_ids)} already processed")
        return state


class BatchProcessor:
    """Q&A 품질 평가 배치 프로세서 (Rate Limiting 지원)."""

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
            config: 배치 설정 (rate limiting 포함)
            checkpoint_path: 체크포인트 파일 경로
        """
        self.evaluator = evaluator
        self.config = config or BatchConfig()
        self.checkpoint_path = checkpoint_path
        self._last_request_time = 0.0

        if checkpoint_path and checkpoint_path.exists():
            self.state = ProcessingState.load(checkpoint_path)
        else:
            self.state = ProcessingState()

    async def _wait_for_rate_limit(self) -> None:
        """Rate limit을 준수하기 위해 대기."""
        if self._last_request_time == 0:
            self._last_request_time = time.time()
            return

        elapsed = time.time() - self._last_request_time
        wait_time = self.config.min_request_interval - elapsed

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    async def _evaluate_with_retry(
        self,
        item: EvaluationInput,
    ) -> tuple[str, QualityEvaluation | None]:
        """재시도 로직을 포함한 단일 항목 평가."""
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                await self._wait_for_rate_limit()

                result = await self.evaluator.evaluate(item)
                return (item.original_id, result)

            except Exception as e:
                error_msg = str(e)

                # Rate limit 에러 감지 (429 또는 관련 메시지)
                is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()

                if attempt < self.config.max_retries - 1:
                    # Rate limit 에러면 더 오래 대기
                    wait_time = self.config.retry_delay * (attempt + 1)
                    if is_rate_limit:
                        wait_time = wait_time * 3  # 3배 더 대기
                        print(f"  [Rate Limit] Waiting {wait_time:.1f}s before retry...")

                    await asyncio.sleep(wait_time)
                else:
                    self.state.errors.append({
                        "original_id": item.original_id,
                        "error": error_msg,
                        "attempts": attempt + 1,
                    })
                    return (item.original_id, None)

        return (item.original_id, None)

    async def process_batch(
        self,
        items: list[EvaluationInput],
    ) -> list[tuple[str, QualityEvaluation | None]]:
        """배치 단위로 평가 실행 (순차 처리로 rate limit 준수)."""
        # 이미 처리된 항목 필터링
        to_process = [
            item for item in items
            if item.original_id not in self.state.processed_ids
        ]

        if not to_process:
            return []

        results = []

        # Rate limit을 위해 순차 처리
        for item in to_process:
            result = await self._evaluate_with_retry(item)
            results.append(result)

            original_id, evaluation = result
            self.state.processed_ids.add(original_id)
            if evaluation:
                self.state.results[original_id] = evaluation.model_dump()

        return results

    async def process_all(
        self,
        items: list[EvaluationInput],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ProcessingState:
        """전체 항목 배치 처리."""
        # 이미 처리된 항목 제외
        remaining_items = [
            item for item in items
            if item.original_id not in self.state.processed_ids
        ]

        total = len(items)
        already_processed = len(self.state.processed_ids)
        remaining = len(remaining_items)

        if already_processed > 0:
            print(f"  [Resume] Skipping {already_processed} already processed items")
            print(f"  [Resume] {remaining} items remaining")

        processed_count = already_processed

        for i, item in enumerate(remaining_items):
            # 단일 항목 처리
            await self.process_batch([item])
            processed_count += 1

            # 체크포인트 저장 (checkpoint_interval마다)
            if (
                self.checkpoint_path
                and (i + 1) % self.config.checkpoint_interval == 0
            ):
                self.state.save(self.checkpoint_path)
                print(f"  [Checkpoint] Saved at {processed_count}/{total}")

            # 진행상황 콜백
            if progress_callback:
                progress_callback(processed_count, total)

        # 최종 체크포인트 저장
        if self.checkpoint_path:
            self.state.save(self.checkpoint_path)

        return self.state

    def get_stats(self) -> dict:
        """현재 처리 통계 반환."""
        return {
            "processed": len(self.state.processed_ids),
            "success": len(self.state.results),
            "errors": len(self.state.errors),
            "success_rate": (
                len(self.state.results) / len(self.state.processed_ids) * 100
                if self.state.processed_ids else 0
            ),
        }
