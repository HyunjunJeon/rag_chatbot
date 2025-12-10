"""
BaseAuditor 추상 클래스.

모든 점검 모듈의 기반 클래스로, 공통 인터페이스와 유틸리티를 제공합니다.

사용 예:
    ```python
    from document_processing.audit.auditors.base import BaseAuditor
    from document_processing.audit.models import LayerResult

    class MyAuditor(BaseAuditor):
        layer_name = "my_layer"

        async def audit(self) -> LayerResult:
            result = self.create_result()
            # 점검 로직
            return result
    ```
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from document_processing.audit.models.audit_result import (
    Issue,
    LayerResult,
    LayerStats,
    Severity,
)

logger = logging.getLogger(__name__)


class BaseAuditor(ABC):
    """
    점검 모듈의 기반 추상 클래스.

    Attributes:
        layer_name: 점검 레이어 이름 (서브클래스에서 정의)
        base_path: 프로젝트 루트 경로
        verbose: 상세 로깅 여부
    """

    layer_name: str = "base"  # 서브클래스에서 오버라이드

    def __init__(
        self,
        base_path: Path | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            base_path: 프로젝트 루트 경로 (기본: 현재 파일 기준 계산)
            verbose: 상세 로깅 여부
        """
        if base_path is None:
            # document_processing/audit/auditors/base.py 기준으로 루트 찾기
            self.base_path = Path(__file__).parent.parent.parent.parent
        else:
            self.base_path = base_path

        self.verbose = verbose
        self._progress_callback: Callable[[int, int, str], None] | None = None
        self._start_time: float = 0
        self._result: LayerResult | None = None

    @abstractmethod
    async def audit(self) -> LayerResult:
        """
        점검을 실행하고 결과를 반환합니다.

        서브클래스에서 반드시 구현해야 합니다.

        Returns:
            LayerResult: 점검 결과
        """
        raise NotImplementedError

    def create_result(self) -> LayerResult:
        """새로운 LayerResult 인스턴스를 생성합니다."""
        self._result = LayerResult(
            layer_name=self.layer_name,
            status="pass",
            total_items=0,
            issues=[],
            stats=LayerStats(),
        )
        return self._result

    def add_issue(
        self,
        severity: Severity | str,
        category: str,
        message: str,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        현재 결과에 이슈를 추가합니다.

        Args:
            severity: 심각도 (critical, warning, info)
            category: 카테고리 (metadata, content, sync 등)
            message: 이슈 설명
            file_path: 관련 파일 경로
            details: 추가 상세 정보
        """
        if self._result is None:
            raise RuntimeError("create_result()를 먼저 호출하세요")

        if isinstance(severity, str):
            severity = Severity(severity)

        self._result.add_issue(
            severity=severity,
            category=category,
            message=message,
            file_path=file_path,
            details=details,
        )

        if self.verbose:
            issue = Issue(
                severity=severity,
                category=category,
                message=message,
                file_path=file_path,
                details=details,
            )
            logger.info(str(issue))

    def set_progress_callback(
        self,
        callback: Callable[[int, int, str], None],
    ) -> None:
        """진행 상황 콜백을 설정합니다."""
        self._progress_callback = callback

    def log_progress(
        self,
        current: int,
        total: int,
        item: str = "",
    ) -> None:
        """
        진행 상황을 로깅합니다.

        Args:
            current: 현재 진행 항목 번호
            total: 전체 항목 수
            item: 현재 처리 중인 항목 설명
        """
        if self._progress_callback:
            self._progress_callback(current, total, item)

        if self.verbose:
            percent = (current / total * 100) if total > 0 else 0
            logger.info(f"[{self.layer_name}] {current}/{total} ({percent:.1f}%) {item}")

    def start_timer(self) -> None:
        """점검 시간 측정을 시작합니다."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> float:
        """
        점검 시간 측정을 종료하고 소요 시간을 반환합니다.

        Returns:
            소요 시간 (초)
        """
        duration = time.perf_counter() - self._start_time
        if self._result:
            self._result.duration_seconds = round(duration, 3)
        return duration

    def finalize_result(self) -> LayerResult:
        """
        결과를 마무리하고 반환합니다.

        - 타이머 종료
        - 상태 결정
        - 통계 업데이트

        Returns:
            최종 LayerResult
        """
        if self._result is None:
            raise RuntimeError("create_result()를 먼저 호출하세요")

        self.stop_timer()
        self._result.determine_status()

        if self.verbose:
            logger.info(
                f"[{self.layer_name}] 완료: {self._result.status} "
                f"(이슈 {len(self._result.issues)}개, {self._result.duration_seconds:.2f}초)"
            )

        return self._result

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    def resolve_path(self, relative_path: str) -> Path:
        """
        상대 경로를 절대 경로로 변환합니다.

        Args:
            relative_path: base_path 기준 상대 경로

        Returns:
            절대 경로
        """
        return self.base_path / relative_path

    async def run_in_executor(
        self,
        func: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """
        동기 함수를 executor에서 실행합니다.

        CPU 바운드 작업이나 블로킹 I/O에 유용합니다.

        Args:
            func: 실행할 동기 함수
            *args: 함수 인자

        Returns:
            함수 실행 결과
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        파일 크기를 사람이 읽기 좋은 형식으로 변환합니다.

        Args:
            size_bytes: 바이트 단위 크기

        Returns:
            형식화된 문자열 (예: "1.5 MB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def count_files_by_extension(directory: Path) -> dict[str, int]:
        """
        디렉토리 내 파일을 확장자별로 카운트합니다.

        Args:
            directory: 대상 디렉토리

        Returns:
            {확장자: 개수} 딕셔너리
        """
        counts: dict[str, int] = {}
        if not directory.exists():
            return counts

        for file in directory.rglob("*"):
            if file.is_file():
                ext = file.suffix.lower() or "(no ext)"
                counts[ext] = counts.get(ext, 0) + 1

        return counts


# =============================================================================
# 테스트용 구현
# =============================================================================


class DummyAuditor(BaseAuditor):
    """테스트용 더미 점검기."""

    layer_name = "dummy"

    async def audit(self) -> LayerResult:
        """더미 점검 실행."""
        result = self.create_result()
        self.start_timer()

        # 테스트 이슈 추가
        self.add_issue(
            severity="info",
            category="test",
            message="This is a test issue",
        )

        result.total_items = 10
        result.stats.total_items = 10
        result.stats.checked_items = 10
        result.stats.passed_items = 9
        result.stats.failed_items = 1

        return self.finalize_result()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = DummyAuditor(verbose=True)
        result = await auditor.audit()
        print(f"\n결과: {result.model_dump_json(indent=2)}")

    asyncio.run(main())
