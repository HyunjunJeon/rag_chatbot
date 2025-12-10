"""
JSON 형식 리포트 생성기.

점검 결과를 JSON 파일로 저장합니다.

사용 예:
    ```python
    from document_processing.audit.reporters.json_reporter import JSONReporter
    from document_processing.audit.models import AuditReport

    reporter = JSONReporter(output_dir="audit_reports")
    filepath = reporter.generate(report)
    print(f"리포트 저장됨: {filepath}")
    ```
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from document_processing.audit.models.audit_result import AuditReport

logger = logging.getLogger(__name__)


class JSONReporter:
    """JSON 형식 리포트 생성기."""

    def __init__(
        self,
        output_dir: str | Path = "audit_reports",
        indent: int = 2,
        include_details: bool = True,
    ):
        """
        Args:
            output_dir: 리포트 출력 디렉토리
            indent: JSON 들여쓰기 공백 수
            include_details: 상세 정보 포함 여부
        """
        self.output_dir = Path(output_dir)
        self.indent = indent
        self.include_details = include_details

    def generate(
        self,
        report: AuditReport,
        filename: str | None = None,
    ) -> Path:
        """
        리포트를 JSON 파일로 생성합니다.

        Args:
            report: 점검 결과 리포트
            filename: 출력 파일명 (기본: audit_report_YYYYMMDD_HHMMSS.json)

        Returns:
            생성된 파일 경로
        """
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.json"

        filepath = self.output_dir / filename

        # 리포트 데이터 준비
        data = self._prepare_report_data(report)

        # JSON 파일 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=self.indent, default=str)

        logger.info(f"JSON 리포트 생성됨: {filepath}")
        return filepath

    def _prepare_report_data(self, report: AuditReport) -> dict[str, Any]:
        """리포트 데이터를 딕셔너리로 변환."""
        data: dict[str, Any] = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "audit_timestamp": report.timestamp.isoformat(),
                "version": "1.0.0",
            },
            "summary": report.to_summary_dict(),
            "layers": [],
            "recommendations": report.recommendations,
        }

        # 레이어별 결과
        for layer in report.layers:
            layer_data: dict[str, Any] = {
                "name": layer.layer_name,
                "status": layer.status,
                "total_items": layer.total_items,
                "duration_seconds": layer.duration_seconds,
                "issue_counts": {
                    "critical": layer.critical_count,
                    "warning": layer.warning_count,
                    "info": layer.info_count,
                },
                "stats": {
                    "total_items": layer.stats.total_items,
                    "checked_items": layer.stats.checked_items,
                    "passed_items": layer.stats.passed_items,
                    "failed_items": layer.stats.failed_items,
                    "pass_rate": layer.stats.pass_rate,
                },
            }

            # 상세 정보 포함
            if self.include_details:
                layer_data["stats"]["extra"] = layer.stats.extra
                layer_data["issues"] = [
                    {
                        "severity": issue.severity.value,
                        "category": issue.category,
                        "message": issue.message,
                        "file_path": issue.file_path,
                        "details": issue.details,
                    }
                    for issue in layer.issues
                ]

            data["layers"].append(layer_data)

        return data

    def generate_summary_only(
        self,
        report: AuditReport,
        filename: str | None = None,
    ) -> Path:
        """
        요약만 포함된 간단한 리포트를 생성합니다.

        Args:
            report: 점검 결과 리포트
            filename: 출력 파일명

        Returns:
            생성된 파일 경로
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_summary_{timestamp}.json"

        filepath = self.output_dir / filename

        # 요약 데이터만
        data = report.to_summary_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=self.indent, default=str)

        logger.info(f"JSON 요약 리포트 생성됨: {filepath}")
        return filepath


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    from document_processing.audit.models.audit_result import (
        AuditReport,
        LayerResult,
        LayerStats,
        Severity,
    )

    # 테스트 리포트 생성
    layer = LayerResult(
        layer_name="test",
        status="warning",
        total_items=100,
        stats=LayerStats(
            total_items=100,
            checked_items=100,
            passed_items=95,
            failed_items=5,
            extra={"test_metric": 42},
        ),
    )
    layer.add_issue(
        severity=Severity.WARNING,
        category="test",
        message="테스트 이슈입니다",
        details={"key": "value"},
    )

    report = AuditReport(layers=[layer])
    report.generate_recommendations()

    # 리포트 생성
    reporter = JSONReporter(output_dir="/tmp/audit_test")
    filepath = reporter.generate(report)
    print(f"리포트 생성됨: {filepath}")

    # 내용 출력
    with open(filepath) as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
