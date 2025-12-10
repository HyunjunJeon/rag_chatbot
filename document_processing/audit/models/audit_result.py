"""
점검 결과를 위한 Pydantic 모델.

사용 예:
    ```python
    from document_processing.audit.models import Issue, LayerResult, AuditReport

    # 이슈 생성
    issue = Issue(
        severity="warning",
        category="metadata",
        message="Missing doc_type field",
        file_path="/path/to/file.json",
    )

    # 레이어 결과 생성
    result = LayerResult(
        layer_name="sources",
        status="warning",
        total_items=100,
        issues=[issue],
        stats={"pdf_count": 50, "json_count": 50},
    )

    # 전체 리포트 생성
    report = AuditReport(layers=[result])
    print(report.model_dump_json(indent=2))
    ```
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class Severity(str, Enum):
    """이슈 심각도."""

    CRITICAL = "critical"  # 즉시 조치 필요
    WARNING = "warning"  # 주의 필요
    INFO = "info"  # 참고 사항


class Issue(BaseModel):
    """점검 중 발견된 이슈."""

    severity: Severity = Field(description="이슈 심각도")
    category: str = Field(description="이슈 카테고리 (예: metadata, content, sync)")
    message: str = Field(description="이슈 설명")
    file_path: str | None = Field(default=None, description="관련 파일 경로")
    details: dict[str, Any] | None = Field(default=None, description="추가 상세 정보")

    def __str__(self) -> str:
        """사람이 읽기 좋은 형식으로 출력."""
        prefix = {
            Severity.CRITICAL: "🔴",
            Severity.WARNING: "🟡",
            Severity.INFO: "🔵",
        }[self.severity]
        return f"{prefix} [{self.category}] {self.message}"


class LayerStats(BaseModel):
    """레이어별 통계 정보."""

    total_items: int = Field(default=0, description="전체 항목 수")
    checked_items: int = Field(default=0, description="점검된 항목 수")
    passed_items: int = Field(default=0, description="통과한 항목 수")
    failed_items: int = Field(default=0, description="실패한 항목 수")
    skipped_items: int = Field(default=0, description="건너뛴 항목 수")
    extra: dict[str, Any] = Field(default_factory=dict, description="추가 통계")

    @computed_field
    @property
    def pass_rate(self) -> float:
        """통과율 계산."""
        if self.checked_items == 0:
            return 0.0
        return round(self.passed_items / self.checked_items * 100, 2)


class LayerResult(BaseModel):
    """개별 레이어 점검 결과."""

    layer_name: str = Field(description="레이어 이름 (sources, chunks, indexes, quality, search)")
    status: Literal["pass", "warning", "fail"] = Field(description="점검 결과 상태")
    total_items: int = Field(default=0, description="점검 대상 항목 수")
    issues: list[Issue] = Field(default_factory=list, description="발견된 이슈 목록")
    stats: LayerStats = Field(default_factory=LayerStats, description="통계 정보")
    duration_seconds: float = Field(default=0.0, description="점검 소요 시간 (초)")

    @computed_field
    @property
    def critical_count(self) -> int:
        """치명적 이슈 개수."""
        return len([i for i in self.issues if i.severity == Severity.CRITICAL])

    @computed_field
    @property
    def warning_count(self) -> int:
        """경고 이슈 개수."""
        return len([i for i in self.issues if i.severity == Severity.WARNING])

    @computed_field
    @property
    def info_count(self) -> int:
        """정보 이슈 개수."""
        return len([i for i in self.issues if i.severity == Severity.INFO])

    def add_issue(
        self,
        severity: Severity | str,
        category: str,
        message: str,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """이슈 추가 헬퍼 메서드."""
        if isinstance(severity, str):
            severity = Severity(severity)
        self.issues.append(
            Issue(
                severity=severity,
                category=category,
                message=message,
                file_path=file_path,
                details=details,
            )
        )

    def determine_status(self) -> None:
        """이슈 기반으로 상태를 결정합니다."""
        if self.critical_count > 0:
            self.status = "fail"
        elif self.warning_count > 0:
            self.status = "warning"
        else:
            self.status = "pass"


class AuditReport(BaseModel):
    """전체 점검 리포트."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="점검 실행 시간 (UTC)",
    )
    layers: list[LayerResult] = Field(default_factory=list, description="레이어별 점검 결과")
    summary: dict[str, Any] = Field(default_factory=dict, description="요약 정보")
    recommendations: list[str] = Field(default_factory=list, description="권장 조치 사항")

    @computed_field
    @property
    def overall_status(self) -> Literal["pass", "warning", "fail"]:
        """전체 점검 상태."""
        if any(layer.status == "fail" for layer in self.layers):
            return "fail"
        if any(layer.status == "warning" for layer in self.layers):
            return "warning"
        return "pass"

    @computed_field
    @property
    def total_issues(self) -> int:
        """전체 이슈 개수."""
        return sum(len(layer.issues) for layer in self.layers)

    @computed_field
    @property
    def total_critical(self) -> int:
        """전체 치명적 이슈 개수."""
        return sum(layer.critical_count for layer in self.layers)

    @computed_field
    @property
    def total_warnings(self) -> int:
        """전체 경고 개수."""
        return sum(layer.warning_count for layer in self.layers)

    @computed_field
    @property
    def total_duration_seconds(self) -> float:
        """총 점검 소요 시간."""
        return sum(layer.duration_seconds for layer in self.layers)

    def add_layer(self, layer: LayerResult) -> None:
        """레이어 결과 추가."""
        self.layers.append(layer)

    def generate_recommendations(self) -> None:
        """이슈 기반으로 권장 사항을 자동 생성합니다."""
        recommendations: list[str] = []

        for layer in self.layers:
            if layer.critical_count > 0:
                recommendations.append(
                    f"🔴 [{layer.layer_name}] {layer.critical_count}개의 치명적 이슈 즉시 해결 필요"
                )

            # 레이어별 특정 권장사항
            if layer.layer_name == "sources":
                if any("손상" in i.message or "읽을 수 없" in i.message for i in layer.issues):
                    recommendations.append("📄 손상된 원본 파일을 확인하고 복구하세요")

            elif layer.layer_name == "chunks":
                if any("중복" in i.message for i in layer.issues):
                    recommendations.append("🔄 중복 청크를 정리하고 인덱스를 재구축하세요")
                if any("메타데이터" in i.message for i in layer.issues):
                    recommendations.append("📋 누락된 메타데이터 필드를 채우세요")

            elif layer.layer_name == "indexes":
                if any("동기화" in i.message or "불일치" in i.message for i in layer.issues):
                    recommendations.append("🔄 BM25와 Qdrant 인덱스를 재구축하세요")

            elif layer.layer_name == "quality":
                if any("저품질" in i.message for i in layer.issues):
                    recommendations.append("📊 저품질 데이터를 검토하고 필터링 기준을 조정하세요")

            elif layer.layer_name == "search":
                if any("낮은 관련성" in i.message for i in layer.issues):
                    recommendations.append("🔍 검색 파라미터(RRF 가중치, top-k)를 조정하세요")

        self.recommendations = list(dict.fromkeys(recommendations))  # 중복 제거

    def to_summary_dict(self) -> dict[str, Any]:
        """요약 정보를 딕셔너리로 반환."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status,
            "total_issues": self.total_issues,
            "total_critical": self.total_critical,
            "total_warnings": self.total_warnings,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "layers": {
                layer.layer_name: {
                    "status": layer.status,
                    "total_items": layer.total_items,
                    "issues": len(layer.issues),
                    "critical": layer.critical_count,
                    "warnings": layer.warning_count,
                }
                for layer in self.layers
            },
            "recommendations": self.recommendations,
        }


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    # 테스트용 예제
    layer = LayerResult(
        layer_name="sources",
        status="pass",
        total_items=100,
    )
    layer.add_issue(
        severity="warning",
        category="metadata",
        message="Missing doc_type field in 5 files",
        details={"affected_files": ["a.json", "b.json"]},
    )
    layer.add_issue(
        severity="critical",
        category="content",
        message="손상된 PDF 파일 발견",
        file_path="/path/to/broken.pdf",
    )
    layer.determine_status()

    report = AuditReport(layers=[layer])
    report.generate_recommendations()

    print("=== 점검 리포트 ===")
    print(f"상태: {report.overall_status}")
    print(f"전체 이슈: {report.total_issues}")
    print(f"치명적: {report.total_critical}, 경고: {report.total_warnings}")
    print("\n권장 사항:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("\n=== JSON 출력 ===")
    print(report.model_dump_json(indent=2, exclude_none=True))
