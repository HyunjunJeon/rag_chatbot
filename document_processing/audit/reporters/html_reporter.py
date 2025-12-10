"""
HTML 형식 상세 리포트 생성기.

점검 결과를 시각적으로 보기 좋은 HTML 파일로 생성합니다.

사용 예:
    ```python
    from document_processing.audit.reporters.html_reporter import HTMLReporter
    from document_processing.audit.models import AuditReport

    reporter = HTMLReporter(output_dir="audit_reports")
    filepath = reporter.generate(report)
    print(f"리포트 저장됨: {filepath}")
    ```
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from document_processing.audit.models.audit_result import AuditReport, Severity

logger = logging.getLogger(__name__)


class HTMLReporter:
    """HTML 형식 리포트 생성기."""

    def __init__(
        self,
        output_dir: str | Path = "audit_reports",
    ):
        """
        Args:
            output_dir: 리포트 출력 디렉토리
        """
        self.output_dir = Path(output_dir)

    def generate(
        self,
        report: AuditReport,
        filename: str | None = None,
    ) -> Path:
        """
        리포트를 HTML 파일로 생성합니다.

        Args:
            report: 점검 결과 리포트
            filename: 출력 파일명 (기본: audit_report_YYYYMMDD_HHMMSS.html)

        Returns:
            생성된 파일 경로
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.html"

        filepath = self.output_dir / filename

        # HTML 생성
        html_content = self._generate_html(report)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML 리포트 생성됨: {filepath}")
        return filepath

    def _generate_html(self, report: AuditReport) -> str:
        """HTML 콘텐츠 생성."""
        status_colors = {
            "pass": "#10b981",
            "warning": "#f59e0b",
            "fail": "#ef4444",
        }

        severity_colors = {
            "critical": "#ef4444",
            "warning": "#f59e0b",
            "info": "#3b82f6",
        }

        overall_color = status_colors.get(report.overall_status, "#6b7280")

        # 레이어별 HTML 생성
        layers_html = ""
        for layer in report.layers:
            layer_color = status_colors.get(layer.status, "#6b7280")

            # 이슈 목록
            issues_html = ""
            if layer.issues:
                issues_html = "<div class='issues'><h4>발견된 이슈</h4><ul>"
                for issue in layer.issues:
                    sev_color = severity_colors.get(issue.severity.value, "#6b7280")
                    issues_html += f"""
                    <li class='issue'>
                        <span class='severity' style='background-color: {sev_color};'>{issue.severity.value.upper()}</span>
                        <span class='category'>[{issue.category}]</span>
                        <span class='message'>{issue.message}</span>
                        {f"<span class='file'>{issue.file_path}</span>" if issue.file_path else ""}
                    </li>
                    """
                issues_html += "</ul></div>"

            # 통계 테이블
            stats_html = self._generate_stats_table(layer.stats.extra)

            layers_html += f"""
            <div class='layer'>
                <div class='layer-header'>
                    <h3>{layer.layer_name.upper()}</h3>
                    <span class='status' style='background-color: {layer_color};'>{layer.status.upper()}</span>
                </div>
                <div class='layer-meta'>
                    <span>📊 총 {layer.total_items}개 항목</span>
                    <span>⏱️ {layer.duration_seconds:.2f}초</span>
                    <span>✓ {layer.stats.pass_rate}% 통과</span>
                </div>
                <div class='layer-issues'>
                    <span class='critical'>🔴 {layer.critical_count}</span>
                    <span class='warning'>🟡 {layer.warning_count}</span>
                    <span class='info'>🔵 {layer.info_count}</span>
                </div>
                {stats_html}
                {issues_html}
            </div>
            """

        # 권장 사항
        recommendations_html = ""
        if report.recommendations:
            recommendations_html = "<div class='recommendations'><h3>권장 조치 사항</h3><ul>"
            for rec in report.recommendations:
                recommendations_html += f"<li>{rec}</li>"
            recommendations_html += "</ul></div>"

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 데이터 점검 리포트</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f3f4f6;
            color: #1f2937;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 16px;
        }}
        .header-meta {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            color: #6b7280;
            font-size: 14px;
        }}
        .overall-status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 16px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .summary-item {{
            background: #f9fafb;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 28px;
            font-weight: bold;
            color: #111827;
        }}
        .summary-label {{
            font-size: 12px;
            color: #6b7280;
            margin-top: 4px;
        }}
        .layer {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .layer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .layer-header h3 {{
            font-size: 18px;
        }}
        .status {{
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .layer-meta {{
            display: flex;
            gap: 16px;
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 12px;
        }}
        .layer-issues {{
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }}
        .layer-issues span {{
            font-size: 14px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        .stats-table th, .stats-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        .stats-table th {{
            background: #f9fafb;
            font-weight: 600;
        }}
        .issues {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #e5e7eb;
        }}
        .issues h4 {{
            font-size: 14px;
            margin-bottom: 8px;
            color: #374151;
        }}
        .issues ul {{
            list-style: none;
        }}
        .issue {{
            padding: 8px 0;
            border-bottom: 1px solid #f3f4f6;
            font-size: 13px;
        }}
        .severity {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
            font-size: 10px;
            font-weight: bold;
            margin-right: 8px;
        }}
        .category {{
            color: #6b7280;
            margin-right: 8px;
        }}
        .file {{
            display: block;
            color: #9ca3af;
            font-size: 11px;
            margin-top: 4px;
            font-family: monospace;
        }}
        .recommendations {{
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
        }}
        .recommendations h3 {{
            color: #92400e;
            margin-bottom: 12px;
        }}
        .recommendations ul {{
            margin-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 8px;
            color: #78350f;
        }}
        .footer {{
            text-align: center;
            padding: 24px;
            color: #9ca3af;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 RAG 데이터 점검 리포트</h1>
            <div class="overall-status" style="background-color: {overall_color};">
                {report.overall_status.upper()}
            </div>
            <div class="header-meta">
                <span>📅 {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC</span>
                <span>⏱️ 총 {report.total_duration_seconds:.2f}초 소요</span>
                <span>📋 {len(report.layers)}개 레이어 점검</span>
            </div>
            <div class="summary">
                <div class="summary-item">
                    <div class="summary-value">{report.total_issues}</div>
                    <div class="summary-label">전체 이슈</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value" style="color: #ef4444;">{report.total_critical}</div>
                    <div class="summary-label">치명적</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value" style="color: #f59e0b;">{report.total_warnings}</div>
                    <div class="summary-label">경고</div>
                </div>
            </div>
        </div>

        {layers_html}

        {recommendations_html}

        <div class="footer">
            Generated by RAG Data Audit Tool
        </div>
    </div>
</body>
</html>
"""

    def _generate_stats_table(self, extra: dict[str, Any]) -> str:
        """통계 테이블 HTML 생성."""
        if not extra:
            return ""

        rows = ""
        for key, value in extra.items():
            if isinstance(value, dict):
                # 중첩 딕셔너리는 JSON 형태로 표시
                formatted = "<br>".join(f"{k}: {v}" for k, v in list(value.items())[:5])
                if len(value) > 5:
                    formatted += f"<br>... (+{len(value) - 5} more)"
                rows += f"<tr><td>{key}</td><td>{formatted}</td></tr>"
            else:
                rows += f"<tr><td>{key}</td><td>{value}</td></tr>"

        return f"""
        <table class="stats-table">
            <thead>
                <tr><th>항목</th><th>값</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    from document_processing.audit.models.audit_result import (
        AuditReport,
        LayerResult,
        LayerStats,
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
            extra={"metric1": 42, "metric2": {"a": 1, "b": 2}},
        ),
    )
    layer.add_issue(
        severity=Severity.WARNING,
        category="test",
        message="테스트 경고 이슈입니다",
    )
    layer.add_issue(
        severity=Severity.CRITICAL,
        category="test",
        message="테스트 치명적 이슈입니다",
        file_path="/path/to/file.json",
    )

    report = AuditReport(layers=[layer])
    report.generate_recommendations()

    reporter = HTMLReporter(output_dir="/tmp/audit_test")
    filepath = reporter.generate(report)
    print(f"HTML 리포트 생성됨: {filepath}")
