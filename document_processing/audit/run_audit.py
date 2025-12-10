"""
RAG 데이터 점검 CLI 진입점.

사용법:
    # 전체 점검 실행
    python -m document_processing.audit.run_audit

    # 특정 레이어만 점검
    python -m document_processing.audit.run_audit --layer sources
    python -m document_processing.audit.run_audit --layer chunks
    python -m document_processing.audit.run_audit --layer indexes
    python -m document_processing.audit.run_audit --layer quality
    python -m document_processing.audit.run_audit --layer search

    # 여러 레이어 선택
    python -m document_processing.audit.run_audit --layer sources --layer chunks

    # 리포트 형식 지정
    python -m document_processing.audit.run_audit --format json
    python -m document_processing.audit.run_audit --format html
    python -m document_processing.audit.run_audit --format all

    # 출력 디렉토리 지정
    python -m document_processing.audit.run_audit --output ./my_reports

    # 상세 모드
    python -m document_processing.audit.run_audit --verbose
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.auditors.source_auditor import SourceAuditor
from document_processing.audit.auditors.chunk_auditor import ChunkAuditor
from document_processing.audit.auditors.index_auditor import IndexAuditor
from document_processing.audit.auditors.quality_auditor import QualityAuditor
from document_processing.audit.auditors.search_auditor import SearchAuditor
from document_processing.audit.models.audit_result import AuditReport
from document_processing.audit.reporters.json_reporter import JSONReporter
from document_processing.audit.reporters.html_reporter import HTMLReporter


# 레이어 이름과 Auditor 클래스 매핑
LAYER_AUDITORS: dict[str, type[BaseAuditor]] = {
    "sources": SourceAuditor,
    "chunks": ChunkAuditor,
    "indexes": IndexAuditor,
    "quality": QualityAuditor,
    "search": SearchAuditor,
}

ALL_LAYERS = list(LAYER_AUDITORS.keys())


def setup_logging(verbose: bool = False) -> None:
    """로깅 설정."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="RAG 데이터 점검 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m document_processing.audit.run_audit                    # 전체 점검
  python -m document_processing.audit.run_audit --layer sources    # 원본 데이터만
  python -m document_processing.audit.run_audit --format html      # HTML 리포트
  python -m document_processing.audit.run_audit -v                 # 상세 모드
        """,
    )

    parser.add_argument(
        "--layer",
        "-l",
        action="append",
        choices=ALL_LAYERS,
        dest="layers",
        help="점검할 레이어 (여러 번 지정 가능, 기본: 전체)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "html", "all", "none"],
        default="all",
        help="리포트 출력 형식 (기본: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("audit_reports"),
        help="리포트 출력 디렉토리 (기본: audit_reports)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로깅 활성화",
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="프로젝트 루트 경로 (기본: 자동 감지)",
    )

    parser.add_argument(
        "--no-recommendations",
        action="store_true",
        help="권장 사항 생성 비활성화",
    )

    return parser.parse_args()


async def run_audit(
    layers: list[str],
    base_path: Path | None,
    verbose: bool,
) -> AuditReport:
    """점검 실행."""
    report = AuditReport()

    print("\n" + "=" * 60)
    print("🔍 RAG 데이터 점검 시작")
    print("=" * 60)
    print(f"📋 점검 레이어: {', '.join(layers)}")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for i, layer_name in enumerate(layers, 1):
        auditor_class = LAYER_AUDITORS.get(layer_name)
        if auditor_class is None:
            print(f"⚠️ 알 수 없는 레이어: {layer_name}")
            continue

        print(f"[{i}/{len(layers)}] {layer_name.upper()} 점검 중...", end=" ", flush=True)

        auditor = auditor_class(base_path=base_path, verbose=verbose)
        result = await auditor.audit()
        report.add_layer(result)

        # 상태 이모지
        status_emoji = {
            "pass": "✅",
            "warning": "⚠️",
            "fail": "❌",
        }.get(result.status, "❓")

        print(
            f"{status_emoji} {result.status.upper()} "
            f"({result.duration_seconds:.2f}초, 이슈 {len(result.issues)}개)"
        )

    return report


def generate_reports(
    report: AuditReport,
    output_dir: Path,
    report_format: Literal["json", "html", "all", "none"],
) -> list[Path]:
    """리포트 생성."""
    generated: list[Path] = []

    if report_format == "none":
        return generated

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if report_format in ("json", "all"):
        json_reporter = JSONReporter(output_dir=output_dir)
        json_path = json_reporter.generate(report, f"audit_report_{timestamp}.json")
        generated.append(json_path)

    if report_format in ("html", "all"):
        html_reporter = HTMLReporter(output_dir=output_dir)
        html_path = html_reporter.generate(report, f"audit_report_{timestamp}.html")
        generated.append(html_path)

    return generated


def print_summary(report: AuditReport) -> None:
    """점검 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("📊 점검 결과 요약")
    print("=" * 60)

    # 전체 상태
    status_display = {
        "pass": "✅ PASS (모든 점검 통과)",
        "warning": "⚠️ WARNING (경고 있음)",
        "fail": "❌ FAIL (치명적 문제 발견)",
    }
    print(f"\n전체 상태: {status_display.get(report.overall_status, report.overall_status)}")
    print(f"총 소요 시간: {report.total_duration_seconds:.2f}초")

    # 이슈 요약
    print(f"\n📋 이슈 요약:")
    print(f"  - 전체: {report.total_issues}개")
    print(f"  - 🔴 치명적: {report.total_critical}개")
    print(f"  - 🟡 경고: {report.total_warnings}개")

    # 레이어별 결과
    print(f"\n📁 레이어별 결과:")
    for layer in report.layers:
        status_emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(layer.status, "❓")
        print(
            f"  {status_emoji} {layer.layer_name}: "
            f"{layer.stats.pass_rate}% 통과, {len(layer.issues)}개 이슈"
        )

    # 권장 사항
    if report.recommendations:
        print(f"\n💡 권장 조치 사항:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")


def main() -> int:
    """메인 함수."""
    args = parse_args()
    setup_logging(args.verbose)

    # 점검할 레이어 결정
    layers = args.layers if args.layers else ALL_LAYERS

    try:
        # 점검 실행
        report = asyncio.run(
            run_audit(
                layers=layers,
                base_path=args.base_path,
                verbose=args.verbose,
            )
        )

        # 권장 사항 생성
        if not args.no_recommendations:
            report.generate_recommendations()

        # 결과 요약 출력
        print_summary(report)

        # 리포트 생성
        generated_files = generate_reports(report, args.output, args.format)

        if generated_files:
            print(f"\n📄 생성된 리포트:")
            for filepath in generated_files:
                print(f"  - {filepath}")

        # 종료 코드 결정
        if report.overall_status == "fail":
            return 1
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단됨")
        return 130

    except Exception as e:
        logging.exception("점검 중 오류 발생")
        print(f"\n❌ 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
