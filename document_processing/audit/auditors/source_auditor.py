"""
원본 데이터 소스 점검 모듈.

점검 대상:
- PDF 파일 (original_documents/lecture_content/)
- Slack Q&A JSON (original_documents/qa_dataset_from_slack/)
- 강의 녹음 청크 (document_chunks/lecture_transcript_chunks/)

점검 항목:
- 파일 존재 및 읽기 가능 여부
- 파일 크기 및 페이지 수 통계
- 손상된 파일 탐지
- JSON 파싱 및 필수 필드 검증

사용 예:
    ```python
    from document_processing.audit.auditors.source_auditor import SourceAuditor

    auditor = SourceAuditor(verbose=True)
    result = await auditor.audit()
    print(result.model_dump_json(indent=2))
    ```
"""

import json
import logging
from pathlib import Path
from typing import Any

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.models.audit_result import LayerResult, LayerStats, Severity

logger = logging.getLogger(__name__)


class SourceAuditor(BaseAuditor):
    """원본 데이터 소스 점검기."""

    layer_name = "sources"

    # 점검 대상 디렉토리 (base_path 기준 상대 경로)
    PDF_DIR = "original_documents/lecture_content"
    SLACK_QA_DIR = "original_documents/qa_dataset_from_slack"
    TRANSCRIPT_DIR = "document_chunks/lecture_transcript_chunks"

    # 기대하는 최소 파일 수 (이 수보다 적으면 경고)
    MIN_PDF_COUNT = 100
    MIN_SLACK_JSON_COUNT = 500
    MIN_TRANSCRIPT_COUNT = 50

    async def audit(self) -> LayerResult:
        """원본 데이터 소스 점검 실행."""
        result = self.create_result()
        self.start_timer()

        stats_extra: dict[str, Any] = {}
        total_items = 0
        checked_items = 0
        passed_items = 0
        failed_items = 0

        # 1. PDF 파일 점검
        pdf_stats = await self._audit_pdf_files()
        stats_extra["pdf"] = pdf_stats
        total_items += pdf_stats.get("total", 0)
        checked_items += pdf_stats.get("checked", 0)
        passed_items += pdf_stats.get("passed", 0)
        failed_items += pdf_stats.get("failed", 0)

        # 2. Slack Q&A JSON 점검
        slack_stats = await self._audit_slack_json()
        stats_extra["slack_qa"] = slack_stats
        total_items += slack_stats.get("total", 0)
        checked_items += slack_stats.get("checked", 0)
        passed_items += slack_stats.get("passed", 0)
        failed_items += slack_stats.get("failed", 0)

        # 3. 강의 녹음 청크 점검
        transcript_stats = await self._audit_transcript_chunks()
        stats_extra["lecture_transcript"] = transcript_stats
        total_items += transcript_stats.get("total", 0)
        checked_items += transcript_stats.get("checked", 0)
        passed_items += transcript_stats.get("passed", 0)
        failed_items += transcript_stats.get("failed", 0)

        # 통계 업데이트
        result.total_items = total_items
        result.stats = LayerStats(
            total_items=total_items,
            checked_items=checked_items,
            passed_items=passed_items,
            failed_items=failed_items,
            extra=stats_extra,
        )

        return self.finalize_result()

    async def _audit_pdf_files(self) -> dict[str, Any]:
        """PDF 파일 점검."""
        pdf_dir = self.resolve_path(self.PDF_DIR)
        stats: dict[str, Any] = {
            "total": 0,
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "total_size_bytes": 0,
            "by_course": {},
        }

        if not pdf_dir.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="directory",
                message=f"PDF 디렉토리가 존재하지 않습니다: {pdf_dir}",
            )
            return stats

        # PDF 파일 목록 수집
        pdf_files = list(pdf_dir.rglob("*.pdf"))
        stats["total"] = len(pdf_files)

        if len(pdf_files) < self.MIN_PDF_COUNT:
            self.add_issue(
                severity=Severity.WARNING,
                category="count",
                message=f"PDF 파일 수가 예상보다 적습니다: {len(pdf_files)}개 (최소 {self.MIN_PDF_COUNT}개 기대)",
            )

        # 각 PDF 파일 점검
        for i, pdf_path in enumerate(pdf_files):
            self.log_progress(i + 1, len(pdf_files), pdf_path.name)

            stats["checked"] += 1

            # 파일 크기 확인
            try:
                file_size = pdf_path.stat().st_size
                stats["total_size_bytes"] += file_size

                if file_size == 0:
                    self.add_issue(
                        severity=Severity.CRITICAL,
                        category="content",
                        message="빈 PDF 파일",
                        file_path=str(pdf_path),
                    )
                    stats["failed"] += 1
                    continue

                # 코스별 통계
                course = pdf_path.parent.name
                if course not in stats["by_course"]:
                    stats["by_course"][course] = {"count": 0, "size_bytes": 0}
                stats["by_course"][course]["count"] += 1
                stats["by_course"][course]["size_bytes"] += file_size

                # PDF 헤더 검증 (간단한 무결성 체크)
                is_valid = await self._validate_pdf_header(pdf_path)
                if not is_valid:
                    self.add_issue(
                        severity=Severity.WARNING,
                        category="content",
                        message="손상된 PDF 파일 (헤더 검증 실패)",
                        file_path=str(pdf_path),
                    )
                    stats["failed"] += 1
                else:
                    stats["passed"] += 1

            except OSError as e:
                self.add_issue(
                    severity=Severity.CRITICAL,
                    category="access",
                    message=f"파일 접근 오류: {e}",
                    file_path=str(pdf_path),
                )
                stats["failed"] += 1

        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        return stats

    async def _validate_pdf_header(self, pdf_path: Path) -> bool:
        """PDF 파일 헤더 검증 (간단한 무결성 체크)."""

        def check_header() -> bool:
            try:
                with open(pdf_path, "rb") as f:
                    header = f.read(8)
                    # PDF 파일은 %PDF-로 시작
                    return header.startswith(b"%PDF-")
            except Exception:
                return False

        return await self.run_in_executor(check_header)

    async def _audit_slack_json(self) -> dict[str, Any]:
        """Slack Q&A JSON 파일 점검."""
        slack_dir = self.resolve_path(self.SLACK_QA_DIR)
        stats: dict[str, Any] = {
            "total": 0,
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "total_qa_pairs": 0,
            "by_course": {},
            "by_generation": {},
        }

        if not slack_dir.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="directory",
                message=f"Slack Q&A 디렉토리가 존재하지 않습니다: {slack_dir}",
            )
            return stats

        # JSON 파일 목록 수집
        json_files = list(slack_dir.rglob("*.json"))
        stats["total"] = len(json_files)

        if len(json_files) < self.MIN_SLACK_JSON_COUNT:
            self.add_issue(
                severity=Severity.WARNING,
                category="count",
                message=f"Slack JSON 파일 수가 예상보다 적습니다: {len(json_files)}개 (최소 {self.MIN_SLACK_JSON_COUNT}개 기대)",
            )

        # 각 JSON 파일 점검
        for i, json_path in enumerate(json_files):
            self.log_progress(i + 1, len(json_files), json_path.name)

            stats["checked"] += 1

            validation_result = await self._validate_slack_json(json_path)
            if validation_result["valid"]:
                stats["passed"] += 1
                stats["total_qa_pairs"] += validation_result.get("qa_pairs", 0)

                # 코스별/기수별 통계
                course = validation_result.get("course", "unknown")
                generation = validation_result.get("generation", "unknown")

                stats["by_course"][course] = stats["by_course"].get(course, 0) + 1
                stats["by_generation"][generation] = stats["by_generation"].get(generation, 0) + 1
            else:
                stats["failed"] += 1
                self.add_issue(
                    severity=validation_result.get("severity", Severity.WARNING),
                    category="content",
                    message=validation_result.get("error", "Unknown error"),
                    file_path=str(json_path),
                )

        return stats

    async def _validate_slack_json(self, json_path: Path) -> dict[str, Any]:
        """Slack Q&A JSON 파일 검증."""

        def validate() -> dict[str, Any]:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 기본 구조 검증
                if not isinstance(data, dict):
                    return {
                        "valid": False,
                        "error": "JSON이 딕셔너리가 아님",
                        "severity": Severity.WARNING,
                    }

                # 필수 필드 검증 (다양한 형식 지원)
                qa_pairs = 0

                # 형식 1: qa_pairs 배열
                if "qa_pairs" in data:
                    qa_pairs = len(data["qa_pairs"])
                    for qa in data["qa_pairs"]:
                        if "question" not in qa:
                            return {
                                "valid": False,
                                "error": "Q&A에 question 필드 누락",
                                "severity": Severity.WARNING,
                            }

                # 형식 2: 직접 question/answers
                elif "question" in data:
                    qa_pairs = 1

                # 형식 3: messages 배열 (원본 Slack 형식)
                elif "messages" in data:
                    qa_pairs = len(data.get("messages", []))

                # 코스 및 기수 추출 (경로에서)
                parts = json_path.parts
                course = "unknown"
                generation = "unknown"

                for part in parts:
                    if part.startswith("level") or part in ["bot_common", "general"]:
                        course = part
                    elif part.isdigit():
                        generation = part

                return {
                    "valid": True,
                    "qa_pairs": qa_pairs,
                    "course": course,
                    "generation": generation,
                }

            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "error": f"JSON 파싱 오류: {e}",
                    "severity": Severity.CRITICAL,
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"파일 읽기 오류: {e}",
                    "severity": Severity.CRITICAL,
                }

        return await self.run_in_executor(validate)

    async def _audit_transcript_chunks(self) -> dict[str, Any]:
        """강의 녹음 청크 파일 점검."""
        transcript_dir = self.resolve_path(self.TRANSCRIPT_DIR)
        stats: dict[str, Any] = {
            "total": 0,
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "total_chunks": 0,
            "by_course": {},
        }

        if not transcript_dir.exists():
            self.add_issue(
                severity=Severity.WARNING,
                category="directory",
                message=f"강의 녹음 청크 디렉토리가 존재하지 않습니다: {transcript_dir}",
            )
            return stats

        # JSON 파일 목록 수집
        json_files = list(transcript_dir.glob("*_chunks.json"))
        stats["total"] = len(json_files)

        if len(json_files) < self.MIN_TRANSCRIPT_COUNT:
            self.add_issue(
                severity=Severity.WARNING,
                category="count",
                message=f"강의 녹음 청크 파일 수가 예상보다 적습니다: {len(json_files)}개 (최소 {self.MIN_TRANSCRIPT_COUNT}개 기대)",
            )

        # 각 청크 파일 점검
        for i, json_path in enumerate(json_files):
            self.log_progress(i + 1, len(json_files), json_path.name)

            stats["checked"] += 1

            validation_result = await self._validate_transcript_chunk(json_path)
            if validation_result["valid"]:
                stats["passed"] += 1
                stats["total_chunks"] += validation_result.get("chunk_count", 0)

                # 코스별 통계 (파일명에서 추출)
                course = self._extract_course_from_filename(json_path.stem)
                stats["by_course"][course] = stats["by_course"].get(course, 0) + 1
            else:
                stats["failed"] += 1
                self.add_issue(
                    severity=validation_result.get("severity", Severity.WARNING),
                    category="content",
                    message=validation_result.get("error", "Unknown error"),
                    file_path=str(json_path),
                )

        return stats

    async def _validate_transcript_chunk(self, json_path: Path) -> dict[str, Any]:
        """강의 녹음 청크 파일 검증."""

        def validate() -> dict[str, Any]:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 배열 또는 chunks 필드 확인
                chunks = data if isinstance(data, list) else data.get("chunks", [])

                if not chunks:
                    return {
                        "valid": False,
                        "error": "청크가 비어있음",
                        "severity": Severity.WARNING,
                    }

                # 청크 내용 검증
                for chunk in chunks[:5]:  # 처음 5개만 샘플 검증
                    if isinstance(chunk, dict):
                        if not chunk.get("page_content") and not chunk.get("content"):
                            return {
                                "valid": False,
                                "error": "청크에 콘텐츠가 없음",
                                "severity": Severity.WARNING,
                            }

                return {
                    "valid": True,
                    "chunk_count": len(chunks),
                }

            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "error": f"JSON 파싱 오류: {e}",
                    "severity": Severity.CRITICAL,
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"파일 읽기 오류: {e}",
                    "severity": Severity.CRITICAL,
                }

        return await self.run_in_executor(validate)

    def _extract_course_from_filename(self, filename: str) -> str:
        """파일명에서 코스명 추출."""
        # 예: "[NLP 이론] (8강) Transformer 2_chunks" -> "NLP 이론"
        if filename.startswith("["):
            end_bracket = filename.find("]")
            if end_bracket > 0:
                return filename[1:end_bracket]

        # 예: "(1강) PyTorch Intro_chunks" -> "general"
        return "general"


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = SourceAuditor(verbose=True)
        result = await auditor.audit()

        print("\n" + "=" * 60)
        print("원본 데이터 소스 점검 결과")
        print("=" * 60)
        print(f"상태: {result.status}")
        print(f"전체 항목: {result.total_items}")
        print(f"이슈: {len(result.issues)}개")
        print(f"소요 시간: {result.duration_seconds:.2f}초")

        print("\n📊 통계:")
        for key, value in result.stats.extra.items():
            print(f"  {key}: {value}")

        if result.issues:
            print("\n⚠️ 발견된 이슈:")
            for issue in result.issues[:10]:  # 처음 10개만 출력
                print(f"  {issue}")

    asyncio.run(main())
