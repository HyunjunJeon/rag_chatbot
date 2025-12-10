"""
처리된 청크 품질 점검 모듈.

점검 대상:
- 청크 메타데이터 일관성
- 청크 콘텐츠 품질
- 중복 청크 탐지
- Slack QA 품질 평가 결과

점검 항목:
- doc_type, schema_version 등 필수 필드 존재
- 청크 길이 분포 (너무 짧거나 긴 청크 탐지)
- 빈 page_content 탐지
- 해시 기반 중복 탐지

사용 예:
    ```python
    from document_processing.audit.auditors.chunk_auditor import ChunkAuditor

    auditor = ChunkAuditor(verbose=True)
    result = await auditor.audit()
    print(result.model_dump_json(indent=2))
    ```
"""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.models.audit_result import LayerResult, LayerStats, Severity
from document_processing.common.versioning import SCHEMA_VERSION, PIPELINE_VERSION

logger = logging.getLogger(__name__)


class ChunkAuditor(BaseAuditor):
    """처리된 청크 품질 점검기."""

    layer_name = "chunks"

    # 점검 대상 디렉토리
    TRANSCRIPT_CHUNKS_DIR = "document_chunks/lecture_transcript_chunks"
    SLACK_QA_SCORED_DIR = "document_chunks/slack_qa_scored"

    # 청크 길이 임계값
    MIN_CHUNK_LENGTH = 50  # 50자 미만은 너무 짧음
    MAX_CHUNK_LENGTH = 5000  # 5000자 초과는 너무 김
    WARN_SHORT_LENGTH = 100  # 100자 미만은 경고

    # 필수 메타데이터 필드
    REQUIRED_METADATA_FIELDS = ["doc_type"]
    OPTIONAL_METADATA_FIELDS = ["schema_version", "pipeline_version", "source_file"]

    async def audit(self) -> LayerResult:
        """청크 품질 점검 실행."""
        result = self.create_result()
        self.start_timer()

        stats_extra: dict[str, Any] = {
            "length_distribution": {},
            "metadata_coverage": {},
            "duplicates": {},
            "by_doc_type": {},
        }

        all_chunks: list[dict[str, Any]] = []
        total_items = 0
        checked_items = 0
        passed_items = 0
        failed_items = 0

        # 1. 강의 녹음 청크 점검
        transcript_chunks = await self._load_transcript_chunks()
        all_chunks.extend(transcript_chunks)

        # 2. Slack QA 청크 점검
        slack_chunks = await self._load_slack_qa_chunks()
        all_chunks.extend(slack_chunks)

        total_items = len(all_chunks)
        stats_extra["total_chunks"] = total_items

        # 3. 메타데이터 일관성 점검
        metadata_stats = self._audit_metadata(all_chunks)
        stats_extra["metadata_coverage"] = metadata_stats
        failed_items += metadata_stats.get("missing_required", 0)

        # 4. 청크 길이 점검
        length_stats = self._audit_chunk_lengths(all_chunks)
        stats_extra["length_distribution"] = length_stats
        failed_items += length_stats.get("too_short", 0) + length_stats.get("too_long", 0)

        # 5. 중복 청크 탐지
        duplicate_stats = self._detect_duplicates(all_chunks)
        stats_extra["duplicates"] = duplicate_stats

        # 6. doc_type별 통계
        doc_type_stats = self._analyze_by_doc_type(all_chunks)
        stats_extra["by_doc_type"] = doc_type_stats

        # 7. 빈 콘텐츠 점검
        empty_count = self._count_empty_content(all_chunks)
        stats_extra["empty_content"] = empty_count
        failed_items += empty_count

        checked_items = total_items
        passed_items = checked_items - failed_items

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

    async def _load_transcript_chunks(self) -> list[dict[str, Any]]:
        """강의 녹음 청크 로드."""
        chunks_dir = self.resolve_path(self.TRANSCRIPT_CHUNKS_DIR)
        all_chunks: list[dict[str, Any]] = []

        if not chunks_dir.exists():
            self.add_issue(
                severity=Severity.WARNING,
                category="directory",
                message=f"강의 녹음 청크 디렉토리 없음: {chunks_dir}",
            )
            return all_chunks

        json_files = list(chunks_dir.glob("*_chunks.json"))
        self.log_progress(0, len(json_files), "Loading transcript chunks...")

        for i, json_path in enumerate(json_files):
            self.log_progress(i + 1, len(json_files), json_path.name)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 청크 추출 (리스트 또는 딕셔너리 형태 지원)
                chunks = data if isinstance(data, list) else data.get("chunks", [])

                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk["_source_file"] = str(json_path)
                        chunk["_doc_type"] = "lecture_transcript"
                        all_chunks.append(chunk)

            except Exception as e:
                self.add_issue(
                    severity=Severity.WARNING,
                    category="parse",
                    message=f"청크 파일 로드 실패: {e}",
                    file_path=str(json_path),
                )

        return all_chunks

    async def _load_slack_qa_chunks(self) -> list[dict[str, Any]]:
        """Slack QA 청크 로드."""
        chunks_dir = self.resolve_path(self.SLACK_QA_SCORED_DIR)
        all_chunks: list[dict[str, Any]] = []

        if not chunks_dir.exists():
            self.add_issue(
                severity=Severity.WARNING,
                category="directory",
                message=f"Slack QA 청크 디렉토리 없음: {chunks_dir}",
            )
            return all_chunks

        json_files = list(chunks_dir.glob("*.json"))
        self.log_progress(0, len(json_files), "Loading Slack QA chunks...")

        for i, json_path in enumerate(json_files):
            self.log_progress(i + 1, len(json_files), json_path.name)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # qa_pairs 배열 처리
                qa_pairs = data.get("qa_pairs", [])
                for qa in qa_pairs:
                    chunk = {
                        "page_content": self._format_qa_content(qa),
                        "metadata": qa.get("evaluation", {}),
                        "_source_file": str(json_path),
                        "_doc_type": "slack_qa",
                    }
                    all_chunks.append(chunk)

            except Exception as e:
                self.add_issue(
                    severity=Severity.WARNING,
                    category="parse",
                    message=f"Slack QA 파일 로드 실패: {e}",
                    file_path=str(json_path),
                )

        return all_chunks

    def _format_qa_content(self, qa: dict) -> str:
        """Q&A 쌍을 텍스트로 포맷."""
        question = qa.get("question", {})
        question_text = question.get("text", "") if isinstance(question, dict) else str(question)

        answers = qa.get("answers", [])
        answer_texts = []
        for ans in answers:
            if isinstance(ans, dict):
                answer_texts.append(ans.get("text", ""))
            else:
                answer_texts.append(str(ans))

        return f"Q: {question_text}\nA: {' '.join(answer_texts)}"

    def _audit_metadata(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """메타데이터 일관성 점검."""
        stats = {
            "total": len(chunks),
            "missing_required": 0,
            "missing_optional": {},
            "version_mismatch": 0,
            "field_coverage": {},
        }

        field_counts: Counter[str] = Counter()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # 필수 필드 확인
            for field in self.REQUIRED_METADATA_FIELDS:
                if field in metadata or f"_{field}" in chunk:
                    field_counts[field] += 1
                else:
                    stats["missing_required"] += 1

            # 선택 필드 확인
            for field in self.OPTIONAL_METADATA_FIELDS:
                if field in metadata:
                    field_counts[field] += 1

            # 버전 일치 확인
            schema_ver = metadata.get("schema_version", "")
            pipeline_ver = metadata.get("pipeline_version", "")

            if schema_ver and schema_ver != SCHEMA_VERSION:
                stats["version_mismatch"] += 1
            if pipeline_ver and pipeline_ver != PIPELINE_VERSION:
                stats["version_mismatch"] += 1

        # 필드별 커버리지 계산
        total = len(chunks) if chunks else 1
        for field, count in field_counts.items():
            stats["field_coverage"][field] = round(count / total * 100, 2)

        # 이슈 추가
        if stats["missing_required"] > 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="metadata",
                message=f"필수 메타데이터 필드 누락: {stats['missing_required']}개 청크",
                details={"missing_fields": self.REQUIRED_METADATA_FIELDS},
            )

        if stats["version_mismatch"] > 0:
            self.add_issue(
                severity=Severity.INFO,
                category="metadata",
                message=f"버전 불일치 청크: {stats['version_mismatch']}개 (현재: schema={SCHEMA_VERSION}, pipeline={PIPELINE_VERSION})",
            )

        return stats

    def _audit_chunk_lengths(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """청크 길이 분포 분석."""
        stats = {
            "total": len(chunks),
            "too_short": 0,
            "too_long": 0,
            "warning_short": 0,
            "min_length": float("inf"),
            "max_length": 0,
            "avg_length": 0,
            "distribution": {
                "0-50": 0,
                "50-100": 0,
                "100-500": 0,
                "500-1000": 0,
                "1000-3000": 0,
                "3000-5000": 0,
                "5000+": 0,
            },
        }

        total_length = 0
        short_chunks: list[str] = []
        long_chunks: list[str] = []

        for chunk in chunks:
            content = chunk.get("page_content", "") or chunk.get("content", "")
            length = len(content)
            total_length += length

            # 최소/최대
            stats["min_length"] = min(stats["min_length"], length)
            stats["max_length"] = max(stats["max_length"], length)

            # 분포
            if length < 50:
                stats["distribution"]["0-50"] += 1
            elif length < 100:
                stats["distribution"]["50-100"] += 1
            elif length < 500:
                stats["distribution"]["100-500"] += 1
            elif length < 1000:
                stats["distribution"]["500-1000"] += 1
            elif length < 3000:
                stats["distribution"]["1000-3000"] += 1
            elif length < 5000:
                stats["distribution"]["3000-5000"] += 1
            else:
                stats["distribution"]["5000+"] += 1

            # 임계값 확인
            if length < self.MIN_CHUNK_LENGTH:
                stats["too_short"] += 1
                short_chunks.append(chunk.get("_source_file", "unknown"))
            elif length < self.WARN_SHORT_LENGTH:
                stats["warning_short"] += 1

            if length > self.MAX_CHUNK_LENGTH:
                stats["too_long"] += 1
                long_chunks.append(chunk.get("_source_file", "unknown"))

        # 평균 계산
        if chunks:
            stats["avg_length"] = round(total_length / len(chunks), 2)

        # 무한대 처리
        if stats["min_length"] == float("inf"):
            stats["min_length"] = 0

        # 이슈 추가
        if stats["too_short"] > 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="length",
                message=f"너무 짧은 청크 ({self.MIN_CHUNK_LENGTH}자 미만): {stats['too_short']}개",
                details={"sample_files": list(set(short_chunks))[:5]},
            )

        if stats["too_long"] > 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="length",
                message=f"너무 긴 청크 ({self.MAX_CHUNK_LENGTH}자 초과): {stats['too_long']}개",
                details={"sample_files": list(set(long_chunks))[:5]},
            )

        return stats

    def _detect_duplicates(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """중복 청크 탐지 (해시 기반)."""
        stats = {
            "total": len(chunks),
            "unique": 0,
            "duplicates": 0,
            "duplicate_groups": 0,
        }

        # 콘텐츠 해시 계산
        hash_to_chunks: dict[str, list[int]] = {}

        for i, chunk in enumerate(chunks):
            content = chunk.get("page_content", "") or chunk.get("content", "")
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            if content_hash not in hash_to_chunks:
                hash_to_chunks[content_hash] = []
            hash_to_chunks[content_hash].append(i)

        # 중복 카운트
        for indices in hash_to_chunks.values():
            if len(indices) == 1:
                stats["unique"] += 1
            else:
                stats["duplicate_groups"] += 1
                stats["duplicates"] += len(indices) - 1  # 원본 제외

        # 이슈 추가
        if stats["duplicates"] > 0:
            duplicate_rate = round(stats["duplicates"] / len(chunks) * 100, 2)
            self.add_issue(
                severity=Severity.WARNING if duplicate_rate > 5 else Severity.INFO,
                category="duplicate",
                message=f"중복 청크 발견: {stats['duplicates']}개 ({duplicate_rate}%)",
                details={"duplicate_groups": stats["duplicate_groups"]},
            )

        return stats

    def _analyze_by_doc_type(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """doc_type별 통계 분석."""
        stats: dict[str, dict[str, Any]] = {}

        for chunk in chunks:
            doc_type = (
                chunk.get("metadata", {}).get("doc_type")
                or chunk.get("_doc_type")
                or "unknown"
            )

            if doc_type not in stats:
                stats[doc_type] = {"count": 0, "total_length": 0}

            stats[doc_type]["count"] += 1

            content = chunk.get("page_content", "") or chunk.get("content", "")
            stats[doc_type]["total_length"] += len(content)

        # 평균 길이 계산
        for doc_type, data in stats.items():
            if data["count"] > 0:
                data["avg_length"] = round(data["total_length"] / data["count"], 2)
            else:
                data["avg_length"] = 0

        return stats

    def _count_empty_content(self, chunks: list[dict[str, Any]]) -> int:
        """빈 콘텐츠 청크 카운트."""
        empty_count = 0

        for chunk in chunks:
            content = chunk.get("page_content", "") or chunk.get("content", "")
            if not content or not content.strip():
                empty_count += 1

        if empty_count > 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="content",
                message=f"빈 콘텐츠 청크: {empty_count}개",
            )

        return empty_count


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = ChunkAuditor(verbose=True)
        result = await auditor.audit()

        print("\n" + "=" * 60)
        print("청크 품질 점검 결과")
        print("=" * 60)
        print(f"상태: {result.status}")
        print(f"전체 청크: {result.total_items}")
        print(f"이슈: {len(result.issues)}개")
        print(f"소요 시간: {result.duration_seconds:.2f}초")

        print("\n📊 통계:")
        for key, value in result.stats.extra.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        if result.issues:
            print("\n⚠️ 발견된 이슈:")
            for issue in result.issues:
                print(f"  {issue}")

    asyncio.run(main())
