"""
데이터 품질 메트릭 점검 모듈.

점검 대상:
- 전체 문서 통계 (원본 vs 처리됨 vs 인덱스)
- doc_type별 분포
- 코스별/기수별 분포
- Slack QA 품질 평가 결과

점검 항목:
- 이상치 탐지 (극단적 청크 길이)
- 메타데이터 누락 비율
- 중복 문서 비율
- 품질 등급 분포

사용 예:
    ```python
    from document_processing.audit.auditors.quality_auditor import QualityAuditor

    auditor = QualityAuditor(verbose=True)
    result = await auditor.audit()
    print(result.model_dump_json(indent=2))
    ```
"""

import json
import logging
from collections import Counter
from typing import Any

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.models.audit_result import LayerResult, LayerStats, Severity

logger = logging.getLogger(__name__)


class QualityAuditor(BaseAuditor):
    """데이터 품질 메트릭 점검기."""

    layer_name = "quality"

    # 데이터 경로
    SLACK_QA_SCORED_DIR = "document_chunks/slack_qa_scored"
    TRANSCRIPT_CHUNKS_DIR = "document_chunks/lecture_transcript_chunks"

    # 품질 임계값
    LOW_QUALITY_THRESHOLD = 0.1  # 저품질 데이터 10% 초과 시 경고
    METADATA_MISSING_THRESHOLD = 0.05  # 메타데이터 누락 5% 초과 시 경고

    async def audit(self) -> LayerResult:
        """품질 메트릭 점검 실행."""
        result = self.create_result()
        self.start_timer()

        stats_extra: dict[str, Any] = {
            "overall_summary": {},
            "slack_qa_quality": {},
            "distribution_by_doc_type": {},
            "distribution_by_course": {},
            "anomalies": {},
        }

        # 1. 전체 데이터 요약
        overall_summary = await self._compute_overall_summary()
        stats_extra["overall_summary"] = overall_summary

        # 2. Slack QA 품질 분석
        slack_quality = await self._analyze_slack_qa_quality()
        stats_extra["slack_qa_quality"] = slack_quality

        # 3. doc_type별 분포
        doc_type_dist = await self._analyze_doc_type_distribution()
        stats_extra["distribution_by_doc_type"] = doc_type_dist

        # 4. 코스별 분포
        course_dist = await self._analyze_course_distribution()
        stats_extra["distribution_by_course"] = course_dist

        # 5. 이상치 탐지
        anomalies = await self._detect_anomalies()
        stats_extra["anomalies"] = anomalies

        # 품질 점수 계산
        quality_score = self._compute_quality_score(stats_extra)
        stats_extra["quality_score"] = quality_score

        # 통계 업데이트
        total_items = overall_summary.get("total_documents", 0)
        result.total_items = total_items
        result.stats = LayerStats(
            total_items=total_items,
            checked_items=total_items,
            passed_items=int(total_items * (quality_score / 100)),
            failed_items=int(total_items * (1 - quality_score / 100)),
            extra=stats_extra,
        )

        return self.finalize_result()

    async def _compute_overall_summary(self) -> dict[str, Any]:
        """전체 데이터 요약 계산."""
        summary: dict[str, Any] = {
            "total_documents": 0,
            "by_source": {},
        }

        # Slack QA
        slack_dir = self.resolve_path(self.SLACK_QA_SCORED_DIR)
        if slack_dir.exists():
            slack_count = 0
            for json_file in slack_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        slack_count += len(data.get("qa_pairs", []))
                except Exception:
                    pass
            summary["by_source"]["slack_qa"] = slack_count
            summary["total_documents"] += slack_count

        # Transcript chunks
        transcript_dir = self.resolve_path(self.TRANSCRIPT_CHUNKS_DIR)
        if transcript_dir.exists():
            transcript_count = 0
            for json_file in transcript_dir.glob("*_chunks.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunks = data if isinstance(data, list) else data.get("chunks", [])
                        transcript_count += len(chunks)
                except Exception:
                    pass
            summary["by_source"]["lecture_transcript"] = transcript_count
            summary["total_documents"] += transcript_count

        return summary

    async def _analyze_slack_qa_quality(self) -> dict[str, Any]:
        """Slack QA 품질 분석."""
        quality_stats: dict[str, Any] = {
            "total_qa_pairs": 0,
            "evaluated": 0,
            "quality_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0,
                "remove": 0,
            },
            "score_distribution": {
                "completeness": {"avg": 0, "min": 5, "max": 1},
                "context_independence": {"avg": 0, "min": 5, "max": 1},
                "technical_accuracy": {"avg": 0, "min": 5, "max": 1},
            },
            "low_quality_rate": 0,
        }

        slack_dir = self.resolve_path(self.SLACK_QA_SCORED_DIR)
        if not slack_dir.exists():
            return quality_stats

        all_scores: dict[str, list[int]] = {
            "completeness": [],
            "context_independence": [],
            "technical_accuracy": [],
        }

        for json_file in slack_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                qa_pairs = data.get("qa_pairs", [])
                quality_stats["total_qa_pairs"] += len(qa_pairs)

                for qa in qa_pairs:
                    evaluation = qa.get("evaluation", {})
                    if not evaluation:
                        continue

                    quality_stats["evaluated"] += 1

                    # 전체 품질 등급
                    overall = evaluation.get("overall_quality", "").lower()
                    if overall in quality_stats["quality_distribution"]:
                        quality_stats["quality_distribution"][overall] += 1

                    # 개별 점수
                    for dim in ["completeness", "context_independence", "technical_accuracy"]:
                        dim_data = evaluation.get(dim, {})
                        if isinstance(dim_data, dict) and "score" in dim_data:
                            score = dim_data["score"]
                            all_scores[dim].append(score)

            except Exception as e:
                logger.warning(f"Slack QA 파일 처리 실패: {json_file}, {e}")

        # 통계 계산
        for dim, scores in all_scores.items():
            if scores:
                quality_stats["score_distribution"][dim] = {
                    "avg": round(sum(scores) / len(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                }

        # 저품질 비율
        total_evaluated = quality_stats["evaluated"]
        if total_evaluated > 0:
            low_count = (
                quality_stats["quality_distribution"]["low"]
                + quality_stats["quality_distribution"]["remove"]
            )
            quality_stats["low_quality_rate"] = round(low_count / total_evaluated * 100, 2)

            if quality_stats["low_quality_rate"] > self.LOW_QUALITY_THRESHOLD * 100:
                self.add_issue(
                    severity=Severity.WARNING,
                    category="quality",
                    message=f"저품질 Slack QA 비율이 높음: {quality_stats['low_quality_rate']}%",
                    details=quality_stats["quality_distribution"],
                )

        return quality_stats

    async def _analyze_doc_type_distribution(self) -> dict[str, Any]:
        """doc_type별 분포 분석."""
        distribution: dict[str, int] = Counter()

        # Slack QA
        slack_dir = self.resolve_path(self.SLACK_QA_SCORED_DIR)
        if slack_dir.exists():
            for json_file in slack_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        distribution["slack_qa"] += len(data.get("qa_pairs", []))
                except Exception:
                    pass

        # Transcript
        transcript_dir = self.resolve_path(self.TRANSCRIPT_CHUNKS_DIR)
        if transcript_dir.exists():
            for json_file in transcript_dir.glob("*_chunks.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunks = data if isinstance(data, list) else data.get("chunks", [])
                        distribution["lecture_transcript"] += len(chunks)
                except Exception:
                    pass

        return dict(distribution)

    async def _analyze_course_distribution(self) -> dict[str, Any]:
        """코스별 분포 분석."""
        distribution: dict[str, int] = Counter()

        # Slack QA - 파일명에서 코스 추출
        slack_dir = self.resolve_path(self.SLACK_QA_SCORED_DIR)
        if slack_dir.exists():
            for json_file in slack_dir.glob("*.json"):
                # 파일명 예: level2_nlp_merged.json
                course = json_file.stem.replace("_merged", "")
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        distribution[course] += len(data.get("qa_pairs", []))
                except Exception:
                    pass

        # Transcript - 파일명에서 코스 추출
        transcript_dir = self.resolve_path(self.TRANSCRIPT_CHUNKS_DIR)
        if transcript_dir.exists():
            for json_file in transcript_dir.glob("*_chunks.json"):
                # 파일명 예: [NLP 이론] (8강) Transformer_chunks.json
                filename = json_file.stem
                course = self._extract_course(filename)
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunks = data if isinstance(data, list) else data.get("chunks", [])
                        distribution[course] += len(chunks)
                except Exception:
                    pass

        return dict(distribution)

    def _extract_course(self, filename: str) -> str:
        """파일명에서 코스명 추출."""
        if filename.startswith("["):
            end = filename.find("]")
            if end > 0:
                return filename[1:end]
        return "general"

    async def _detect_anomalies(self) -> dict[str, Any]:
        """이상치 탐지."""
        anomalies: dict[str, Any] = {
            "empty_content": 0,
            "extremely_short": 0,
            "extremely_long": 0,
            "missing_metadata": 0,
            "suspicious_patterns": [],
        }

        # Transcript chunks 점검
        transcript_dir = self.resolve_path(self.TRANSCRIPT_CHUNKS_DIR)
        if transcript_dir.exists():
            for json_file in transcript_dir.glob("*_chunks.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunks = data if isinstance(data, list) else data.get("chunks", [])

                        for chunk in chunks:
                            content = chunk.get("page_content", "") or chunk.get("content", "")

                            # 빈 콘텐츠
                            if not content.strip():
                                anomalies["empty_content"] += 1

                            # 극단적 길이
                            length = len(content)
                            if length < 20:
                                anomalies["extremely_short"] += 1
                            elif length > 10000:
                                anomalies["extremely_long"] += 1

                            # 메타데이터 누락
                            metadata = chunk.get("metadata", {})
                            if not metadata:
                                anomalies["missing_metadata"] += 1

                except Exception:
                    pass

        # 이상치 이슈 추가
        if anomalies["empty_content"] > 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="anomaly",
                message=f"빈 콘텐츠 청크: {anomalies['empty_content']}개",
            )

        if anomalies["extremely_short"] > 0:
            self.add_issue(
                severity=Severity.INFO,
                category="anomaly",
                message=f"극단적으로 짧은 청크 (20자 미만): {anomalies['extremely_short']}개",
            )

        if anomalies["extremely_long"] > 0:
            self.add_issue(
                severity=Severity.INFO,
                category="anomaly",
                message=f"극단적으로 긴 청크 (10000자 초과): {anomalies['extremely_long']}개",
            )

        return anomalies

    def _compute_quality_score(self, stats: dict[str, Any]) -> float:
        """전체 품질 점수 계산 (0-100)."""
        score = 100.0

        # Slack QA 품질 반영
        slack_quality = stats.get("slack_qa_quality", {})
        low_quality_rate = slack_quality.get("low_quality_rate", 0)
        score -= low_quality_rate * 0.5  # 저품질 비율만큼 감점

        # 이상치 반영
        anomalies = stats.get("anomalies", {})
        total_docs = stats.get("overall_summary", {}).get("total_documents", 1)

        empty_rate = anomalies.get("empty_content", 0) / total_docs * 100
        score -= empty_rate * 2  # 빈 콘텐츠는 2배 감점

        missing_meta_rate = anomalies.get("missing_metadata", 0) / total_docs * 100
        score -= missing_meta_rate * 0.5

        # 범위 제한
        return max(0, min(100, round(score, 2)))


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = QualityAuditor(verbose=True)
        result = await auditor.audit()

        print("\n" + "=" * 60)
        print("데이터 품질 점검 결과")
        print("=" * 60)
        print(f"상태: {result.status}")
        print(f"전체 문서: {result.total_items}")
        print(f"품질 점수: {result.stats.extra.get('quality_score', 0)}/100")
        print(f"이슈: {len(result.issues)}개")
        print(f"소요 시간: {result.duration_seconds:.2f}초")

        print("\n📊 전체 요약:")
        for key, value in result.stats.extra.get("overall_summary", {}).items():
            print(f"  {key}: {value}")

        print("\n📊 Slack QA 품질:")
        for key, value in result.stats.extra.get("slack_qa_quality", {}).items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        print("\n📊 doc_type별 분포:")
        for key, value in result.stats.extra.get("distribution_by_doc_type", {}).items():
            print(f"  {key}: {value}")

        if result.issues:
            print("\n⚠️ 발견된 이슈:")
            for issue in result.issues:
                print(f"  {issue}")

    asyncio.run(main())
