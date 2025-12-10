"""
검색 성능 점검 모듈.

점검 대상:
- BM25 검색 성능
- 하이브리드 검색 성능 (선택적)

점검 항목:
- 샘플 쿼리 응답 시간
- 검색 결과 존재 여부
- 결과 개수 적절성

주의:
- 이 모듈은 실제 검색을 수행하므로 BM25 인덱스가 필요합니다.
- 전체 하이브리드 검색은 Qdrant 서버와 LLM API가 필요할 수 있습니다.

사용 예:
    ```python
    from document_processing.audit.auditors.search_auditor import SearchAuditor

    auditor = SearchAuditor(verbose=True)
    result = await auditor.audit()
    print(result.model_dump_json(indent=2))
    ```
"""

import logging
import time
from pathlib import Path
from typing import Any

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.models.audit_result import LayerResult, LayerStats, Severity

logger = logging.getLogger(__name__)


class SearchAuditor(BaseAuditor):
    """검색 성능 점검기."""

    layer_name = "search"

    # BM25 인덱스 경로
    BM25_INDEX_DIR = "sparse_index/unified_bm25"

    # 테스트 쿼리 (다양한 유형)
    TEST_QUERIES = [
        # 일반 질문
        {"query": "PyTorch 텐서 연산", "expected_doc_type": "lecture_transcript"},
        {"query": "Transformer 모델 구조", "expected_doc_type": "lecture_transcript"},
        {"query": "학습률 스케줄러", "expected_doc_type": "lecture_transcript"},
        # 코드 관련
        {"query": "loss function 구현", "expected_doc_type": "lecture_transcript"},
        {"query": "DataLoader 사용법", "expected_doc_type": "lecture_transcript"},
        # Slack Q&A 스타일
        {"query": "GPU 메모리 부족 해결", "expected_doc_type": "slack_qa"},
        {"query": "모델 저장 방법", "expected_doc_type": "slack_qa"},
    ]

    # 성능 임계값
    MAX_SEARCH_TIME_MS = 1000  # 1초 이내
    MIN_RESULTS = 1  # 최소 1개 결과

    async def audit(self) -> LayerResult:
        """검색 성능 점검 실행."""
        result = self.create_result()
        self.start_timer()

        stats_extra: dict[str, Any] = {
            "bm25_available": False,
            "queries_tested": 0,
            "queries_passed": 0,
            "avg_search_time_ms": 0,
            "query_results": [],
        }

        # BM25 검색 테스트
        bm25_retriever = await self._load_bm25_retriever()

        if bm25_retriever is None:
            self.add_issue(
                severity=Severity.WARNING,
                category="search",
                message="BM25 인덱스를 로드할 수 없어 검색 테스트를 건너뜁니다",
            )
            result.total_items = 0
            result.stats = LayerStats(extra=stats_extra)
            return self.finalize_result()

        stats_extra["bm25_available"] = True

        # 테스트 쿼리 실행
        total_time_ms = 0
        passed = 0

        for test in self.TEST_QUERIES:
            query = test["query"]
            self.log_progress(
                stats_extra["queries_tested"] + 1,
                len(self.TEST_QUERIES),
                query[:30],
            )

            query_result = await self._test_search(bm25_retriever, query)
            stats_extra["query_results"].append(query_result)
            stats_extra["queries_tested"] += 1

            total_time_ms += query_result.get("time_ms", 0)

            if query_result.get("passed"):
                passed += 1
            else:
                self.add_issue(
                    severity=Severity.INFO,
                    category="search",
                    message=f"쿼리 테스트 실패: '{query[:30]}...' - {query_result.get('reason')}",
                    details=query_result,
                )

        stats_extra["queries_passed"] = passed

        if stats_extra["queries_tested"] > 0:
            stats_extra["avg_search_time_ms"] = round(
                total_time_ms / stats_extra["queries_tested"], 2
            )

        # 전체 성능 평가
        if passed < len(self.TEST_QUERIES) * 0.7:  # 70% 미만 통과 시 경고
            self.add_issue(
                severity=Severity.WARNING,
                category="search",
                message=f"검색 테스트 통과율 낮음: {passed}/{len(self.TEST_QUERIES)} ({passed / len(self.TEST_QUERIES) * 100:.1f}%)",
            )

        if stats_extra["avg_search_time_ms"] > self.MAX_SEARCH_TIME_MS:
            self.add_issue(
                severity=Severity.WARNING,
                category="performance",
                message=f"평균 검색 시간이 느림: {stats_extra['avg_search_time_ms']}ms (임계값: {self.MAX_SEARCH_TIME_MS}ms)",
            )

        # 통계 업데이트
        result.total_items = len(self.TEST_QUERIES)
        result.stats = LayerStats(
            total_items=len(self.TEST_QUERIES),
            checked_items=stats_extra["queries_tested"],
            passed_items=passed,
            failed_items=stats_extra["queries_tested"] - passed,
            extra=stats_extra,
        )

        return self.finalize_result()

    async def _load_bm25_retriever(self) -> Any:
        """BM25 검색기 로드."""
        import pickle

        index_dir = self.resolve_path(self.BM25_INDEX_DIR)
        index_file = index_dir / "bm25_index.pkl"

        if not index_file.exists():
            return None

        def load() -> Any:
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)

                # KiwiBM25Retriever 형식 확인
                if isinstance(data, dict):
                    return data
                return {"bm25": data, "docs": [], "doc_ids": []}

            except Exception as e:
                logger.warning(f"BM25 인덱스 로드 실패: {e}")
                return None

        return await self.run_in_executor(load)

    async def _test_search(
        self,
        retriever_data: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        """개별 검색 테스트 실행."""

        def search() -> dict[str, Any]:
            result: dict[str, Any] = {
                "query": query,
                "passed": False,
                "time_ms": 0,
                "num_results": 0,
                "reason": "",
            }

            try:
                bm25 = retriever_data.get("bm25")
                docs = retriever_data.get("docs", [])

                if bm25 is None:
                    result["reason"] = "BM25 객체 없음"
                    return result

                # Kiwi 토크나이저로 쿼리 토큰화
                try:
                    from kiwipiepy import Kiwi

                    kiwi = Kiwi()
                    tokens = [
                        token.form
                        for token in kiwi.tokenize(query)
                        if token.tag.startswith(("N", "V", "MA"))
                    ]
                except ImportError:
                    # Kiwi 없으면 단순 분리
                    tokens = query.split()

                if not tokens:
                    result["reason"] = "토큰화 결과 없음"
                    return result

                # 검색 실행
                start = time.perf_counter()
                scores = bm25.get_scores(tokens)
                elapsed_ms = (time.perf_counter() - start) * 1000

                result["time_ms"] = round(elapsed_ms, 2)

                # 상위 결과 카운트 (점수 > 0)
                num_results = sum(1 for s in scores if s > 0)
                result["num_results"] = min(num_results, 10)  # 상위 10개까지만 표시

                # 평가
                if num_results >= self.MIN_RESULTS:
                    result["passed"] = True
                else:
                    result["reason"] = f"결과 부족 ({num_results}개)"

                if elapsed_ms > self.MAX_SEARCH_TIME_MS:
                    result["passed"] = False
                    result["reason"] = f"검색 시간 초과 ({elapsed_ms:.0f}ms)"

            except Exception as e:
                result["reason"] = f"검색 오류: {str(e)}"

            return result

        return await self.run_in_executor(search)


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = SearchAuditor(verbose=True)
        result = await auditor.audit()

        print("\n" + "=" * 60)
        print("검색 성능 점검 결과")
        print("=" * 60)
        print(f"상태: {result.status}")
        print(f"테스트 쿼리: {result.total_items}개")
        print(f"통과: {result.stats.passed_items}개")
        print(f"평균 검색 시간: {result.stats.extra.get('avg_search_time_ms', 0)}ms")
        print(f"소요 시간: {result.duration_seconds:.2f}초")

        print("\n📊 쿼리별 결과:")
        for qr in result.stats.extra.get("query_results", []):
            status = "✓" if qr.get("passed") else "✗"
            print(
                f"  {status} '{qr['query'][:25]}...' "
                f"- {qr['num_results']}개 결과, {qr['time_ms']}ms"
            )

        if result.issues:
            print("\n⚠️ 발견된 이슈:")
            for issue in result.issues:
                print(f"  {issue}")

    asyncio.run(main())
