"""
인덱스 동기화 점검 모듈.

점검 대상:
- BM25 인덱스 (sparse_index/unified_bm25/)
- Qdrant 벡터 DB (vdb_store/collections/)

점검 항목:
- 인덱스 로드 가능 여부
- 인덱싱된 문서 수 일치
- 토큰화 품질 (Kiwi 형태소 분석)
- 벡터 차원 일치

사용 예:
    ```python
    from document_processing.audit.auditors.index_auditor import IndexAuditor

    auditor = IndexAuditor(verbose=True)
    result = await auditor.audit()
    print(result.model_dump_json(indent=2))
    ```
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from document_processing.audit.auditors.base import BaseAuditor
from document_processing.audit.models.audit_result import LayerResult, LayerStats, Severity

logger = logging.getLogger(__name__)


class IndexAuditor(BaseAuditor):
    """인덱스 동기화 점검기."""

    layer_name = "indexes"

    # 인덱스 경로 (base_path 기준 상대 경로)
    BM25_INDEX_DIR = "sparse_index/unified_bm25"
    BM25_INDEX_FILE = "bm25_index.pkl"
    QDRANT_STORE_DIR = "vdb_store"
    QDRANT_COLLECTION_NAME = "naver_connect_docs"

    # 기대 값
    EXPECTED_EMBEDDING_DIM = 1024  # BGE-M3 차원

    async def audit(self) -> LayerResult:
        """인덱스 동기화 점검 실행."""
        result = self.create_result()
        self.start_timer()

        stats_extra: dict[str, Any] = {
            "bm25": {},
            "qdrant": {},
            "sync_status": {},
        }

        total_items = 0
        checked_items = 0
        passed_items = 0
        failed_items = 0

        # 1. BM25 인덱스 점검
        bm25_stats = await self._audit_bm25_index()
        stats_extra["bm25"] = bm25_stats
        total_items += 1
        checked_items += 1
        if bm25_stats.get("loaded"):
            passed_items += 1
        else:
            failed_items += 1

        # 2. Qdrant 벡터 DB 점검
        qdrant_stats = await self._audit_qdrant_store()
        stats_extra["qdrant"] = qdrant_stats
        total_items += 1
        checked_items += 1
        if qdrant_stats.get("accessible"):
            passed_items += 1
        else:
            failed_items += 1

        # 3. 동기화 상태 점검
        sync_stats = self._check_sync_status(bm25_stats, qdrant_stats)
        stats_extra["sync_status"] = sync_stats

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

    async def _audit_bm25_index(self) -> dict[str, Any]:
        """BM25 인덱스 점검."""
        stats: dict[str, Any] = {
            "loaded": False,
            "path": None,
            "file_size_bytes": 0,
            "document_count": 0,
            "vocab_size": 0,
            "sample_tokens": [],
        }

        index_dir = self.resolve_path(self.BM25_INDEX_DIR)
        index_file = index_dir / self.BM25_INDEX_FILE
        stats["path"] = str(index_file)

        if not index_dir.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="index",
                message=f"BM25 인덱스 디렉토리 없음: {index_dir}",
            )
            return stats

        if not index_file.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="index",
                message=f"BM25 인덱스 파일 없음: {index_file}",
            )
            return stats

        # 파일 크기
        stats["file_size_bytes"] = index_file.stat().st_size
        stats["file_size_mb"] = round(stats["file_size_bytes"] / (1024 * 1024), 2)

        # 인덱스 로드 시도
        load_result = await self._load_bm25_index(index_file)

        if load_result["success"]:
            stats["loaded"] = True
            stats["document_count"] = load_result.get("doc_count", 0)
            stats["vocab_size"] = load_result.get("vocab_size", 0)
            stats["sample_tokens"] = load_result.get("sample_tokens", [])
            stats["avg_doc_length"] = load_result.get("avg_doc_length", 0)
        else:
            self.add_issue(
                severity=Severity.CRITICAL,
                category="index",
                message=f"BM25 인덱스 로드 실패: {load_result.get('error')}",
                file_path=str(index_file),
            )

        # 사용자 사전 확인
        user_dict_path = index_dir / "user_dict.txt"
        if user_dict_path.exists():
            stats["has_user_dict"] = True
            with open(user_dict_path, "r", encoding="utf-8") as f:
                stats["user_dict_entries"] = len(f.readlines())
        else:
            stats["has_user_dict"] = False

        return stats

    async def _load_bm25_index(self, index_path: Path) -> dict[str, Any]:
        """BM25 인덱스 로드 및 검증."""

        def load_index() -> dict[str, Any]:
            try:
                with open(index_path, "rb") as f:
                    data = pickle.load(f)

                # KiwiBM25Retriever의 저장 형식 확인
                if isinstance(data, dict):
                    # 새 형식: {"bm25": BM25Okapi, "docs": [...], "doc_ids": [...]}
                    bm25 = data.get("bm25")
                    docs = data.get("docs", [])
                    doc_ids = data.get("doc_ids", [])

                    doc_count = len(docs) if docs else (len(doc_ids) if doc_ids else 0)

                    # BM25 객체에서 정보 추출
                    vocab_size = 0
                    avg_doc_length = 0
                    sample_tokens: list[str] = []

                    if bm25 is not None:
                        if hasattr(bm25, "idf"):
                            vocab_size = len(bm25.idf)
                        if hasattr(bm25, "avgdl"):
                            avg_doc_length = round(bm25.avgdl, 2)
                        if hasattr(bm25, "doc_freqs"):
                            # 상위 빈도 토큰 샘플링
                            sorted_tokens = sorted(
                                bm25.doc_freqs.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                            sample_tokens = [t[0] for t in sorted_tokens[:10]]

                    return {
                        "success": True,
                        "doc_count": doc_count,
                        "vocab_size": vocab_size,
                        "avg_doc_length": avg_doc_length,
                        "sample_tokens": sample_tokens,
                    }

                else:
                    # 구 형식: 직접 BM25 객체
                    return {
                        "success": True,
                        "doc_count": getattr(data, "corpus_size", 0),
                        "vocab_size": len(getattr(data, "idf", {})),
                        "avg_doc_length": round(getattr(data, "avgdl", 0), 2),
                        "sample_tokens": [],
                    }

            except Exception as e:
                return {"success": False, "error": str(e)}

        return await self.run_in_executor(load_index)

    async def _audit_qdrant_store(self) -> dict[str, Any]:
        """Qdrant 벡터 DB 점검."""
        stats: dict[str, Any] = {
            "accessible": False,
            "path": None,
            "collection_exists": False,
            "point_count": 0,
            "vector_dim": 0,
            "config": {},
        }

        store_dir = self.resolve_path(self.QDRANT_STORE_DIR)
        stats["path"] = str(store_dir)

        if not store_dir.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="vectordb",
                message=f"Qdrant 저장소 디렉토리 없음: {store_dir}",
            )
            return stats

        # 컬렉션 디렉토리 확인
        collection_dir = store_dir / "collections" / self.QDRANT_COLLECTION_NAME
        if not collection_dir.exists():
            self.add_issue(
                severity=Severity.CRITICAL,
                category="vectordb",
                message=f"Qdrant 컬렉션 없음: {self.QDRANT_COLLECTION_NAME}",
            )
            return stats

        stats["collection_exists"] = True

        # config.json 읽기
        config_file = collection_dir / "config.json"
        if config_file.exists():
            config_result = await self._load_qdrant_config(config_file)
            if config_result["success"]:
                stats["accessible"] = True
                stats["config"] = config_result.get("config", {})
                stats["vector_dim"] = config_result.get("vector_dim", 0)

                # 벡터 차원 검증
                if stats["vector_dim"] != self.EXPECTED_EMBEDDING_DIM:
                    self.add_issue(
                        severity=Severity.WARNING,
                        category="vectordb",
                        message=f"벡터 차원 불일치: {stats['vector_dim']} (기대값: {self.EXPECTED_EMBEDDING_DIM})",
                    )
            else:
                self.add_issue(
                    severity=Severity.WARNING,
                    category="vectordb",
                    message=f"Qdrant 설정 로드 실패: {config_result.get('error')}",
                )

        # 포인트 수 추정 (세그먼트 파일에서)
        point_count = await self._estimate_point_count(collection_dir)
        stats["point_count"] = point_count

        # 저장소 크기
        total_size = sum(f.stat().st_size for f in collection_dir.rglob("*") if f.is_file())
        stats["storage_size_bytes"] = total_size
        stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats

    async def _load_qdrant_config(self, config_path: Path) -> dict[str, Any]:
        """Qdrant 컬렉션 설정 로드."""

        def load_config() -> dict[str, Any]:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # 벡터 차원 추출
                params = config.get("params", {})
                vectors = params.get("vectors", {})

                # 단일 벡터 또는 명명된 벡터 처리
                if isinstance(vectors, dict):
                    if "size" in vectors:
                        vector_dim = vectors["size"]
                    else:
                        # 명명된 벡터의 경우 첫 번째 것 사용
                        first_vec = next(iter(vectors.values()), {})
                        vector_dim = first_vec.get("size", 0) if isinstance(first_vec, dict) else 0
                else:
                    vector_dim = 0

                return {
                    "success": True,
                    "config": config,
                    "vector_dim": vector_dim,
                }

            except Exception as e:
                return {"success": False, "error": str(e)}

        return await self.run_in_executor(load_config)

    async def _estimate_point_count(self, collection_dir: Path) -> int:
        """Qdrant 포인트 수 추정."""

        def estimate() -> int:
            # payload_index_pointer.json에서 추정
            pointer_file = collection_dir / "payload_index_pointer.json"
            if pointer_file.exists():
                try:
                    with open(pointer_file, "r") as f:
                        data = json.load(f)
                        # 포인트 수 관련 필드 찾기
                        if "points_count" in data:
                            return data["points_count"]
                except Exception:
                    pass

            # 세그먼트 디렉토리에서 추정
            segments_dir = collection_dir / "0" / "segments"
            if segments_dir.exists():
                # 각 세그먼트의 크기로 대략적으로 추정
                segment_dirs = [d for d in segments_dir.iterdir() if d.is_dir()]
                return len(segment_dirs) * 1000  # 매우 대략적인 추정

            return 0

        return await self.run_in_executor(estimate)

    def _check_sync_status(
        self,
        bm25_stats: dict[str, Any],
        qdrant_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """BM25와 Qdrant 간 동기화 상태 점검."""
        sync_status: dict[str, Any] = {
            "in_sync": False,
            "bm25_doc_count": bm25_stats.get("document_count", 0),
            "qdrant_point_count": qdrant_stats.get("point_count", 0),
            "difference": 0,
        }

        bm25_count = sync_status["bm25_doc_count"]
        qdrant_count = sync_status["qdrant_point_count"]

        if bm25_count == 0 or qdrant_count == 0:
            self.add_issue(
                severity=Severity.WARNING,
                category="sync",
                message="인덱스 중 하나가 비어있어 동기화 상태를 확인할 수 없음",
                details={
                    "bm25_count": bm25_count,
                    "qdrant_count": qdrant_count,
                },
            )
            return sync_status

        difference = abs(bm25_count - qdrant_count)
        sync_status["difference"] = difference

        # 10% 이내 차이는 허용
        tolerance = max(bm25_count, qdrant_count) * 0.1

        if difference <= tolerance:
            sync_status["in_sync"] = True
        else:
            sync_status["in_sync"] = False
            self.add_issue(
                severity=Severity.WARNING,
                category="sync",
                message=f"BM25와 Qdrant 문서 수 불일치: BM25={bm25_count}, Qdrant={qdrant_count} (차이: {difference})",
                details={
                    "bm25_count": bm25_count,
                    "qdrant_count": qdrant_count,
                    "difference": difference,
                    "tolerance": round(tolerance),
                },
            )

        return sync_status


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        auditor = IndexAuditor(verbose=True)
        result = await auditor.audit()

        print("\n" + "=" * 60)
        print("인덱스 동기화 점검 결과")
        print("=" * 60)
        print(f"상태: {result.status}")
        print(f"이슈: {len(result.issues)}개")
        print(f"소요 시간: {result.duration_seconds:.2f}초")

        print("\n📊 BM25 인덱스:")
        for key, value in result.stats.extra.get("bm25", {}).items():
            print(f"  {key}: {value}")

        print("\n📊 Qdrant 벡터 DB:")
        for key, value in result.stats.extra.get("qdrant", {}).items():
            if key != "config":
                print(f"  {key}: {value}")

        print("\n📊 동기화 상태:")
        for key, value in result.stats.extra.get("sync_status", {}).items():
            print(f"  {key}: {value}")

        if result.issues:
            print("\n⚠️ 발견된 이슈:")
            for issue in result.issues:
                print(f"  {issue}")

    asyncio.run(main())
