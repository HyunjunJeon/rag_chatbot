"""
Hybrid Search: 벡터 검색(Qdrant) + BM25 검색을 결합한 검색기.

Reciprocal Rank Fusion (RRF)을 사용하여 두 검색 결과를 통합합니다.
"""

from pathlib import Path
from typing import Any

from bm25_indexer import BM25Indexer
from openrouter_embeddings import OpenRouterEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """
    벡터 검색과 BM25 검색을 결합한 Hybrid Search 클래스.

    예시:
        ```python
        retriever = HybridRetriever(
            qdrant_url="http://localhost:6333",
            collection_name="slack_qa",
            bm25_index_path="bm25_index.pkl",
            embedding_model="jhgan/ko-sroberta-multitask"
        )

        results = retriever.search(
            query="GPU 메모리 부족",
            course="level2_cv",
            alpha=0.7,  # 벡터 70%, BM25 30%
            limit=10
        )
        ```
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "slack_qa",
        bm25_index_path: Path | str = "document_chunks/bm25_index.pkl",
        embedding_model: str = "qwen/qwen3-embedding-4b",
    ) -> None:
        """
        초기화.

        매개변수:
            qdrant_url: Qdrant 서버 URL
            collection_name: Qdrant 컬렉션 이름
            bm25_index_path: BM25 인덱스 파일 경로
            embedding_model: 임베딩 모델 이름
        """
        print("🚀 HybridRetriever 초기화 중...")

        # Qdrant 클라이언트
        print(f"   Qdrant 연결: {qdrant_url}")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # 임베딩 모델
        print(f"   임베딩 모델 로드: {embedding_model}")
        self.embedding_model = OpenRouterEmbeddings(model=embedding_model)

        # BM25 인덱서
        print(f"   BM25 인덱스 로드: {bm25_index_path}")
        self.bm25_indexer = BM25Indexer()
        self.bm25_indexer.load_index(bm25_index_path)

        print("   ✅ 초기화 완료\n")

    def _vector_search(
        self,
        query: str,
        limit: int = 50,
        course_filter: str | None = None,
        year_from: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Qdrant 벡터 검색을 수행합니다.

        매개변수:
            query: 검색 쿼리
            limit: 결과 개수
            course_filter: 과정 필터
            year_from: 시작 연도 필터

        반환값:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # 필터 생성
        filters = []
        if course_filter:
            filters.append(
                FieldCondition(key="course", match=MatchValue(value=course_filter))
            )
        if year_from:
            from qdrant_client.models import Range

            filters.append(FieldCondition(key="year", range=Range(gte=year_from)))

        query_filter = Filter(must=filters) if filters else None

        # 검색
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        # 결과 변환
        vector_results = []
        for i, result in enumerate(results, 1):
            vector_results.append(
                {
                    "doc_id": str(result.id),
                    "vector_score": result.score,
                    "vector_rank": i,
                    "payload": result.payload,
                }
            )

        return vector_results

    def _bm25_search(
        self,
        query: str,
        limit: int = 50,
        course_filter: str | None = None,
        question_weight: float = 0.7,
        answer_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        BM25 검색을 수행합니다.

        매개변수:
            query: 검색 쿼리
            limit: 결과 개수
            course_filter: 과정 필터
            question_weight: 질문 가중치
            answer_weight: 답변 가중치

        반환값:
            검색 결과 리스트
        """
        results = self.bm25_indexer.search(
            query=query,
            question_weight=question_weight,
            answer_weight=answer_weight,
            limit=limit,
            course_filter=course_filter,
        )

        # 결과 변환
        bm25_results = []
        for i, result in enumerate(results, 1):
            bm25_results.append(
                {
                    "doc_id": result["doc_id"],
                    "bm25_score": result["bm25_score"],
                    "bm25_rank": i,
                    "document": result,
                }
            )

        return bm25_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF)로 결과를 병합합니다.

        RRF 공식: score(d) = Σ 1 / (k + rank(d))

        매개변수:
            vector_results: 벡터 검색 결과
            bm25_results: BM25 검색 결과
            k: RRF 파라미터 (일반적으로 60)

        반환값:
            (doc_id, rrf_score) 튜플의 리스트
        """
        scores: dict[str, float] = {}

        # 벡터 검색 점수
        for result in vector_results:
            doc_id = result["doc_id"]
            rank = result["vector_rank"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        # BM25 검색 점수
        for result in bm25_results:
            doc_id = result["doc_id"]
            rank = result["bm25_rank"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        # 점수 순 정렬
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_docs

    def _weighted_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        alpha: float = 0.7,
    ) -> list[tuple[str, float]]:
        """
        가중치 기반 점수 융합을 수행합니다.

        매개변수:
            vector_results: 벡터 검색 결과
            bm25_results: BM25 검색 결과
            alpha: 벡터 검색 가중치 (0~1), BM25는 (1-alpha)

        반환값:
            (doc_id, weighted_score) 튜플의 리스트
        """
        scores: dict[str, float] = {}

        # 벡터 검색 정규화 점수
        for result in vector_results:
            doc_id = result["doc_id"]
            # 순위 기반 정규화 (1/rank)
            normalized_score = 1 / result["vector_rank"]
            scores[doc_id] = alpha * normalized_score

        # BM25 검색 정규화 점수
        for result in bm25_results:
            doc_id = result["doc_id"]
            normalized_score = 1 / result["bm25_rank"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * normalized_score

        # 점수 순 정렬
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_docs

    def search(
        self,
        query: str,
        course: str | None = None,
        year_from: int | None = None,
        alpha: float = 0.7,
        fusion_method: str = "rrf",
        limit: int = 10,
        bm25_question_weight: float = 0.7,
        bm25_answer_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Hybrid Search를 수행합니다.

        매개변수:
            query: 검색 쿼리
            course: 과정 필터 (예: "level2_cv")
            year_from: 시작 연도 (예: 2023)
            alpha: 벡터 검색 가중치 (0~1)
            fusion_method: 융합 방법 ("rrf" 또는 "weighted")
            limit: 최종 결과 개수
            bm25_question_weight: BM25 질문 가중치
            bm25_answer_weight: BM25 답변 가중치

        반환값:
            검색 결과 리스트
        """
        # 1. 벡터 검색
        vector_results = self._vector_search(
            query=query, limit=50, course_filter=course, year_from=year_from
        )

        # 2. BM25 검색
        bm25_results = self._bm25_search(
            query=query,
            limit=50,
            course_filter=course,
            question_weight=bm25_question_weight,
            answer_weight=bm25_answer_weight,
        )

        # 3. 결과 융합
        if fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        else:  # weighted
            fused_results = self._weighted_fusion(vector_results, bm25_results, alpha=alpha)

        # 4. 상위 결과 선택 및 정보 결합
        final_results = []

        # 문서 정보 매핑 생성
        vector_map = {r["doc_id"]: r for r in vector_results}
        bm25_map = {r["doc_id"]: r for r in bm25_results}

        for doc_id, fusion_score in fused_results[:limit]:
            # 벡터 결과와 BM25 결과 결합
            vector_info = vector_map.get(doc_id)
            bm25_info = bm25_map.get(doc_id)

            if not (vector_info or bm25_info):
                continue

            # 결과 구성
            result = {
                "doc_id": doc_id,
                "fusion_score": fusion_score,
                "vector_score": vector_info["vector_score"] if vector_info else 0.0,
                "bm25_score": bm25_info["bm25_score"] if bm25_info else 0.0,
                "vector_rank": vector_info["vector_rank"] if vector_info else None,
                "bm25_rank": bm25_info["bm25_rank"] if bm25_info else None,
            }

            # 문서 정보 (둘 중 하나에서 가져오기)
            if vector_info:
                result.update(
                    {
                        "question_text": vector_info["payload"]["question_text"],
                        "answer_text": vector_info["payload"]["answer_text"],
                        "course": vector_info["payload"]["course"],
                        "generation": vector_info["payload"]["generation"],
                        "date": vector_info["payload"]["date"],
                        "metadata": vector_info["payload"],
                    }
                )
            elif bm25_info:
                doc = bm25_info["document"]
                result.update(
                    {
                        "question_text": doc["question_text"],
                        "answer_text": doc["answer_text"],
                        "course": doc["course"],
                        "generation": doc["generation"],
                        "date": doc["date"],
                        "metadata": doc,
                    }
                )

            final_results.append(result)

        return final_results

    def compare_search_methods(
        self, query: str, course: str | None = None, limit: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """
        벡터, BM25, 하이브리드 검색 결과를 비교합니다.

        매개변수:
            query: 검색 쿼리
            course: 과정 필터
            limit: 각 결과 개수

        반환값:
            {"vector": [...], "bm25": [...], "hybrid": [...]}
        """
        # 벡터만
        vector_only = self._vector_search(query=query, limit=limit, course_filter=course)

        # BM25만
        bm25_only = self._bm25_search(query=query, limit=limit, course_filter=course)

        # 하이브리드
        hybrid = self.search(query=query, course=course, limit=limit)

        return {"vector": vector_only, "bm25": bm25_only, "hybrid": hybrid}


def main() -> None:
    """메인 함수."""
    import json
    from datetime import datetime

    print("=" * 80)
    print("🚀 Hybrid Retriever 테스트")
    print("=" * 80)

    # PROJECT_ROOT 기준 상대 경로 사용
    project_root = Path(__file__).parent.parent

    try:
        # Retriever 초기화
        retriever = HybridRetriever(
            qdrant_url="http://localhost:6333",
            collection_name="slack_qa",
            bm25_index_path=str(project_root / "document_chunks" / "bm25_index.pkl"),
        )

        # 테스트 검색
        test_queries = [
            ("GPU 메모리 부족 해결 방법", "level2_cv", 0.7),
            ("데이터 증강 기법", "level2_cv", 0.6),
            ("optimizer 선택", None, 0.5),
        ]

        # 결과 저장용
        all_test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "model": "qwen/qwen3-embedding-4b",
            "total_documents": 4581,
            "queries": []
        }

        for query, course, alpha in test_queries:
            print(f"\n{'=' * 80}")
            print(f"🔍 쿼리: {query}")
            print(f"   과정: {course or '전체'}")
            print(f"   Alpha: {alpha} (벡터 {int(alpha*100)}%, BM25 {int((1-alpha)*100)}%)")
            print("=" * 80)

            results = retriever.search(query=query, course=course, alpha=alpha, limit=5)

            # 쿼리 결과 저장
            query_result = {
                "query": query,
                "course_filter": course,
                "alpha": alpha,
                "results": []
            }

            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Fusion: {result['fusion_score']:.4f}")
                print(
                    f"    Vector: {result['vector_score']:.3f} (rank {result['vector_rank']})"
                )
                print(f"    BM25:   {result['bm25_score']:.3f} (rank {result['bm25_rank']})")
                print(f"    과정: {result['course']} ({result['generation']}기)")
                print(f"    질문: {result['question_text'][:80]}...")
                print(f"    답변: {result['answer_text'][:80]}...")

                # 결과 데이터 저장
                query_result["results"].append({
                    "rank": i,
                    "fusion_score": result['fusion_score'],
                    "vector_score": result['vector_score'],
                    "vector_rank": result['vector_rank'],
                    "bm25_score": result['bm25_score'],
                    "bm25_rank": result['bm25_rank'],
                    "course": result['course'],
                    "generation": result['generation'],
                    "date": result['date'],
                    "question_text": result['question_text'],
                    "answer_text": result['answer_text'],
                })

            all_test_results["queries"].append(query_result)

        # JSON 파일로 저장
        output_path = project_root / "document_chunks" / "hybrid_search_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_test_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("✅ 테스트 완료")
        print("=" * 80)
        print(f"\n💾 결과 저장: {output_path}")
        print(f"   총 {len(test_queries)}개 쿼리, {sum(len(q['results']) for q in all_test_results['queries'])}개 결과")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n💡 확인 사항:")
        print("   1. Qdrant가 실행 중인가요? (docker run -p 6333:6333 qdrant/qdrant)")
        print("   2. slack_qa 컬렉션이 생성되었나요?")
        print("   3. BM25 인덱스가 생성되었나요?")


if __name__ == "__main__":
    main()

