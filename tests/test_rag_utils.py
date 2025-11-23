"""
RAG 유틸리티 함수 테스트 모듈.

이 모듈은 naver_connect_chatbot.rag.utils의 모든 함수를 테스트합니다.
"""

import numpy as np
import pytest
from langchain_core.documents import Document

from naver_connect_chatbot.rag.utils import (
    deduplicate_documents,
    deduplicate_strings,
    merge_document_metadata,
    min_max_normalize,
    softmax_normalize,
    unique_by_key,
    z_score_normalize,
)


# ============================================================================
# 중복 제거 테스트
# ============================================================================


class TestUniqueByKey:
    """unique_by_key 함수 테스트."""

    def test_unique_by_key_with_strings(self):
        """문자열 리스트에서 길이 기준 중복 제거."""
        strings = ["a", "bb", "c", "dd", "e"]
        # 길이 기준으로 중복 제거 (bb와 dd는 길이가 2로 같음)
        result = list(unique_by_key(strings, key=len))
        assert len(result) == 3
        assert result == ["a", "bb", "c"]  # 첫 번째 발생만 유지

    def test_unique_by_key_with_documents(self):
        """Document 리스트에서 page_content 기준 중복 제거."""
        docs = [
            Document(page_content="Hello", metadata={"id": "1"}),
            Document(page_content="World", metadata={"id": "2"}),
            Document(page_content="Hello", metadata={"id": "3"}),  # 중복
        ]
        result = list(unique_by_key(docs, key=lambda d: d.page_content))
        assert len(result) == 2
        assert result[0].page_content == "Hello"
        assert result[0].metadata["id"] == "1"  # 첫 번째 발생 유지

    def test_unique_by_key_preserves_order(self):
        """첫 번째 발생 순서를 유지하는지 확인."""
        items = [("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5)]
        result = list(unique_by_key(items, key=lambda x: x[0]))
        assert result == [("a", 1), ("b", 2), ("c", 3)]

    def test_unique_by_key_empty_iterable(self):
        """빈 iterable 처리."""
        result = list(unique_by_key([], key=lambda x: x))
        assert result == []


class TestDeduplicateStrings:
    """deduplicate_strings 함수 테스트."""

    def test_case_sensitive_deduplication(self):
        """대소문자 구분 중복 제거."""
        strings = ["Hello", "HELLO", "World", "hello"]
        result = deduplicate_strings(strings, case_sensitive=True)
        assert len(result) == 4
        assert result == ["Hello", "HELLO", "World", "hello"]

    def test_case_insensitive_deduplication(self):
        """대소문자 무시 중복 제거."""
        strings = ["Hello", "HELLO", "World", "hello"]
        result = deduplicate_strings(strings, case_sensitive=False)
        assert len(result) == 2
        assert result == ["Hello", "World"]  # 첫 번째 발생 유지

    def test_preserves_first_occurrence(self):
        """첫 번째 발생을 유지하는지 확인."""
        strings = ["apple", "APPLE", "Apple"]
        result = deduplicate_strings(strings, case_sensitive=False)
        assert result == ["apple"]

    def test_empty_list(self):
        """빈 리스트 처리."""
        result = deduplicate_strings([])
        assert result == []

    def test_single_item(self):
        """단일 항목 처리."""
        result = deduplicate_strings(["Hello"])
        assert result == ["Hello"]


class TestDeduplicateDocuments:
    """deduplicate_documents 함수 테스트."""

    def test_dedup_by_page_content(self):
        """page_content 기준 중복 제거."""
        docs = [
            Document(page_content="Hello", metadata={"id": "1"}),
            Document(page_content="World", metadata={"id": "2"}),
            Document(page_content="Hello", metadata={"id": "3"}),  # 중복
        ]
        result = deduplicate_documents(docs)
        assert len(result) == 2
        assert result[0].metadata["id"] == "1"  # 첫 번째 발생 유지

    def test_dedup_by_metadata_key(self):
        """metadata 키 기준 중복 제거."""
        docs = [
            Document(page_content="Different text 1", metadata={"doc_id": "A"}),
            Document(page_content="Different text 2", metadata={"doc_id": "B"}),
            Document(page_content="Different text 3", metadata={"doc_id": "A"}),  # 중복
        ]
        result = deduplicate_documents(docs, id_key="doc_id")
        assert len(result) == 2
        assert result[0].page_content == "Different text 1"

    def test_dedup_missing_id_key_fallback(self):
        """id_key가 없는 문서는 page_content로 fallback."""
        docs = [
            Document(page_content="Same content", metadata={"doc_id": "A"}),
            Document(page_content="Same content", metadata={}),  # id_key 없음
        ]
        result = deduplicate_documents(docs, id_key="doc_id")
        assert len(result) == 1  # page_content가 같으므로 중복 제거

    def test_empty_list(self):
        """빈 리스트 처리."""
        result = deduplicate_documents([])
        assert result == []


# ============================================================================
# 점수 정규화 테스트
# ============================================================================


class TestSoftmaxNormalize:
    """softmax_normalize 함수 테스트."""

    def test_softmax_with_list(self):
        """리스트 입력으로 softmax 계산."""
        scores = [2.0, 1.0, 0.1]
        result = softmax_normalize(scores)

        # 결과는 numpy 배열이어야 함
        assert isinstance(result, np.ndarray)

        # 합이 1이어야 함
        assert np.isclose(result.sum(), 1.0)

        # 가장 높은 점수가 가장 높은 확률
        assert result[0] > result[1] > result[2]

    def test_softmax_with_ndarray(self):
        """numpy 배열 입력으로 softmax 계산."""
        scores = np.array([3.0, 1.0, 0.2])
        result = softmax_normalize(scores)

        assert np.isclose(result.sum(), 1.0)
        assert result[0] > result[1] > result[2]

    def test_softmax_numerical_stability(self):
        """큰 값에서도 수치적 안정성 확인."""
        scores = [1000.0, 999.0, 998.0]
        result = softmax_normalize(scores)

        # Overflow 없이 계산되어야 함
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.isclose(result.sum(), 1.0)

    def test_softmax_equal_scores(self):
        """모든 점수가 같을 때 균등 분포."""
        scores = [5.0, 5.0, 5.0]
        result = softmax_normalize(scores)

        # 모든 확률이 같아야 함
        assert np.allclose(result, [1/3, 1/3, 1/3])


class TestMinMaxNormalize:
    """min_max_normalize 함수 테스트."""

    def test_basic_normalization(self):
        """기본 min-max 정규화."""
        scores = [10, 20, 30]
        result = min_max_normalize(scores)

        assert result == [0.0, 0.5, 1.0]
        assert min(result) == 0.0
        assert max(result) == 1.0

    def test_negative_scores(self):
        """음수 점수 처리."""
        scores = [-10, 0, 10]
        result = min_max_normalize(scores)

        assert result == [0.0, 0.5, 1.0]

    def test_equal_scores(self):
        """모든 점수가 같을 때 1.0 반환."""
        scores = [42, 42, 42]
        result = min_max_normalize(scores)

        assert result == [1.0, 1.0, 1.0]

    def test_empty_list(self):
        """빈 리스트 처리."""
        result = min_max_normalize([])
        assert result == []

    def test_single_score(self):
        """단일 점수 처리."""
        result = min_max_normalize([5.0])
        assert result == [1.0]


class TestZScoreNormalize:
    """z_score_normalize 함수 테스트."""

    def test_basic_zscore(self):
        """기본 z-score 정규화."""
        scores = [10, 20, 30]
        result = z_score_normalize(scores)

        # 평균이 0에 가까워야 함
        mean = sum(result) / len(result)
        assert abs(mean) < 1e-10

    def test_standard_deviation(self):
        """표준편차가 1에 가까운지 확인."""
        scores = [1, 2, 3, 4, 5]
        result = z_score_normalize(scores)

        mean = sum(result) / len(result)
        variance = sum((x - mean) ** 2 for x in result) / len(result)
        std = variance ** 0.5

        assert abs(std - 1.0) < 1e-10

    def test_equal_scores(self):
        """모든 점수가 같을 때 0 반환."""
        scores = [100, 100, 100]
        result = z_score_normalize(scores)

        assert result == [0.0, 0.0, 0.0]

    def test_empty_list(self):
        """빈 리스트 처리."""
        result = z_score_normalize([])
        assert result == []

    def test_single_score(self):
        """단일 점수 처리."""
        result = z_score_normalize([42])
        assert result == [0.0]


# ============================================================================
# 문서 병합 테스트
# ============================================================================


class TestMergeDocumentMetadata:
    """merge_document_metadata 함수 테스트."""

    def test_overwrite_mode(self):
        """덮어쓰기 모드에서 새 메타데이터가 기존 값을 덮어씀."""
        doc = Document(
            page_content="Hello",
            metadata={"source": "file1.txt", "score": 0.5}
        )
        new_metadata = {"score": 0.9, "rank": 1}

        result = merge_document_metadata(doc, new_metadata, overwrite=True)

        assert result.metadata["score"] == 0.9  # 덮어쓰기됨
        assert result.metadata["rank"] == 1  # 새 키 추가됨
        assert result.metadata["source"] == "file1.txt"  # 기존 키 유지

    def test_preserve_mode(self):
        """보존 모드에서 기존 메타데이터 유지."""
        doc = Document(
            page_content="Hello",
            metadata={"source": "file1.txt", "score": 0.5}
        )
        new_metadata = {"score": 0.9, "rank": 1}

        result = merge_document_metadata(doc, new_metadata, overwrite=False)

        assert result.metadata["score"] == 0.5  # 기존 값 유지
        assert result.metadata["rank"] == 1  # 새 키 추가됨

    def test_page_content_unchanged(self):
        """page_content는 변경되지 않아야 함."""
        doc = Document(page_content="Original content", metadata={})
        result = merge_document_metadata(doc, {"new_key": "value"})

        assert result.page_content == "Original content"

    def test_empty_new_metadata(self):
        """새 메타데이터가 비어있을 때."""
        doc = Document(
            page_content="Hello",
            metadata={"existing": "value"}
        )
        result = merge_document_metadata(doc, {})

        assert result.metadata == {"existing": "value"}

    def test_empty_existing_metadata(self):
        """기존 메타데이터가 비어있을 때."""
        doc = Document(page_content="Hello", metadata={})
        result = merge_document_metadata(doc, {"new_key": "value"})

        assert result.metadata == {"new_key": "value"}


# ============================================================================
# 통합 테스트
# ============================================================================


class TestIntegration:
    """여러 유틸리티 함수를 조합한 통합 테스트."""

    def test_dedup_and_normalize_workflow(self):
        """중복 제거 후 점수 정규화 워크플로우."""
        # 1. 중복된 문서 생성
        docs = [
            Document(page_content="Doc A", metadata={"score": 10}),
            Document(page_content="Doc B", metadata={"score": 20}),
            Document(page_content="Doc A", metadata={"score": 15}),  # 중복
            Document(page_content="Doc C", metadata={"score": 30}),
        ]

        # 2. 중복 제거
        unique_docs = deduplicate_documents(docs)
        assert len(unique_docs) == 3

        # 3. 점수 추출 및 정규화
        scores = [doc.metadata["score"] for doc in unique_docs]
        normalized_scores = min_max_normalize(scores)

        # 4. 정규화된 점수를 메타데이터에 병합
        final_docs = [
            merge_document_metadata(doc, {"normalized_score": score})
            for doc, score in zip(unique_docs, normalized_scores)
        ]

        # 검증
        assert final_docs[0].metadata["normalized_score"] == 0.0  # 10 -> 0.0
        assert final_docs[1].metadata["normalized_score"] == 0.5  # 20 -> 0.5
        assert final_docs[2].metadata["normalized_score"] == 1.0  # 30 -> 1.0

    def test_string_dedup_with_document_creation(self):
        """문자열 중복 제거 후 문서 생성."""
        queries = ["What is AI?", "WHAT IS AI?", "What is ML?"]

        # 대소문자 무시 중복 제거
        unique_queries = deduplicate_strings(queries, case_sensitive=False)
        assert len(unique_queries) == 2

        # 문서로 변환
        docs = [
            Document(page_content=q, metadata={"query_type": "original"})
            for q in unique_queries
        ]
        assert len(docs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
