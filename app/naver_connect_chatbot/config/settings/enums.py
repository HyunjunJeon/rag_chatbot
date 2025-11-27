"""
설정에 사용되는 Enum 타입 정의 모듈

이 모듈은 애플리케이션 설정에서 사용되는 열거형(Enum) 타입들을 정의합니다.
타입 안정성과 가독성을 높이기 위해 문자열 기반 Enum을 사용합니다.
"""

from enum import Enum


class RetrieverStrategy(str, Enum):
    """
    검색 전략 타입

    RAG 시스템에서 사용할 검색 전략을 정의합니다.
    """

    SPARSE_ONLY = "sparse_only"  # BM25만 사용
    DENSE_ONLY = "dense_only"  # 벡터 검색만 사용
    HYBRID = "hybrid"  # Dense + Sparse
    MULTI_QUERY = "multi_query"  # MultiQuery (Hybrid 기반)
    ADVANCED = "advanced"  # MultiQuery + Final Hybrid (모든 전략 결합)


class HybridMethodType(str, Enum):
    """
    하이브리드 병합 방식

    Sparse와 Dense 검색 결과를 병합하는 방법을 정의합니다.
    """

    RRF = "rrf"  # Reciprocal Rank Fusion
    CC = "cc"  # Convex Combination


__all__ = [
    "RetrieverStrategy",
    "HybridMethodType",
]
