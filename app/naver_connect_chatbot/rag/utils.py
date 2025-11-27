"""
RAG 컴포넌트를 위한 공유 유틸리티 모듈.

이 모듈은 여러 retriever와 reranker에서 공통으로 사용되는
유틸리티 함수들을 제공합니다:

- 문서/문자열 중복 제거
- 점수 정규화 (softmax, min-max, z-score)
- 문서 병합 전략
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import TypeVar

import numpy as np
from langchain_core.documents import Document

__all__ = [
    "unique_by_key",
    "deduplicate_strings",
    "deduplicate_documents",
    "softmax_normalize",
    "min_max_normalize",
    "z_score_normalize",
    "merge_document_metadata",
]

# Type variables
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


# ============================================================================
# 중복 제거 유틸리티
# ============================================================================


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """
    키 함수 기준으로 중복을 제거하며 요소를 순차적으로 반환합니다.

    첫 번째 발생 순서를 유지하며, 이미 본 키는 건너뜁니다.

    매개변수:
        iterable: 중복 제거를 적용할 반복 가능 객체
        key: 각 요소에서 해시 가능한 키를 추출하는 함수

    Yields:
        key 함수 기준으로 고유한 요소

    시간 복잡도:
        O(n), n은 요소 개수

    예시:
        >>> docs = [
        ...     Document(page_content="Hello"),
        ...     Document(page_content="World"),
        ...     Document(page_content="Hello"),  # 중복
        ... ]
        >>> unique = list(unique_by_key(docs, lambda d: d.page_content))
        >>> len(unique)
        2
    """
    seen: set[H] = set()
    for element in iterable:
        k = key(element)
        if k not in seen:
            seen.add(k)
            yield element


def deduplicate_strings(
    strings: Iterable[str],
    *,
    case_sensitive: bool = False,
) -> list[str]:
    """
    문자열 리스트에서 중복을 제거합니다.

    첫 번째 발생 순서를 유지하며, 선택적으로 대소문자를 무시할 수 있습니다.

    매개변수:
        strings: 중복 제거할 문자열 시퀀스
        case_sensitive: True면 대소문자 구분, False면 무시

    반환값:
        중복이 제거된 문자열 리스트

    예시:
        >>> deduplicate_strings(["Hello", "HELLO", "World"], case_sensitive=False)
        ['Hello', 'World']
        >>> deduplicate_strings(["Hello", "HELLO", "World"], case_sensitive=True)
        ['Hello', 'HELLO', 'World']
    """
    seen: set[str] = set()
    result: list[str] = []

    for s in strings:
        # 대소문자 무시 시 casefold() 사용 (Python 3.13에서 권장)
        key = s if case_sensitive else s.casefold()

        if key not in seen:
            seen.add(key)
            result.append(s)

    return result


def deduplicate_documents(
    documents: Iterable[Document],
    *,
    id_key: str | None = None,
) -> list[Document]:
    """
    LangChain Document 리스트에서 중복을 제거합니다.

    id_key가 지정되면 metadata[id_key]를 기준으로 중복을 판단하고,
    없으면 page_content를 기준으로 판단합니다.

    매개변수:
        documents: 중복 제거할 Document 시퀀스
        id_key: 중복 판단에 사용할 metadata 키 (None이면 page_content 사용)

    반환값:
        중복이 제거된 Document 리스트

    예시:
        >>> docs = [
        ...     Document(page_content="Hello", metadata={"id": "1"}),
        ...     Document(page_content="World", metadata={"id": "2"}),
        ...     Document(page_content="Hello", metadata={"id": "1"}),  # 중복
        ... ]
        >>> unique = deduplicate_documents(docs, id_key="id")
        >>> len(unique)
        2
    """
    if id_key is None:
        # page_content 기준 중복 제거
        return list(unique_by_key(documents, lambda d: d.page_content))

    # metadata[id_key] 기준 중복 제거
    return list(
        unique_by_key(
            documents,
            lambda d: d.metadata.get(id_key, d.page_content),
        )
    )


# ============================================================================
# 점수 정규화 유틸리티
# ============================================================================


def softmax_normalize(scores: np.ndarray | list[float]) -> np.ndarray:
    """
    Softmax 정규화를 적용하여 점수를 확률 분포로 변환합니다.

    raw BM25 점수나 유사도 점수를 [0, 1] 범위의 확률로 변환하며,
    합이 1이 되도록 정규화합니다.

    매개변수:
        scores: 정규화할 점수 배열 또는 리스트

    반환값:
        Softmax가 적용된 numpy 배열 (합이 1)

    수식:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

    예시:
        >>> scores = [2.3, 1.8, 4.2]
        >>> normalized = softmax_normalize(scores)
        >>> np.allclose(normalized.sum(), 1.0)
        True
        >>> normalized[2] > normalized[0]  # 가장 높은 점수가 가장 높은 확률
        True
    """
    if isinstance(scores, list):
        scores = np.array(scores, dtype=np.float64)

    # Numerical stability: subtract max before exp
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Min-Max 정규화를 적용하여 점수를 [0, 1] 범위로 변환합니다.

    최소값은 0, 최대값은 1이 되도록 선형 변환합니다.

    매개변수:
        scores: 정규화할 점수 리스트

    반환값:
        [0, 1] 범위로 정규화된 점수 리스트

    수식:
        normalized(x) = (x - min(scores)) / (max(scores) - min(scores))

    예시:
        >>> scores = [10, 20, 30]
        >>> normalized = min_max_normalize(scores)
        >>> normalized
        [0.0, 0.5, 1.0]
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    # 모든 점수가 같으면 1.0 반환
    if max_score == min_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def z_score_normalize(scores: list[float]) -> list[float]:
    """
    Z-score 정규화를 적용하여 점수를 표준화합니다.

    평균이 0, 표준편차가 1이 되도록 변환합니다.
    이상치(outlier)의 영향을 줄이는 데 유용합니다.

    매개변수:
        scores: 정규화할 점수 리스트

    반환값:
        Z-score로 정규화된 점수 리스트

    수식:
        z_score(x) = (x - mean(scores)) / std(scores)

    예시:
        >>> scores = [10, 20, 30]
        >>> normalized = z_score_normalize(scores)
        >>> abs(sum(normalized) / len(normalized)) < 1e-10  # 평균이 0에 가까움
        True
    """
    if not scores:
        return []

    if len(scores) == 1:
        return [0.0]

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = variance**0.5

    # 표준편차가 0이면 모든 점수가 같음
    if std == 0:
        return [0.0] * len(scores)

    return [(score - mean) / std for score in scores]


# ============================================================================
# 문서 병합 유틸리티
# ============================================================================


def merge_document_metadata(
    doc: Document,
    new_metadata: dict,
    *,
    overwrite: bool = True,
) -> Document:
    """
    기존 문서에 새로운 메타데이터를 병합합니다.

    기존 메타데이터를 보존하면서 새로운 필드를 추가하거나 갱신합니다.

    매개변수:
        doc: 원본 Document 객체
        new_metadata: 병합할 메타데이터 딕셔너리
        overwrite: True면 기존 키를 덮어쓰고, False면 보존

    반환값:
        메타데이터가 업데이트된 새로운 Document 객체

    예시:
        >>> doc = Document(
        ...     page_content="Hello",
        ...     metadata={"source": "file1.txt", "score": 0.5}
        ... )
        >>> updated = merge_document_metadata(
        ...     doc,
        ...     {"score": 0.9, "rank": 1},
        ...     overwrite=True
        ... )
        >>> updated.metadata["score"]
        0.9
        >>> updated.metadata["rank"]
        1
    """
    if overwrite:
        # 기존 메타데이터에 새 메타데이터를 덮어씀
        merged_metadata = {**doc.metadata, **new_metadata}
    else:
        # 새 메타데이터를 먼저 두고 기존 메타데이터로 덮어씀 (기존 값 보존)
        merged_metadata = {**new_metadata, **doc.metadata}

    return Document(
        page_content=doc.page_content,
        metadata=merged_metadata,
        type=doc.type,
        id=doc.id,
    )
