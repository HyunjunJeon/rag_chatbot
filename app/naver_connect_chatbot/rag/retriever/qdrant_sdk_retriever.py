"""
공식 Qdrant Python SDK(qdrant-client)를 사용해 벡터 검색을 수행하고,
LangChain `BaseRetriever`를 확장하여 하이브리드 검색 파이프라인과 호환되도록 합니다.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter as QdrantFilter,
    NamedVector,
    ScoredPoint,
    SearchParams,
)


class QdrantVDBRetriever(BaseRetriever):
    """공식 Qdrant SDK를 활용해 밀집 벡터 검색을 수행하는 리트리버입니다.

    매개변수:
        client: 초기화된 `QdrantClient` 인스턴스
        embedding_model: 질의 임베딩에 사용할 LangChain `Embeddings` 모델
        collection_name: 검색 대상 Qdrant 컬렉션 이름
        vector_name: 다중 벡터 컬렉션에서 사용할 벡터 이름 (기본값 None)
        default_k: 검색 시 반환할 기본 문서 수
        payload_content_keys: payload에서 본문 텍스트를 우선적으로 찾을 키 순서
        search_params: Qdrant 고급 검색 파라미터 (`SearchParams`)
        query_filter: Qdrant 필터 객체
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: QdrantClient = Field(repr=False)
    embedding_model: Embeddings = Field(repr=False)
    collection_name: str
    vector_name: str | None = None
    default_k: int = 10
    payload_content_keys: tuple[str, ...] = ("page_content", "text", "content", "document")
    search_params: SearchParams | None = Field(default=None, repr=False)
    query_filter: QdrantFilter | None = Field(default=None, repr=False)

    def _build_query_vector(self, embedding: Sequence[float]) -> Sequence[float] | NamedVector:
        """NamedVector 필요 여부에 따라 질의 벡터 형식을 결정합니다."""
        if self.vector_name is None:
            return embedding
        return NamedVector(name=self.vector_name, vector=list(embedding))

    def _search(self, query_embedding: Sequence[float], limit: int) -> list[ScoredPoint]:
        """Qdrant `query_points` API를 호출하여 점수화된 포인트를 반환합니다."""
        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": list(query_embedding),  # query_points는 query 파라미터 사용
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if self.search_params is not None:
            search_kwargs["search_params"] = self.search_params
        if self.query_filter is not None:
            search_kwargs["query_filter"] = self.query_filter  # filter -> query_filter

        return self.client.query_points(**search_kwargs).points

    def _normalize_payload(self, payload: Any) -> dict[str, Any]:
        """Qdrant 포인트의 payload를 dict 형식으로 정규화합니다."""
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        return {"payload": payload}

    def _extract_page_content(self, payload: dict[str, Any]) -> str:
        """payload에서 문서 본문 텍스트를 추출합니다."""
        for key in self.payload_content_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        for value in payload.values():
            if isinstance(value, str) and value.strip():
                return value

        # 텍스트를 찾지 못한 경우 payload 전체를 문자열로 반환한다.
        return str(payload)

    def _point_to_document(self, point: ScoredPoint) -> Document:
        """Qdrant 검색 결과 포인트를 LangChain Document로 변환합니다."""
        payload = self._normalize_payload(point.payload)
        page_content = self._extract_page_content(payload)

        metadata = dict(payload)
        metadata.setdefault("id", point.id)
        metadata["score"] = point.score

        return Document(page_content=page_content, metadata=metadata)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """동기 검색 구현."""
        query_embedding = self.embedding_model.embed_query(query)
        points = self._search(query_embedding, self.default_k)
        return [self._point_to_document(point) for point in points]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """비동기 검색 구현 (스레드 풀에서 동기 검색을 실행)."""
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)
