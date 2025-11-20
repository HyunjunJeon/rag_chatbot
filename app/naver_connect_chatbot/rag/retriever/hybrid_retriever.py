"""
여러 리트리버의 결과를 가중 Reciprocal Rank Fusion 또는 Convex Combination으로
결합하는 하이브리드 리트리버 구현.
"""

import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    TypeVar,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from pydantic import model_validator
from enum import Enum

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


class HybridMethod(str, Enum):
    RRF = "rrf"
    CC = "cc"


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """키 함수 기준으로 중복을 제거하며 요소를 순차적으로 반환합니다.

    매개변수:
        iterable: 중복 제거를 적용할 반복 가능 객체
        key: 각 요소에서 해시 가능한 키를 추출하는 함수

    Yields:
        key 함수 기준으로 고유한 요소
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class HybridRetriever(BaseRetriever):
    """여러 리트리버의 결과를 결합하는 하이브리드 리트리버입니다.

    Reciprocal Rank Fusion(RRF) 또는 Convex Combination(CC) 방식을 지원합니다.

    매개변수:
        retrievers: 결합 대상 리트리버 목록
        weights: 각 리트리버에 대응하는 가중치 (CC 모드는 합이 1이어야 함)
        method: 사용할 하이브리드 방식 ("rrf" 또는 "cc")
        c: RRF 계산에 사용하는 상수 (기본값 60)
        id_key: 문서 중복 여부를 판별할 metadata 키 (없으면 page_content 사용)
    """

    retrievers: list[RetrieverLike]
    weights: list[float]
    method: HybridMethod = HybridMethod.RRF
    c: int = 60
    id_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_weights(cls, values: dict[str, Any]) -> Any:
        weights = values.get("weights")
        method = values.get("method", HybridMethod.RRF)

        if not weights:
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        elif method == HybridMethod.CC and abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0 for CC method")

        return values

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """이 Runnable이 제공하는 설정 가능한 필드 목록을 반환합니다."""
        return get_unique_config_specs(spec for retriever in self.retrievers for spec in retriever.config_specs)

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def ainvoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        from langchain_core.callbacks import AsyncCallbackManager

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        주어진 질의에 대한 관련 문서를 동기적으로 조회합니다.

        매개변수:
            query: 검색할 질의

        반환값:
            재순위화된 문서 리스트
        """

        # 각 리트리버 결과를 합성합니다.
        fused_documents = self.rank_fusion(query, run_manager)

        return fused_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        주어진 질의에 대한 관련 문서를 비동기로 조회합니다.

        매개변수:
            query: 검색할 질의

        반환값:
            재순위화된 문서 리스트
        """

        # 각 리트리버 결과를 합성합니다.
        fused_documents = await self.arank_fusion(query, run_manager)

        return fused_documents

    def hybrid_results(self, doc_lists: list[list[Document]]) -> list[Document]:
        """
        RRF 또는 CC 방식으로 각 리트리버 결과를 결합합니다.

        매개변수:
            doc_lists: 각 리트리버가 반환한 순위 리스트 목록

        반환값:
            list: 점수 내림차순으로 정렬된 최종 결과 리스트
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError("Number of rank lists must be equal to the number of weights.")

        if self.method == HybridMethod.RRF:
            return self.reciprocal_rank_fusion(doc_lists)
        elif self.method == HybridMethod.CC:
            return self.convex_combination(doc_lists)
        else:
            raise ValueError("Invalid hybrid method")

    def reciprocal_rank_fusion(self, doc_lists: list[list[Document]]) -> list[Document]:
        """
        여러 랭크 리스트에 Reciprocal Rank Fusion을 수행합니다.
        """
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                doc_id = doc.page_content if self.id_key is None else doc.metadata[self.id_key]
                rrf_score[doc_id] += weight / (rank + self.c)

        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: (doc.page_content if self.id_key is None else doc.metadata[self.id_key]),
            ),
            key=lambda doc: rrf_score[doc.page_content if self.id_key is None else doc.metadata[self.id_key]],
            reverse=True,
        )
        return sorted_docs

    def convex_combination(self, doc_lists: list[list[Document]]) -> list[Document]:
        """
        여러 랭크 리스트에 Convex Combination을 수행합니다.
        """
        cc_scores: dict[str, float] = defaultdict(float)

        for doc_list, weight in zip(doc_lists, self.weights):
            max_score = max(doc.metadata.get("score", 0) for doc in doc_list) or 1
            for doc in doc_list:
                doc_id = doc.page_content if self.id_key is None else doc.metadata[self.id_key]
                normalized_score = doc.metadata.get("score", 0) / max_score
                cc_scores[doc_id] += weight * normalized_score

        all_docs = list(
            unique_by_key(
                chain.from_iterable(doc_lists),
                lambda doc: (doc.page_content if self.id_key is None else doc.metadata[self.id_key]),
            )
        )

        sorted_docs = sorted(
            all_docs,
            key=lambda doc: cc_scores[doc.page_content if self.id_key is None else doc.metadata[self.id_key]],
            reverse=True,
        )

        return sorted_docs

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        # 모든 리트리버의 검색 결과를 수집합니다.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(config, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # 각 리스트의 항목을 Document 인스턴스로 통일합니다.
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc for doc in retriever_docs[i]
            ]

        # 앙상블 방식을 적용해 결과를 결합합니다.
        fused_documents = self.hybrid_results(retriever_docs)

        return fused_documents

    async def arank_fusion(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        # 모든 리트리버의 비동기 검색 결과를 수집합니다.
        retriever_docs = await asyncio.gather(*[
            retriever.ainvoke(
                query,
                patch_config(config, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")),
            )
            for i, retriever in enumerate(self.retrievers)
        ])

        # 각 리스트의 항목을 Document 인스턴스로 통일합니다.
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=doc) if not isinstance(doc, Document) else doc  # type: ignore[arg-type]
                for doc in retriever_docs[i]
            ]

        # 앙상블 방식을 적용해 결과를 결합합니다.
        fused_documents = self.hybrid_results(retriever_docs)

        return fused_documents
