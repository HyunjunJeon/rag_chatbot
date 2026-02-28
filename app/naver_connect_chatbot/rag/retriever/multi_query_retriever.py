"""
LLM을 활용해 다양한 변형 질의를 생성하고, 각 검색 결과를 결합하여
최종 순위 리스트를 만드는 리트리버 구현.
"""

from __future__ import annotations

import asyncio
import re

from collections.abc import Iterable
from typing import Final

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.prompts import get_prompt

DocumentList = list[Document]
DocumentMatrix = list[DocumentList]


class MultiQueryOutput(BaseModel):
    queries: list[str]


class MultiQueryRetriever(BaseRetriever):
    """
    LLM으로 다각도의 검색 쿼리를 생성하고, 기본 검색기의 결과를 결합해
    단일 순위 목록으로 반환하는 커스텀 멀티 쿼리 검색기입니다.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(description="Base retriever")
    llm: Runnable = Field(description="LLM for query generation")
    num_queries: int = Field(default=4, description="Number of queries to generate")
    merge_strategy: str = Field(default="rrf", description="Merge strategy: rrf, max, sum")
    rrf_k: int = Field(default=60, description="RRF constant")
    include_original: bool = Field(default=True, description="Include original query")

    _SUPPORTED_STRATEGIES: Final[frozenset[str]] = frozenset({"rrf", "max", "sum"})

    def _generate_queries(self, query: str) -> list[str]:
        """LLM으로 생성한 다양한 질의를 중복 없이 수집합니다."""
        return self._generate_queries_sync(query)

    def _generate_queries_sync(self, query: str) -> list[str]:
        _prompt = get_prompt("multi_query_generation")

        try:
            with_structured = getattr(self.llm, "with_structured_output", None)
            if callable(with_structured):
                structured_llm = with_structured(MultiQueryOutput)
                prompt_value = _prompt.format_prompt(query=query, num=self.num_queries)
                input_text = prompt_value.to_string()
                result = structured_llm.invoke(input_text)
                queries = result.queries
            else:
                raise AttributeError("with_structured_output not supported")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "MultiQuery structured generation failed, falling back to legacy pipeline",
                error=str(exc),
            )
            chain: Runnable = _prompt | self.llm | StrOutputParser()
            try:
                legacy_result = chain.invoke({"query": query, "num": self.num_queries})
                # 개선된 파서 사용: 다양한 형식 (번호, 불릿 등) 지원
                queries = self._parse_legacy_queries(legacy_result)
                if not queries:
                    logger.warning(
                        "Legacy parser returned no queries, using original query",
                        raw_output=legacy_result[:200],
                    )
                    return [query]
            except Exception as legacy_exc:  # pragma: no cover
                logger.error(
                    "MultiQuery generation failed, using original query only",
                    error=str(legacy_exc),
                )
                return [query]

        trimmed = queries[: self.num_queries]
        ordered = [query, *trimmed] if self.include_original else trimmed
        return self._deduplicate_queries(ordered)

    async def _agenerate_queries(self, query: str) -> list[str]:
        _prompt = get_prompt("multi_query_generation")

        try:
            with_structured = getattr(self.llm, "with_structured_output", None)
            if callable(with_structured):
                structured_llm = with_structured(MultiQueryOutput)
                prompt_value = _prompt.format_prompt(query=query, num=self.num_queries)
                input_text = prompt_value.to_string()
                result = await structured_llm.ainvoke(input_text)
                queries = result.queries
            else:
                raise AttributeError("with_structured_output not supported")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Async MultiQuery structured generation failed, falling back to sync path",
                error=str(exc),
            )
            return await asyncio.to_thread(self._generate_queries_sync, query)

        trimmed = queries[: self.num_queries]
        ordered = [query, *trimmed] if self.include_original else trimmed
        return self._deduplicate_queries(ordered)

    def _merge_results(
        self,
        results_list: DocumentMatrix,
    ) -> DocumentList:
        """
        설정된 융합 전략에 따라 다중 검색 결과를 결합합니다.
        """
        strategy = (self.merge_strategy or "rrf").casefold()
        if strategy not in self._SUPPORTED_STRATEGIES:
            strategy = "rrf"

        if strategy == "rrf":
            return self._rrf_merge(results_list)
        if strategy == "max":
            return self._max_merge(results_list)
        return self._sum_merge(results_list)

    def _rrf_merge(self, results_list: DocumentMatrix) -> DocumentList:
        """
        Reciprocal Rank Fusion을 적용해 결과를 병합합니다.
        """
        doc_scores: dict[int, float] = {}
        doc_objects: dict[int, Document] = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = hash(doc.page_content)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        fused: DocumentList = []
        for doc_id, score in sorted_docs:
            doc = doc_objects[doc_id]
            metadata = {**doc.metadata, "rrf_score": score}
            fused.append(Document(page_content=doc.page_content, metadata=metadata))
        return fused

    def _max_merge(self, results_list: DocumentMatrix) -> DocumentList:
        """
        문서별 최고 점수를 선택하여 결과를 병합합니다.
        """
        doc_scores: dict[int, float] = {}
        doc_objects: dict[int, Document] = {}

        for results in results_list:
            for doc in results:
                doc_id = hash(doc.page_content)
                score = doc.metadata.get("score", 0.0)
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score
                    doc_objects[doc_id] = doc

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [doc_objects[doc_id] for doc_id, _ in sorted_docs]

    def _sum_merge(self, results_list: DocumentMatrix) -> DocumentList:
        """
        문서별 점수를 누적하여 결과를 병합합니다.
        """
        doc_scores: dict[int, float] = {}
        doc_objects: dict[int, Document] = {}

        for results in results_list:
            for doc in results:
                doc_id = hash(doc.page_content)
                score = doc.metadata.get("score", 1.0)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc
                doc_scores[doc_id] += score

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [doc_objects[doc_id] for doc_id, _ in sorted_docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> DocumentList:
        """
        동기 환경에서 다중 쿼리 검색 파이프라인을 실행합니다.
        """
        queries = self._generate_queries(query)
        results_list = [self._invoke_base_retriever(candidate) for candidate in queries]
        merged = self._merge_results(results_list)
        if run_manager is not None:
            run_manager.on_retriever_end(merged)
        return merged

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> DocumentList:
        """비동기 환경에서 다중 쿼리 검색 파이프라인을 실행합니다."""
        queries = await self._agenerate_queries(query)
        tasks = [self._ainvoke_base_retriever(candidate) for candidate in queries]
        results_list = await asyncio.gather(*tasks)
        merged = self._merge_results(results_list)
        if run_manager is not None:
            await run_manager.on_retriever_end(merged)
        return merged

    def _invoke_base_retriever(self, query: str) -> DocumentList:
        """단일 질의에 대해 기본 리트리버를 동기적으로 호출합니다."""
        return self.base_retriever.invoke(query)

    async def _ainvoke_base_retriever(self, query: str) -> DocumentList:
        """단일 질의에 대해 기본 리트리버를 비동기로 호출합니다."""
        ainvoke = getattr(self.base_retriever, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(query)
        return await asyncio.to_thread(self.base_retriever.invoke, query)

    @staticmethod
    def _deduplicate_queries(queries: Iterable[str]) -> list[str]:
        """대소문자 구분 없이 질의 순서를 유지하며 중복을 제거합니다."""
        seen: set[str] = set()
        unique_queries: list[str] = []
        for candidate in queries:
            normalized = candidate.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_queries.append(candidate)
        return unique_queries

    @staticmethod
    def _parse_legacy_queries(text: str, min_length: int = 5) -> list[str]:
        """
        다양한 형식의 LLM 출력을 파싱하여 쿼리 리스트를 반환합니다.

        지원 형식:
        - 줄바꿈 구분: "쿼리1\\n쿼리2"
        - 번호 형식: "1. 쿼리1\\n2. 쿼리2"
        - 불릿 형식: "- 쿼리1\\n- 쿼리2"
        - 혼합 형식

        Args:
            text: LLM의 raw 출력 텍스트
            min_length: 유효한 쿼리의 최소 길이 (기본 5자)

        Returns:
            파싱된 쿼리 리스트
        """
        lines = text.strip().split("\n")
        queries: list[str] = []

        # 접두사 제거 패턴: "1.", "1)", "1:", "-", "*", "•", ">"
        prefix_pattern = re.compile(r"^[\d]+[.\):\s]+|^[-*•>]\s*")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 접두사 제거
            cleaned = prefix_pattern.sub("", line).strip()

            # 너무 짧은 쿼리 스킵
            if len(cleaned) < min_length:
                continue

            # 헤더/코멘트 스킵 (# 로 시작하거나 : 로 끝나는 경우)
            if cleaned.startswith("#") or cleaned.endswith(":"):
                continue

            # 마크다운 코드블록 스킵
            if cleaned.startswith("```") or cleaned.endswith("```"):
                continue

            queries.append(cleaned)

        return queries
