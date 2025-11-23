"""
Retriever를 LangChain Tool로 변환하는 유틸리티 모듈.

이 모듈은 BaseRetriever를 LangChain agent가 사용할 수 있는 Tool로
변환하는 편의 함수를 제공합니다.

LangChain v1.0+의 create_retriever_tool을 활용하여 retriever를
agent-compatible tool로 변환합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import Tool
from langchain_core.tools.retriever import create_retriever_tool

if TYPE_CHECKING:
    from naver_connect_chatbot.rag.rerank import BaseReranker

__all__ = [
    "create_hybrid_retriever_tool",
    "create_reranked_retriever_tool",
    "create_retriever_tool_from_factory",
]


def create_hybrid_retriever_tool(
    retriever: BaseRetriever,
    name: str = "search_knowledge_base",
    description: str | None = None,
) -> Tool:
    """
    Hybrid retriever를 LangChain tool로 변환합니다.

    LangChain agent가 retriever를 사용할 수 있도록
    create_retriever_tool을 활용하여 Tool 객체로 변환합니다.

    매개변수:
        retriever: 변환할 BaseRetriever 인스턴스
            (HybridRetriever, MultiQueryRetriever, KiwiBM25Retriever 등)
        name: Tool 이름 (LLM이 tool 선택 시 사용)
            - snake_case 권장
            - 고유하고 설명적인 이름 사용
        description: Tool 설명 (LLM이 tool 사용 시기 판단에 활용)
            - None이면 자동 생성
            - 구체적이고 명확한 설명 권장

    반환값:
        LangChain agent에서 사용 가능한 Tool 객체

    예시:
        >>> from naver_connect_chatbot.rag import build_dense_sparse_hybrid
        >>> from naver_connect_chatbot.rag.tools import create_hybrid_retriever_tool
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent
        >>>
        >>> # 1. Retriever 생성
        >>> retriever = build_dense_sparse_hybrid(
        ...     documents=docs,
        ...     embedding_model=embeddings,
        ...     qdrant_url="http://localhost:6333",
        ...     collection_name="my_docs",
        ... )
        >>>
        >>> # 2. Tool로 변환
        >>> search_tool = create_hybrid_retriever_tool(
        ...     retriever=retriever,
        ...     name="search_technical_docs",
        ...     description=(
        ...         "Search technical documentation for answers to programming questions. "
        ...         "Use this when you need specific technical information."
        ...     ),
        ... )
        >>>
        >>> # 3. Agent에 등록
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = create_react_agent(llm, tools=[search_tool])
        >>>
        >>> # 4. Agent 실행
        >>> response = agent.invoke({"messages": [("user", "How do I configure the retriever?")]})

    참고:
        - Tool name은 LLM이 이해하기 쉬운 이름으로 설정하세요
        - Description은 언제 이 tool을 사용해야 하는지 명확히 작성하세요
        - 여러 retriever를 다른 이름으로 tool로 등록하여 multi-tool agent 구성 가능
    """
    if description is None:
        # Retriever 클래스 이름 기반 자동 설명 생성
        retriever_type = retriever.__class__.__name__
        description = (
            f"Search the knowledge base using {retriever_type}. "
            f"Returns relevant documents to answer questions. "
            f"Useful when you need factual information from the knowledge base."
        )

    return create_retriever_tool(
        retriever=retriever,
        name=name,
        description=description,
    )


def create_reranked_retriever_tool(
    retriever: BaseRetriever,
    reranker: BaseReranker,
    name: str = "search_with_reranking",
    description: str | None = None,
    *,
    top_k: int = 5,
) -> Tool:
    """
    Retrieval + Reranking을 결합한 tool을 생성합니다.

    Retriever로 문서를 검색한 후 Reranker로 재정렬하여
    더 높은 품질의 검색 결과를 제공하는 tool을 만듭니다.

    매개변수:
        retriever: 문서 검색에 사용할 BaseRetriever
        reranker: 검색 결과 재정렬에 사용할 BaseReranker
            (예: ClovaStudioReranker)
        name: Tool 이름
        description: Tool 설명 (None이면 자동 생성)
        top_k: Reranking 후 반환할 상위 문서 수

    반환값:
        Retrieval + Reranking이 결합된 Tool 객체

    예시:
        >>> from naver_connect_chatbot.rag import build_dense_sparse_hybrid
        >>> from naver_connect_chatbot.rag.rerank import ClovaStudioReranker
        >>> from naver_connect_chatbot.rag.tools import create_reranked_retriever_tool
        >>> from naver_connect_chatbot.config import settings
        >>>
        >>> # 1. Retriever 및 Reranker 생성
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> reranker = ClovaStudioReranker.from_settings(settings.reranker)
        >>>
        >>> # 2. Reranked tool 생성
        >>> search_tool = create_reranked_retriever_tool(
        ...     retriever=retriever,
        ...     reranker=reranker,
        ...     name="search_high_quality_docs",
        ...     description="Search for highly relevant documents with semantic reranking",
        ...     top_k=3,
        ... )
        >>>
        >>> # 3. Agent에서 사용
        >>> agent = create_react_agent(llm, tools=[search_tool])

    작동 방식:
        1. Retriever가 초기 문서 검색 수행 (예: 20개)
        2. Reranker가 검색 결과를 재정렬
        3. 상위 top_k개 문서만 반환

        이 방식은 retrieval recall을 유지하면서
        precision을 크게 향상시킵니다.
    """
    class RerankedRetrieverWrapper(BaseRetriever):
        """
        Retrieval + Reranking을 결합한 wrapper retriever.

        이 클래스는 내부적으로 사용되며, 사용자는 직접 인스턴스화할 필요 없습니다.
        """

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None,
        ) -> list[Document]:
            """동기 retrieval + reranking."""
            # 1. Initial retrieval
            docs = retriever.invoke(query)

            # 2. Reranking
            if docs:
                return reranker.rerank(query, docs, top_k=top_k)
            return []

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None,
        ) -> list[Document]:
            """비동기 retrieval + reranking."""
            # 1. Initial retrieval
            docs = await retriever.ainvoke(query)

            # 2. Reranking
            if docs:
                return await reranker.arerank(query, docs, top_k=top_k)
            return []

    # Wrapper retriever 생성
    wrapped_retriever = RerankedRetrieverWrapper()

    # Description 생성
    if description is None:
        description = (
            f"Search the knowledge base with semantic reranking. "
            f"Returns top {top_k} highly relevant documents ranked by relevance score. "
            f"Use this when you need the most accurate and relevant information."
        )

    return create_retriever_tool(
        retriever=wrapped_retriever,
        name=name,
        description=description,
    )


def create_retriever_tool_from_factory(
    factory_func,
    factory_kwargs: dict,
    tool_name: str,
    tool_description: str,
) -> Tool:
    """
    Factory 함수로 retriever를 생성하고 즉시 tool로 변환합니다.

    이 함수는 retriever 생성과 tool 변환을 한 번에 수행하는
    편의 함수입니다.

    매개변수:
        factory_func: Retriever를 생성하는 factory 함수
            (예: build_dense_sparse_hybrid, build_multi_query_retriever)
        factory_kwargs: Factory 함수에 전달할 인자 딕셔너리
        tool_name: 생성할 tool의 이름
        tool_description: 생성할 tool의 설명

    반환값:
        Factory로 생성된 retriever를 감싼 Tool 객체

    예시:
        >>> from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid
        >>> from naver_connect_chatbot.rag.tools import create_retriever_tool_from_factory
        >>>
        >>> # Factory 함수와 인자 준비
        >>> factory_kwargs = {
        ...     "documents": docs,
        ...     "embedding_model": embeddings,
        ...     "qdrant_url": "http://localhost:6333",
        ...     "collection_name": "my_docs",
        ... }
        >>>
        >>> # Retriever 생성 + Tool 변환을 한 번에
        >>> search_tool = create_retriever_tool_from_factory(
        ...     factory_func=build_dense_sparse_hybrid,
        ...     factory_kwargs=factory_kwargs,
        ...     tool_name="search_docs",
        ...     tool_description="Search documentation for technical answers",
        ... )

    참고:
        이 함수는 설정 기반 tool 생성 시 유용합니다.
        예를 들어, YAML 설정 파일에서 tool 구성을 로드하여
        동적으로 agent를 구성할 수 있습니다.
    """
    # Factory 함수로 retriever 생성
    retriever = factory_func(**factory_kwargs)

    # Tool로 변환
    return create_retriever_tool(
        retriever=retriever,
        name=tool_name,
        description=tool_description,
    )
