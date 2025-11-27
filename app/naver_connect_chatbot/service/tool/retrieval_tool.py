"""
Adaptive RAG에서 사용하는 검색 도구 모음.

검색 기능을 LangChain 도구 형태로 감싸 에이전트나 워크플로에서 재사용합니다.
"""

from typing import Any, TypedDict
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever

from naver_connect_chatbot.config import logger


# 순환 참조 방지를 위해 RetrievalFilters를 여기서 정의
# (원본은 naver_connect_chatbot.service.graph.types에 있음)
class RetrievalFilters(TypedDict, total=False):
    """메타 기반 검색 필터 (순환 참조 방지용 복사본)."""
    doc_type: list[str]
    course: str
    course_level: str
    course_topic: str
    generation: str
    year: str
    year_month: str


def filter_documents_by_metadata(
    documents: list[Document],
    filters: RetrievalFilters,
) -> list[Document]:
    """
    메타데이터 기반으로 문서를 필터링합니다.

    매개변수:
        documents: 필터링할 문서 목록
        filters: 적용할 필터 조건

    반환값:
        필터 조건을 만족하는 문서 목록
    """
    if not filters:
        return documents

    filtered = []
    for doc in documents:
        metadata = doc.metadata or {}
        match = True

        # doc_type 필터 (리스트, OR 조건)
        if "doc_type" in filters and filters["doc_type"]:
            doc_type = metadata.get("doc_type")
            if doc_type not in filters["doc_type"]:
                match = False

        # 문자열 필터들 (정확히 일치)
        string_filters = ["course", "course_level", "course_topic", "generation", "year", "year_month"]
        for key in string_filters:
            if key in filters and filters[key]:
                if metadata.get(key) != filters[key]:
                    match = False
                    break

        if match:
            filtered.append(doc)

    return filtered


def create_retrieval_tool(retriever: BaseRetriever) -> Any:
    """
    리트리버를 LangChain 도구로 감쌉니다.
    
    에이전트나 워크플로에서 호출 가능한 검색 도구를 제공합니다.
    
    매개변수:
        retriever: 도구로 감쌀 리트리버
    
    반환값:
        검색을 수행하는 LangChain 도구
        
    예시:
        >>> from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> tool = create_retrieval_tool(retriever)
        >>> docs = tool.invoke("What is PyTorch?")
    """
    @tool()
    def retrieve_documents_tool(query: str) -> list[Document]:
        """
        Retrieve documents that are relevant to the incoming query.

        Parameters:
            query: Search query string supplied by the caller.

        Returns:
            List of documents ranked by relevance.
        """
        try:
            logger.debug(f"Retrieving documents for query: {query[:100]}...")
            documents = retriever.invoke(query)
            logger.debug(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    return retrieve_documents_tool


def create_multi_query_retrieval_tool(
    retriever: BaseRetriever,
) -> Any:
    """
    다중 질의 검색 도구를 생성합니다.
    
    여러 질의를 받아 각각 검색한 뒤 결과를 중복 제거하여 반환합니다.
    
    매개변수:
        retriever: 사용할 리트리버
    
    반환값:
        다중 질의 검색을 수행하는 LangChain 도구
        
    예시:
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> tool = create_multi_query_retrieval_tool(retriever)
        >>> docs = tool.invoke(["What is PyTorch?", "PyTorch features"])
    """
    @tool()
    def retrieve_multi_query_tool(queries: list[str]) -> list[Document]:
        """
        Retrieve documents that are relevant to the incoming queries.

        Parameters:
            queries: List of search query strings supplied by the caller.

        Returns:
            List of documents ranked by relevance, with duplicates removed.
        """
        try:
            logger.debug(f"Retrieving documents for {len(queries)} queries")
            
            all_docs = []
            seen_contents = set()
            
            for query in queries:
                documents = retriever.invoke(query)
                
                # 문서 내용을 기준으로 중복을 제거합니다.
                for doc in documents:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_docs.append(doc)
            
            logger.debug(f"Retrieved {len(all_docs)} unique documents")
            return all_docs
            
        except Exception as e:
            logger.error(f"Multi-query retrieval error: {e}")
            return []
    
    return retrieve_multi_query_tool


class RetrievalResult:
    """검색 결과와 메타데이터를 담는 클래스."""

    def __init__(
        self,
        documents: list[Document],
        filters_applied: bool = False,
        fallback_used: bool = False,
        original_count: int = 0,
        filtered_count: int = 0,
    ):
        self.documents = documents
        self.filters_applied = filters_applied
        self.fallback_used = fallback_used
        self.original_count = original_count
        self.filtered_count = filtered_count


async def retrieve_documents_async(
    retriever: BaseRetriever,
    query: str,
    filters: RetrievalFilters | None = None,
    fallback_on_empty: bool = True,
    min_results: int = 1,
) -> RetrievalResult:
    """
    문서 검색을 비동기로 실행하고 메타데이터 기반 필터링을 적용합니다.

    매개변수:
        retriever: 사용할 리트리버
        query: 검색 질의 문자열
        filters: 메타데이터 필터 (선택적)
        fallback_on_empty: 필터 적용 후 0건일 때 필터 없이 재시도할지 여부
        min_results: 폴백을 트리거하는 최소 결과 수

    반환값:
        RetrievalResult: 검색 결과와 메타데이터

    예시:
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> result = await retrieve_documents_async(
        ...     retriever,
        ...     "What is PyTorch?",
        ...     filters={"doc_type": ["slack_qa"]}
        ... )
        >>> print(f"Found {len(result.documents)} docs, fallback: {result.fallback_used}")
    """
    try:
        logger.debug(f"Async retrieving documents for query: {query[:100]}...")

        # 리트리버가 비동기를 지원하는지 확인합니다.
        if hasattr(retriever, "ainvoke"):
            documents = await retriever.ainvoke(query)
        else:
            # 지원하지 않으면 동기 호출을 사용합니다.
            documents = retriever.invoke(query)

        original_count = len(documents)
        logger.debug(f"Retrieved {original_count} documents (before filtering)")

        # 필터가 없으면 바로 반환
        if not filters:
            return RetrievalResult(
                documents=documents,
                filters_applied=False,
                fallback_used=False,
                original_count=original_count,
                filtered_count=original_count,
            )

        # 필터 적용
        filtered_documents = filter_documents_by_metadata(documents, filters)
        filtered_count = len(filtered_documents)
        logger.info(f"Filtered {original_count} → {filtered_count} documents with filters: {filters}")

        # 폴백 로직: 결과가 min_results 미만이면 필터 없이 반환
        if filtered_count < min_results and fallback_on_empty:
            logger.warning(
                f"Filter resulted in {filtered_count} docs (< {min_results}), "
                f"falling back to unfiltered results ({original_count} docs)"
            )
            return RetrievalResult(
                documents=documents,
                filters_applied=False,
                fallback_used=True,
                original_count=original_count,
                filtered_count=filtered_count,
            )

        return RetrievalResult(
            documents=filtered_documents,
            filters_applied=True,
            fallback_used=False,
            original_count=original_count,
            filtered_count=filtered_count,
        )

    except Exception as e:
        logger.error(f"Async retrieval error: {e}")
        return RetrievalResult(
            documents=[],
            filters_applied=False,
            fallback_used=False,
            original_count=0,
            filtered_count=0,
        )


async def retrieve_multi_query_async(
    retriever: BaseRetriever,
    queries: list[str]
) -> list[Document]:
    """
    다중 질의 검색을 비동기로 수행하는 보조 함수입니다.
    
    매개변수:
        retriever: 사용할 리트리버
        queries: 검색 질의 문자열 목록
    
    반환값:
        중복 제거된 관련 문서 목록
        
    예시:
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> docs = await retrieve_multi_query_async(retriever, ["Query 1", "Query 2"])
    """
    try:
        logger.debug(f"Async retrieving documents for {len(queries)} queries")
        
        all_docs = []
        seen_contents = set()
        
        for query in queries:
            if hasattr(retriever, "ainvoke"):
                documents = await retriever.ainvoke(query)
            else:
                documents = retriever.invoke(query)
            
            # 문서 내용을 기준으로 중복을 제거합니다.
            for doc in documents:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        logger.debug(f"Retrieved {len(all_docs)} unique documents")
        return all_docs
        
    except Exception as e:
        logger.error(f"Async multi-query retrieval error: {e}")
        return []

