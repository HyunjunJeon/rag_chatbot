"""
Adaptive RAG에서 사용하는 검색 도구 모음.

검색 기능을 LangChain 도구 형태로 감싸 에이전트나 워크플로에서 재사용합니다.
"""

from typing import list, Any
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever

from naver_connect_chatbot.config import logger


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


async def retrieve_documents_async(
    retriever: BaseRetriever,
    query: str
) -> list[Document]:
    """
    문서 검색을 비동기로 실행하는 보조 함수입니다.
    
    매개변수:
        retriever: 사용할 리트리버
        query: 검색 질의 문자열
    
    반환값:
        관련 문서 목록
        
    예시:
        >>> retriever = build_dense_sparse_hybrid(...)
        >>> docs = await retrieve_documents_async(retriever, "What is PyTorch?")
    """
    try:
        logger.debug(f"Async retrieving documents for query: {query[:100]}...")
        
        # 리트리버가 비동기를 지원하는지 확인합니다.
        if hasattr(retriever, "ainvoke"):
            documents = await retriever.ainvoke(query)
        else:
            # 지원하지 않으면 동기 호출을 사용합니다.
            documents = retriever.invoke(query)
        
        logger.debug(f"Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Async retrieval error: {e}")
        return []


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

