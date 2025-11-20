"""
다양한 검색 전략을 구성하는 리트리버 팩토리 함수 모음.

제공 기능:
1. Dense+Sparse 하이브리드 리트리버
2. MultiQuery 보강 리트리버
3. 모든 전략을 결합한 종합 검색 파이프라인
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from qdrant_client import QdrantClient

from naver_connect_chatbot.config import settings
from naver_connect_chatbot.rag.retriever.hybrid_retriever import (
    HybridRetriever,
    HybridMethod,
)
from naver_connect_chatbot.rag.retriever.kiwi_bm25_retriever import (
    KiwiBM25Retriever,
)
from naver_connect_chatbot.rag.retriever.multi_query_retriever import (
    MultiQueryRetriever,
)
from naver_connect_chatbot.rag.retriever.qdrant_sdk_retriever import (
    QdrantVDBRetriever,
)


def build_dense_sparse_hybrid(
    documents: list[Document],
    embedding_model: Embeddings,
    qdrant_url: str,
    collection_name: str,
    qdrant_api_key: str | None = None,
    weights: list[float] | None = None,
    k: int | None = None,
    method: HybridMethod = HybridMethod.RRF,
    rrf_c: int | None = None,
    bm25_kwargs: dict[str, Any] | None = None,
) -> HybridRetriever:
    """
    Dense(Qdrant) + Sparse(Kiwi BM25) 하이브리드 검색기를 생성합니다.
    
    이 함수는 기본 하이브리드 검색기를 구성하며, MultiQuery 래핑의 기반이 됩니다.

    매개변수:
        documents: BM25 검색에 사용할 문서 리스트
        embedding_model: Dense 검색에 사용할 임베딩 모델
        qdrant_url: Qdrant 인스턴스 URL
        collection_name: Qdrant 컬렉션 이름
        qdrant_api_key: Qdrant API 키 (선택)
        weights: [Sparse, Dense] 가중치 (기본값: [0.5, 0.5])
        k: 반환할 문서 수
        method: 하이브리드 방식 (RRF 또는 CC)
        rrf_c: RRF 상수 (기본값: 60)
        bm25_kwargs: KiwiBM25Retriever에 전달할 추가 파라미터

    반환값:
        BaseRetriever: Dense + Sparse 하이브리드 검색기
        
    예시:
        >>> base_hybrid = build_dense_sparse_hybrid(
        ...     documents=docs,
        ...     embedding_model=embeddings,
        ...     qdrant_url="http://localhost:6333",
        ...     collection_name="my_collection",
        ...     weights=[0.3, 0.7],  # Sparse 30%, Dense 70%
        ... )
    """
    # 기본값 설정
    if weights is None:
        weights = [
            settings.retriever.default_sparse_weight,
            settings.retriever.default_dense_weight,
        ]
    
    if k is None:
        k = settings.retriever.default_k
    
    if rrf_c is None:
        rrf_c = settings.retriever.default_rrf_c
    
    if bm25_kwargs is None:
        bm25_kwargs = {}

    # 1. Sparse Retriever (Kiwi BM25)
    sparse_retriever = KiwiBM25Retriever.from_documents(
        documents=documents,
        k=k,
        **bm25_kwargs,
    )

    # 2. Dense Retriever (Qdrant)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    dense_retriever = QdrantVDBRetriever(
        client=qdrant_client,
        embedding_model=embedding_model,
        collection_name=collection_name,
        default_k=k,
    )

    # 3. Hybrid Retriever (Sparse + Dense)
    hybrid_retriever = HybridRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=weights,
        method=method,
        c=rrf_c,
    )

    return hybrid_retriever


def build_multi_query_retriever(
    base_retriever: BaseRetriever,
    llm: Runnable,
    num_queries: int | None = None,
    merge_strategy: str | None = None,
    rrf_k: int | None = None,
    include_original: bool | None = None,
) -> MultiQueryRetriever:
    """
    기본 검색기를 MultiQuery로 래핑합니다.
    
    LLM을 사용해 원본 쿼리를 여러 관점으로 확장하고,
    각 쿼리의 검색 결과를 융합하여 더 풍부한 검색 결과를 제공합니다.

    매개변수:
        base_retriever: 기반이 되는 검색기 (보통 Dense+Sparse Hybrid)
        llm: 쿼리 생성에 사용할 LLM (Runnable 인터페이스)
        num_queries: 생성할 쿼리 개수 (기본값: 4)
        merge_strategy: 결과 병합 전략 ("rrf", "max", "sum")
        rrf_k: RRF 상수 (기본값: 60)
        include_original: 원본 쿼리 포함 여부 (기본값: True)

    반환값:
        MultiQueryRetriever: LLM 기반 다중 쿼리 검색기
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4", temperature=0)
        >>> base = build_dense_sparse_hybrid(...)
        >>> multi_query = build_multi_query_retriever(
        ...     base_retriever=base,
        ...     llm=llm,
        ...     num_queries=5,
        ... )
    """
    # 기본값 설정
    if num_queries is None:
        num_queries = settings.multi_query.num_queries
    if merge_strategy is None:
        merge_strategy = settings.multi_query.default_strategy
    if rrf_k is None:
        rrf_k = settings.multi_query.rrf_k
    if include_original is None:
        include_original = settings.multi_query.include_original
    
    return MultiQueryRetriever(
        base_retriever=base_retriever,
        llm=llm,
        num_queries=num_queries,
        merge_strategy=merge_strategy,
        rrf_k=rrf_k,
        include_original=include_original,
    )


def build_advanced_hybrid_retriever(
    documents: list[Document],
    embedding_model: Embeddings,
    qdrant_url: str,
    collection_name: str,
    llm: Runnable | None = None,
    qdrant_api_key: str | None = None,
    base_weights: list[float] | None = None,
    k: int | None = None,
    method: HybridMethod = HybridMethod.RRF,
    rrf_c: int | None = None,
    bm25_kwargs: dict[str, Any] | None = None,
    enable_multi_query: bool = True,
    num_queries: int | None = None,
    multi_query_strategy: str | None = None,
    multi_query_rrf_k: int | None = None,
    include_original: bool | None = None,
    final_weights: list[float] | None = None,
) -> BaseRetriever:
    """
    모든 검색 전략을 결합한 고급 하이브리드 검색기를 생성합니다.
    
    구성 계층:
    1. Base Hybrid: Dense(Qdrant) + Sparse(Kiwi BM25)
    2. MultiQuery: LLM으로 쿼리 확장
    3. Final Hybrid: Base + MultiQuery 결과 융합 (선택적)
    
    MultiQuery가 비활성화되면 Base Hybrid만 반환합니다.

    매개변수:
        documents: 문서 리스트
        embedding_model: 임베딩 모델
        qdrant_url: Qdrant URL
        collection_name: Qdrant 컬렉션 이름
        llm: MultiQuery에 사용할 LLM (None이면 MultiQuery 비활성화)
        qdrant_api_key: Qdrant API 키
        base_weights: Base Hybrid의 [Sparse, Dense] 가중치 (기본값: [0.5, 0.5])
        k: 반환할 문서 수
        method: Base Hybrid의 병합 방식
        rrf_c: Base Hybrid의 RRF 상수
        bm25_kwargs: KiwiBM25Retriever 추가 파라미터
        enable_multi_query: MultiQuery 활성화 여부
        num_queries: 생성할 쿼리 개수
        multi_query_strategy: MultiQuery 병합 전략
        multi_query_rrf_k: MultiQuery RRF 상수
        include_original: 원본 쿼리 포함 여부
        final_weights: Final Hybrid의 [Base, MultiQuery] 가중치 (None이면 MultiQuery만 반환)

    반환값:
        BaseRetriever: 구성된 고급 검색기
        
    예시:
        >>> # MultiQuery 없이 Base Hybrid만 사용
        >>> retriever = build_advanced_hybrid_retriever(
        ...     documents=docs,
        ...     embedding_model=embeddings,
        ...     qdrant_url="http://localhost:6333",
        ...     collection_name="my_collection",
        ...     enable_multi_query=False,
        ... )
        
        >>> # MultiQuery 포함 (단일 계층)
        >>> from langchain_openai import ChatOpenAI
        >>> retriever = build_advanced_hybrid_retriever(
        ...     documents=docs,
        ...     embedding_model=embeddings,
        ...     qdrant_url="http://localhost:6333",
        ...     collection_name="my_collection",
        ...     llm=ChatOpenAI(model="gpt-4"),
        ...     enable_multi_query=True,
        ... )
        
        >>> # MultiQuery + Final Hybrid (이중 계층)
        >>> retriever = build_advanced_hybrid_retriever(
        ...     documents=docs,
        ...     embedding_model=embeddings,
        ...     qdrant_url="http://localhost:6333",
        ...     collection_name="my_collection",
        ...     llm=ChatOpenAI(model="gpt-4"),
        ...     enable_multi_query=True,
        ...     final_weights=[0.4, 0.6],  # Base 40%, MultiQuery 60%
        ... )
    """
    # 1. Base Hybrid 구성 (Dense + Sparse)
    base_hybrid = build_dense_sparse_hybrid(
        documents=documents,
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        qdrant_api_key=qdrant_api_key,
        weights=base_weights,
        k=k,
        method=method,
        rrf_c=rrf_c,
        bm25_kwargs=bm25_kwargs,
    )

    # 2. MultiQuery가 비활성화되었거나 LLM이 없으면 Base만 반환
    if not enable_multi_query or llm is None:
        return base_hybrid

    # 3. MultiQuery Retriever 구성
    try:
        multi_query_retriever = build_multi_query_retriever(
            base_retriever=base_hybrid,
            llm=llm,
            num_queries=num_queries,
            merge_strategy=multi_query_strategy,
            rrf_k=multi_query_rrf_k,
            include_original=include_original,
        )
    except Exception as e:
        # LLM 초기화 실패 시 Base Hybrid로 폴백
        print(f"⚠️  MultiQuery 초기화 실패, Base Hybrid만 사용: {e}")
        return base_hybrid

    # 4. Final Hybrid 구성 여부 결정
    if final_weights is None:
        # Final Hybrid 없이 MultiQuery만 반환
        return multi_query_retriever

    # 5. Final Hybrid: Base + MultiQuery 결과 융합
    final_hybrid = HybridRetriever(
        retrievers=[base_hybrid, multi_query_retriever],
        weights=final_weights,
        method=method,
        c=rrf_c,
    )

    return final_hybrid


def get_hybrid_retriever(
    documents: list[Document],
    embedding_model: Embeddings,
    qdrant_url: str,
    collection_name: str,
    qdrant_api_key: str | None = None,
    weights: list[float] | None = None,
    k: int = 10,
) -> HybridRetriever:
    """
    Dense + Sparse 하이브리드 리트리버를 생성하는 레거시 팩토리입니다.
    
    하위 호환성을 위해 유지되며 내부적으로 build_dense_sparse_hybrid를 호출합니다.

    매개변수:
        documents: BM25 검색에 사용할 문서 리스트
        embedding_model: Qdrant Dense 검색에 사용할 임베딩 모델
        qdrant_url: Qdrant 인스턴스 URL
        collection_name: Qdrant 컬렉션 이름
        qdrant_api_key: Qdrant API 키
        weights: [Sparse, Dense] 리트리버 가중치
        k: 검색해 반환할 문서 수

    반환값:
        HybridRetriever: 구성된 하이브리드 리트리버
    """
    return build_dense_sparse_hybrid(
        documents=documents,
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        qdrant_api_key=qdrant_api_key,
        weights=weights,
        k=k,
    )
