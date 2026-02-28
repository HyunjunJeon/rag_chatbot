"""
Adaptive RAG에서 사용하는 검색 도구 모음.

검색 기능을 LangChain 도구 형태로 감싸 에이전트나 워크플로에서 재사용합니다.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from naver_connect_chatbot.config import logger

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings
    from naver_connect_chatbot.rag.rerank import ClovaStudioReranker
    from naver_connect_chatbot.service.graph.types import RetrievalFilters


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

        # course 필터 (리스트, OR 조건) - 하위 호환성 유지
        if "course" in filters and filters["course"]:
            course_filter = filters["course"]
            # 하위 호환: str이면 list로 변환 (deprecation warning)
            if isinstance(course_filter, str):
                logger.warning(
                    f"DEPRECATED: course filter should be list[str], got str: '{course_filter}'. "
                    "Auto-converting for backward compatibility. "
                    "Please update your code to use list format."
                )
                course_filter = [course_filter]
            doc_course = metadata.get("course")
            if doc_course not in course_filter:
                match = False

        # 문자열 필터들 (정확히 일치) - course 제외
        string_filters = ["course_level", "course_topic", "generation", "year", "year_month"]
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


# ============================================================================
# 문서 라벨 빌더 (안정 식별자)
# ============================================================================

DOC_TYPE_DISPLAY = {
    "pdf": "강의자료",
    "notebook": "실습노트북",
    "slack_qa": "Slack Q&A",
    "lecture_transcript": "강의녹취록",
    "weekly_mission": "주간미션",
}


def _build_document_label(doc: Document, index: int) -> str:
    """
    메타데이터 기반 안정 식별자를 생성합니다.

    메타데이터의 doc_type, course, lecture_num, topic 등을 조합하여
    멀티턴에서도 일관된 문서 라벨을 생성합니다.
    메타데이터가 없으면 [문서 N] 형태로 폴백합니다.

    매개변수:
        doc: LangChain Document 객체
        index: 문서 인덱스 (0-based, 폴백용)

    반환값:
        "[강의자료: CV 이론/3강]" 같은 형태의 문서 라벨
    """
    metadata = doc.metadata or {}
    doc_type = metadata.get("doc_type", "")
    doc_type_label = DOC_TYPE_DISPLAY.get(doc_type, doc_type)
    course = metadata.get("course", "")

    if not doc_type_label and not course:
        return f"[문서 {index + 1}]"

    parts = []
    if doc_type_label:
        parts.append(doc_type_label)

    detail_parts = []
    if course:
        detail_parts.append(course)

    lecture_num = metadata.get("lecture_num", "")
    topic = metadata.get("topic", "")
    if lecture_num:
        detail_parts.append(f"{lecture_num}강")
    elif topic:
        detail_parts.append(topic)

    if parts and detail_parts:
        return f"[{parts[0]}: {'/'.join(detail_parts)}]"
    elif parts:
        return f"[{parts[0]}]"
    elif detail_parts:
        return f"[{'/'.join(detail_parts)}]"

    return f"[문서 {index + 1}]"


# ============================================================================
# Tool-based Retrieval (LLM Agent용)
# ============================================================================


def create_qdrant_search_tool(
    retriever: BaseRetriever,
    reranker_settings: Any | None = None,
) -> Any:
    """
    LLM Agent용 Qdrant 검색 도구를 생성합니다.

    InjectedState를 통해 analyze_query_node가 추출한 retrieval_filters에 접근합니다.
    검색 → 필터링 → Reranking → 포맷팅을 하나의 도구 호출로 완료합니다.

    매개변수:
        retriever: Qdrant 기반 BaseRetriever
        reranker_settings: ClovaStudioRerankerSettings (None이면 reranking 생략)

    반환값:
        LangChain @tool 데코레이터가 적용된 async 함수
    """

    @tool
    async def qdrant_search(
        query: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """부스트캠프 교육 자료를 검색합니다. 강의자료, 실습 노트북, Slack Q&A, 미션 등에서 관련 정보를 찾습니다."""
        filters = state.get("retrieval_filters")
        if filters:
            logger.info(f"Qdrant search with filters: {filters}")

        # 1. 검색 + 메타데이터 필터링
        result: RetrievalResult = await retrieve_documents_async(
            retriever, query, filters=filters, fallback_on_empty=True, min_results=1
        )

        if not result.documents:
            return f"검색 결과 없음: '{query}'에 대한 관련 교육 자료를 찾지 못했습니다."

        documents = result.documents

        # 2. Reranking (설정이 있을 때만)
        if reranker_settings:
            try:
                from naver_connect_chatbot.rag.rerank import ClovaStudioReranker

                reranker = ClovaStudioReranker.from_settings(reranker_settings)
                documents = await reranker.arerank(
                    query=query,
                    documents=documents,
                    top_k=min(len(documents), 10),
                )
                logger.info(f"Reranked to {len(documents)} documents")
            except Exception as e:
                logger.warning(f"Reranking failed, using original order: {e}")

        # 3. 문서 포맷팅 (라벨 + 본문)
        parts = [f"[검색 결과: {len(documents)}건]"]
        for i, doc in enumerate(documents):
            label = _build_document_label(doc, i)
            parts.append(f"{label}\n{doc.page_content}")

        return "\n\n".join(parts)

    return qdrant_search


# ============================================================================
# Dual Source Retrieval (Qdrant + Web Search)
# ============================================================================


class DualRetrievalResult:
    """Qdrant + WebSearch 이중 검색 결과를 담는 클래스."""

    def __init__(
        self,
        documents: list[Document],
        qdrant_count: int = 0,
        web_count: int = 0,
        duplicates_removed: int = 0,
        web_search_activated: bool = False,
    ):
        self.documents = documents
        self.qdrant_count = qdrant_count
        self.web_count = web_count
        self.duplicates_removed = duplicates_removed
        self.web_search_activated = web_search_activated


def _tag_source_type(documents: list[Document], source_type: str) -> list[Document]:
    """문서 리스트에 source_type 메타데이터를 태깅합니다 (이미 있으면 유지)."""
    for doc in documents:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata.setdefault("source_type", source_type)
    return documents


def _deduplicate_documents(documents: list[Document]) -> tuple[list[Document], int]:
    """
    page_content hash 기반으로 중복 문서를 제거합니다.

    반환값:
        (중복 제거된 문서 리스트, 제거된 문서 수)
    """
    seen: set[int] = set()
    unique: list[Document] = []
    for doc in documents:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    removed = len(documents) - len(unique)
    return unique, removed


async def retrieve_dual_source_async(
    query: str,
    retriever: BaseRetriever,
    llm_settings: GeminiLLMSettings,
    *,
    filters: RetrievalFilters | None = None,
    enable_web_search: bool = True,
    min_qdrant_results: int = 2,
) -> DualRetrievalResult:
    """
    Qdrant VectorDB + Google WebSearch 이중 검색 후 병합합니다.

    전략:
    - enable_web_search=True이면 항쪽 모두 병렬 검색
    - enable_web_search=False이면 Qdrant만 사용
    - 결과를 source_type으로 태깅하여 병합
    - page_content hash 기반 중복 제거

    매개변수:
        query: 검색 질의 문자열
        retriever: Qdrant 기반 BaseRetriever
        llm_settings: GeminiLLMSettings (WebSearch용)
        filters: 메타데이터 필터 (Qdrant 검색에만 적용)
        enable_web_search: WebSearch 활성화 여부
        min_qdrant_results: 이 값 미만이면 WebSearch 강제 활성화

    반환값:
        DualRetrievalResult: 병합된 검색 결과
    """
    # 1. Qdrant 검색
    qdrant_result = await retrieve_documents_async(
        retriever, query, filters=filters, fallback_on_empty=True
    )
    qdrant_docs = _tag_source_type(qdrant_result.documents, "qdrant")

    # Qdrant 결과가 부족하면 WebSearch 강제 활성화
    if len(qdrant_docs) < min_qdrant_results:
        enable_web_search = True

    # 2. WebSearch (조건부 병렬 실행)
    web_docs: list[Document] = []
    web_search_activated = False

    if enable_web_search:
        web_search_activated = True
        try:
            from naver_connect_chatbot.rag.web_search import google_search_retrieve

            web_docs = await google_search_retrieve(query, llm_settings)
            # web_search 모듈에서 이미 source_type 태깅하지만, 안전하게 한번 더
            web_docs = _tag_source_type(web_docs, "web_search")
        except Exception as e:
            logger.warning(f"WebSearch failed, continuing with Qdrant only: {e}")
            web_docs = []

    # 3. 병합 + 중복 제거
    combined = qdrant_docs + web_docs
    merged_docs, duplicates_removed = _deduplicate_documents(combined)

    logger.info(
        f"Dual retrieval: qdrant={len(qdrant_docs)}, web={len(web_docs)}, "
        f"merged={len(merged_docs)}, dedup={duplicates_removed}"
    )

    return DualRetrievalResult(
        documents=merged_docs,
        qdrant_count=len(qdrant_docs),
        web_count=len(web_docs),
        duplicates_removed=duplicates_removed,
        web_search_activated=web_search_activated,
    )


# ============================================================================
# Rerank + Score Cutoff Filter
# ============================================================================


async def rerank_and_filter(
    query: str,
    documents: list[Document],
    reranker: ClovaStudioReranker,
    *,
    top_k: int = 10,
    min_rank: int = 5,
) -> list[Document]:
    """
    ClovaStudio Reranker로 재순위화 후 rank 기반 필터링합니다.

    ClovaStudio Reranker 특성:
    - API가 순위(rank) 기반으로 점수를 합성: 1등=1.0, 2등=0.9, ...
    - 따라서 score cutoff보다 rank cutoff가 더 안정적
    - min_rank=5 → rank 1~5 문서만 반환 (상위 50% at top_k=10)

    매개변수:
        query: 사용자 질의
        documents: 재순위화할 문서 리스트
        reranker: ClovaStudioReranker 인스턴스
        top_k: reranker에 전달할 상위 문서 수
        min_rank: 이 rank 이하만 통과 (1=최고, 5=상위 50%)

    반환값:
        rank 필터링이 적용된 문서 리스트.
        metadata에 rerank_score, rerank_rank 포함.

    예외:
        빈 문서 리스트이면 빈 리스트 반환 (reranker 호출 안 함).
    """
    if not documents:
        return []

    # ClovaStudio Reranker 호출
    reranked = await reranker.arerank(query=query, documents=documents, top_k=top_k)

    # rank cutoff 필터링
    filtered = [
        doc for doc in reranked
        if doc.metadata.get("rerank_rank", float("inf")) <= min_rank
    ]

    logger.info(
        f"Rerank filter: {len(documents)} → reranked {len(reranked)} → "
        f"rank<={min_rank} kept {len(filtered)}"
    )

    return filtered


# ============================================================================
# Full Validated Pipeline
# ============================================================================


class ValidatedRetrievalResult:
    """전체 파이프라인 결과: Dual Retrieval → Rerank → Filter."""

    def __init__(
        self,
        verified_documents: list[Document],
        all_documents: list[Document],
        qdrant_count: int = 0,
        web_count: int = 0,
        reranked_count: int = 0,
        filtered_count: int = 0,
        source_distribution: dict[str, int] | None = None,
    ):
        self.verified_documents = verified_documents
        self.all_documents = all_documents
        self.qdrant_count = qdrant_count
        self.web_count = web_count
        self.reranked_count = reranked_count
        self.filtered_count = filtered_count
        self.source_distribution = source_distribution or {}


async def retrieve_and_validate(
    query: str,
    retriever: BaseRetriever,
    reranker: ClovaStudioReranker,
    llm_settings: GeminiLLMSettings,
    *,
    filters: RetrievalFilters | None = None,
    enable_web_search: bool = True,
    min_qdrant_results: int = 2,
    rerank_top_k: int = 10,
    rerank_min_rank: int = 5,
) -> ValidatedRetrievalResult:
    """
    전체 파이프라인: Dual Retrieval → Rerank → Filter → Validated Documents.

    매개변수:
        query: 검색 질의 문자열
        retriever: Qdrant 기반 BaseRetriever
        reranker: ClovaStudioReranker 인스턴스
        llm_settings: GeminiLLMSettings (WebSearch용)
        filters: 메타데이터 필터 (Qdrant 검색에만 적용)
        enable_web_search: WebSearch 활성화 여부
        min_qdrant_results: Qdrant 결과 최소 수 (미만 시 WebSearch 강제)
        rerank_top_k: reranker에 전달할 상위 문서 수
        rerank_min_rank: rank cutoff (이하만 통과)

    반환값:
        ValidatedRetrievalResult: 검증된 최종 결과
    """
    # Step 1: Dual Source Retrieval
    dual_result = await retrieve_dual_source_async(
        query=query,
        retriever=retriever,
        llm_settings=llm_settings,
        filters=filters,
        enable_web_search=enable_web_search,
        min_qdrant_results=min_qdrant_results,
    )

    all_documents = dual_result.documents

    # Step 2: Rerank + Filter (문서가 있을 때만)
    if all_documents:
        verified = await rerank_and_filter(
            query=query,
            documents=all_documents,
            reranker=reranker,
            top_k=rerank_top_k,
            min_rank=rerank_min_rank,
        )
    else:
        verified = []

    # Step 3: source_distribution 계산
    source_distribution = dict(
        Counter(doc.metadata.get("source_type", "unknown") for doc in verified)
    )

    logger.info(
        f"Validated pipeline: dual={len(all_documents)} → "
        f"verified={len(verified)}, distribution={source_distribution}"
    )

    return ValidatedRetrievalResult(
        verified_documents=verified,
        all_documents=all_documents,
        qdrant_count=dual_result.qdrant_count,
        web_count=dual_result.web_count,
        reranked_count=len(verified),
        filtered_count=len(verified),
        source_distribution=source_distribution,
    )

