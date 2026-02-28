"""
Google Search Grounding 기반 웹 검색 문서 추출 모듈.

Gemini의 google_search tool binding 응답에서 grounding 메타데이터를 파싱하여
LangChain Document 객체로 변환합니다.

grounding 메타데이터 구조:
    response.additional_kwargs = {
        "grounding_metadata": {
            "web_search_queries": ["검색어1", ...],
            "grounding_chunks": [
                {"web": {"uri": "https://...", "title": "제목"}},
            ],
            "grounding_supports": [
                {
                    "segment": {"text": "지원 텍스트"},
                    "grounding_chunk_indices": [0, 1],
                    "confidence_scores": [0.95, 0.87],
                },
            ],
        }
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from naver_connect_chatbot.config import logger

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings


def extract_grounding_documents(response: AIMessage) -> list[Document]:
    """
    Gemini 응답에서 grounding 메타데이터를 Document 리스트로 변환합니다.

    grounding_chunks → URL/제목 (metadata)
    grounding_supports → segment text (page_content) + confidence_scores

    매개변수:
        response: Gemini의 AIMessage 응답 (google_search tool 사용 시)

    반환값:
        grounding 정보가 담긴 Document 리스트.
        grounding 메타데이터가 없으면 빈 리스트 반환.
    """
    additional_kwargs: dict[str, Any] = getattr(response, "additional_kwargs", {})
    grounding_metadata = additional_kwargs.get("grounding_metadata", {})

    if not grounding_metadata:
        logger.debug("No grounding metadata found in response")
        return []

    chunks: list[dict[str, Any]] = grounding_metadata.get("grounding_chunks", [])
    supports: list[dict[str, Any]] = grounding_metadata.get("grounding_supports", [])

    if not chunks and not supports:
        logger.debug("Empty grounding_chunks and grounding_supports")
        return []

    # chunk 인덱스 → (uri, title) 매핑 테이블 구축
    chunk_map: dict[int, dict[str, str]] = {}
    for idx, chunk in enumerate(chunks):
        web_info = chunk.get("web", {})
        chunk_map[idx] = {
            "uri": web_info.get("uri", ""),
            "title": web_info.get("title", ""),
        }

    documents: list[Document] = []

    if supports:
        # supports가 있으면: segment.text를 page_content로, 연결된 chunk 정보를 metadata로
        for support in supports:
            segment = support.get("segment", {})
            text = segment.get("text", "").strip()
            if not text:
                continue

            chunk_indices = support.get("grounding_chunk_indices", [])
            confidence_scores = support.get("confidence_scores", [])

            # 가장 높은 confidence를 가진 chunk 정보 사용
            best_chunk_idx = chunk_indices[0] if chunk_indices else -1
            best_confidence = confidence_scores[0] if confidence_scores else 0.0

            # 여러 chunk가 연결된 경우 가장 높은 confidence 선택
            for i, score in enumerate(confidence_scores):
                if score > best_confidence and i < len(chunk_indices):
                    best_confidence = score
                    best_chunk_idx = chunk_indices[i]

            chunk_info = chunk_map.get(best_chunk_idx, {"uri": "", "title": ""})

            doc = Document(
                page_content=text,
                metadata={
                    "source_type": "web_search",
                    "url": chunk_info["uri"],
                    "title": chunk_info["title"],
                    "grounding_confidence": best_confidence,
                    "grounding_chunk_indices": chunk_indices,
                },
            )
            documents.append(doc)
    else:
        # supports가 없고 chunks만 있는 경우: 제목을 page_content로 사용
        for idx, chunk in enumerate(chunks):
            web_info = chunk.get("web", {})
            title = web_info.get("title", "").strip()
            uri = web_info.get("uri", "")
            if not title:
                continue

            doc = Document(
                page_content=title,
                metadata={
                    "source_type": "web_search",
                    "url": uri,
                    "title": title,
                    "grounding_confidence": 0.0,
                },
            )
            documents.append(doc)

    logger.info(
        f"Extracted {len(documents)} grounding documents "
        f"(chunks={len(chunks)}, supports={len(supports)})"
    )
    return documents


async def google_search_retrieve(
    query: str,
    llm_settings: GeminiLLMSettings,
    *,
    max_results: int = 10,
) -> list[Document]:
    """
    Gemini Google Search grounding을 사용하여 웹 검색 결과를 Document로 반환합니다.

    1. Gemini + google_search tool binding으로 검색 수행
    2. response.additional_kwargs에서 grounding metadata 추출
    3. extract_grounding_documents()로 Document 변환
    4. 각 Document에 source_type="web_search" 메타데이터 태깅

    매개변수:
        query: 검색 질의 문자열
        llm_settings: GeminiLLMSettings 인스턴스 (API 키, 모델명 등)
        max_results: 반환할 최대 문서 수 (기본값: 10)

    반환값:
        웹 검색 결과를 담은 Document 리스트.
        검색 실패 또는 grounding 없는 경우 빈 리스트 반환.
    """
    if not llm_settings.api_key:
        logger.warning("GOOGLE_API_KEY not set, skipping web search")
        return []

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Google Search grounding용 LLM (thinking 불필요 → 빠른 응답)
        search_llm = ChatGoogleGenerativeAI(
            model=llm_settings.model,
            google_api_key=llm_settings.api_key.get_secret_value(),
            temperature=llm_settings.temperature,
        )
        search_tool = {"google_search": {}}
        llm_with_search = search_llm.bind_tools([search_tool])

        # 검색 프롬프트: grounding을 유도하는 간결한 질의
        prompt = (
            f"다음 질문에 대한 최신 정보를 웹에서 검색하여 답변해주세요.\n\n"
            f"질문: {query}"
        )

        response = await llm_with_search.ainvoke(prompt)

        # grounding 메타데이터 추출
        documents = extract_grounding_documents(response)

        # max_results 제한
        if len(documents) > max_results:
            documents = documents[:max_results]

        logger.info(f"Google Search retrieved {len(documents)} documents for: {query[:80]}...")
        return documents

    except Exception as e:
        logger.error(f"Google Search retrieval failed: {e}")
        return []


def create_google_search_tool(llm_settings: GeminiLLMSettings) -> Any:
    """
    LLM Agent용 Google Search 도구를 생성합니다.

    Gemini의 Google Search grounding을 사용하여 웹 검색 후
    결과를 포맷팅된 문자열로 반환합니다.

    매개변수:
        llm_settings: GeminiLLMSettings 인스턴스 (API 키, 모델명 등)

    반환값:
        LangChain @tool 데코레이터가 적용된 async 함수
    """

    @tool
    async def web_search(query: str) -> str:
        """웹에서 최신 정보를 검색합니다. 교육 자료에 없는 일반 개념이나 최신 정보를 보충할 때 사용하세요."""
        docs = await google_search_retrieve(query, llm_settings)

        if not docs:
            return f"웹 검색 결과 없음: '{query}'에 대한 정보를 찾지 못했습니다."

        # 상위 5개 결과만 사용
        docs = docs[:5]

        parts = [f"[웹 검색 결과: {len(docs)}건]"]
        for doc in docs:
            title = doc.metadata.get("title", "")
            url = doc.metadata.get("url", "")
            if title and url:
                parts.append(f"[웹: {title}] ({url})\n{doc.page_content}")
            elif title:
                parts.append(f"[웹: {title}]\n{doc.page_content}")
            else:
                parts.append(doc.page_content)

        return "\n\n".join(parts)

    return web_search


__all__ = [
    "extract_grounding_documents",
    "google_search_retrieve",
    "create_google_search_tool",
]
