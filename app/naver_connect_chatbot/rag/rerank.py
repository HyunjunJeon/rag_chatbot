"""
Naver Clova Studio Reranker 구현 모듈
이 모듈은 검색된 문서의 우선순위를 재정렬하는 Reranker 인터페이스와 구현체를 제공합니다.

LangChain 0.1.x의 DocumentCompressor 패턴:
    - compress_documents(documents, query) -> list[Document]
    - 질문과 문서를 입력받아 관련도에 따라 재정렬
    - 메타데이터에 score/relevance 정보 추가

Clova Studio Reranker API 사양:
    - 엔드포인트: https://clovastudio.stream.ntruss.com/v1/api-tools/reranker
    - 실제 API 문서: https://api.ncloud-docs.com/docs/clovastudio-reranker
    - 인증 헤더:
        * Content-Type: application/json (필수)
        * Authorization: Bearer <api-key> (필수)
    - 요청 페이로드:
        {
            "query": str,
            "documents": [
                {"id": str, "doc": str},
                ...
            ],
            "maxTokens": int  # 선택사항, 기본값: 1024
        }
    - 응답 스키마:
        {
            "status": {
                "code": str,
                "message": str
            },
            "result": {
                "result": str,  # RAG 생성 답변
                "citedDocuments": [
                    {"id": str, "doc": str},
                    ...
                ],
                "usage": {
                    "promptTokens": int,
                    "completionTokens": int,
                    "totalTokens": int
                }
            }
        }
    - 특징:
        * citedDocuments 배열의 순서가 관련도 순서를 나타냄
        * 최대 100개의 문서까지 처리 가능
        * RAG 기반으로 동작하며, 관련 문서를 재정렬하여 반환

사용 예:
    from naver_connect_chatbot.rag.rerank import ClovaStudioReranker
    from naver_connect_chatbot.config import settings

    reranker = ClovaStudioReranker.from_settings(settings.reranker)
    reranked_docs = reranker.rerank(query="질문", documents=docs, top_k=5)
"""

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.documents import Document
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.rag.utils import should_retry_http_error


# ============================================================================
# 유틸리티 함수
# ============================================================================


def _generate_document_id(doc: Document, id_key: str | None = None) -> str:
    """
    문서의 고유 ID를 생성합니다.

    메타데이터에 id_key가 있으면 사용하고, 없으면 page_content의 해시를 생성합니다.

    매개변수:
        doc: LangChain Document 객체
        id_key: 메타데이터에서 ID로 사용할 키 (기본값: None)

    반환값:
        문서의 고유 ID 문자열

    예시:
        >>> doc = Document(page_content="Hello", metadata={"source_id": "doc1"})
        >>> _generate_document_id(doc, "source_id")
        'doc1'
        >>> _generate_document_id(doc, None)  # fallback to hash
        '8b1a9953c4611296a827abf8c47804d7'
    """
    if id_key and id_key in doc.metadata:
        return str(doc.metadata[id_key])

    # Fallback: page_content의 MD5 해시 사용
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return content_hash


def _serialize_documents_for_api(
    documents: Sequence[Document],
) -> list[dict[str, str]]:
    """
    LangChain Document 리스트를 Clova Studio Reranker API 형식으로 변환합니다.

    새로운 API 스펙:
        각 문서를 {"id": "문서ID", "doc": "문서내용"} 형식으로 변환합니다.
        - id: 문서의 고유 식별자 (인덱스 기반으로 생성)
        - doc: 문서의 실제 텍스트 내용

    매개변수:
        documents: LangChain Document 시퀀스

    반환값:
        {"id": str, "doc": str} 형식의 딕셔너리 리스트

    Time Complexity:
        O(n), n은 문서 수

    예시:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> _serialize_documents_for_api(docs)
        [{"id": "doc_0", "doc": "Hello"}, {"id": "doc_1", "doc": "World"}]
    """
    return [
        {
            "id": f"doc_{idx}",
            "doc": doc.page_content,
        }
        for idx, doc in enumerate(documents)
    ]


def _merge_rerank_metadata(
    doc: Document,
    score: float,
    rank: int,
) -> Document:
    """
    재정렬 결과(점수, 순위)를 문서의 메타데이터에 병합합니다.

    기존 메타데이터를 보존하면서 다음 필드를 추가/갱신합니다:
        - rerank_score: Reranker가 부여한 관련도 점수
        - rerank_rank: 재정렬 후 순위 (1-based)
        - score: LangChain 표준 필드에도 rerank_score 복사

    매개변수:
        doc: 원본 Document 객체
        score: Reranker가 부여한 점수
        rank: 재정렬 후 순위 (1-based)

    반환값:
        메타데이터가 업데이트된 새로운 Document 객체

    예시:
        >>> doc = Document(page_content="Hello", metadata={"source": "file.txt"})
        >>> updated = _merge_rerank_metadata(doc, 0.95, 1)
        >>> updated.metadata["rerank_score"]
        0.95
        >>> updated.metadata["rerank_rank"]
        1
    """
    new_metadata = doc.metadata.copy()
    new_metadata.update(
        {
            "rerank_score": score,
            "rerank_rank": rank,
            "score": score,  # LangChain 표준 필드
        }
    )

    return Document(
        page_content=doc.page_content,
        metadata=new_metadata,
    )


# ============================================================================
# Reranker 인터페이스
# ============================================================================


class BaseReranker(ABC):
    """
    문서 재정렬을 위한 추상 베이스 클래스입니다.

    LangChain의 DocumentCompressor와 유사한 역할을 수행하며,
    질문과 문서 리스트를 입력받아 관련도에 따라 재정렬된 문서를 반환합니다.

    구현 클래스는 다음 메서드를 반드시 구현해야 합니다:
        - rerank: 동기 재정렬 메서드
        - arerank: 비동기 재정렬 메서드 (선택적)

    Time Complexity:
        O(n), n은 문서 수 (API 호출 시간 제외)
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        질문과 문서 리스트를 입력받아 관련도 순으로 재정렬합니다.

        매개변수:
            query: 사용자 질문 텍스트
            documents: 재정렬할 Document 시퀀스
            top_k: 반환할 상위 문서 수 (None이면 모든 문서 반환)

        반환값:
            관련도 내림차순으로 정렬된 Document 리스트
            각 문서의 metadata에는 rerank_score, rerank_rank가 추가됨

        예외:
            ValueError: query가 빈 문자열이거나 documents가 비어있는 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생

        예시:
            >>> reranker = SomeReranker()
            >>> docs = [Document(page_content="AI is ..."), ...]
            >>> reranked = reranker.rerank("What is AI?", docs, top_k=3)
            >>> len(reranked)
            3
            >>> reranked[0].metadata["rerank_score"] > reranked[1].metadata["rerank_score"]
            True
        """
        raise NotImplementedError

    async def arerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        비동기 방식으로 문서를 재정렬합니다.

        기본 구현은 동기 메서드를 호출하므로, 진정한 비동기 처리가 필요한 경우
        하위 클래스에서 오버라이드해야 합니다.

        매개변수:
            query: 사용자 질문 텍스트
            documents: 재정렬할 Document 시퀀스
            top_k: 반환할 상위 문서 수

        반환값:
            관련도 내림차순으로 정렬된 Document 리스트

        예외:
            ValueError: query가 빈 문자열이거나 documents가 비어있는 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생
        """
        # 기본 구현: 동기 메서드 호출
        return self.rerank(query, documents, top_k=top_k)


# ============================================================================
# Clova Studio Reranker 구현체
# ============================================================================


class ClovaStudioReranker(BaseReranker):
    """
    Naver Clova Studio Reranker API를 활용한 문서 재정렬 구현체입니다.

    HTTPX를 사용하여 REST API를 호출하고, 응답을 LangChain Document로 변환합니다.

    속성:
        endpoint: Clova Studio Reranker API 엔드포인트 URL
        api_key: CLOVASTUDIO_API_KEY (Authorization Bearer 토큰)
        max_tokens: 최대 생성 토큰 수
        request_timeout: HTTP 요청 타임아웃 (초)
        client: HTTPX Client 인스턴스 (재사용)

    예시:
        >>> reranker = ClovaStudioReranker(
        ...     endpoint="https://clovastudio.stream.ntruss.com/v1/api-tools/reranker",
        ...     api_key="your-api-key",
        ...     max_tokens=1024,
        ... )
        >>> docs = [Document(page_content="AI is ..."), ...]
        >>> reranked = reranker.rerank("What is AI?", docs)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        max_tokens: int = 1024,
        request_timeout: float = 60.0,
    ) -> None:
        """
        ClovaStudioReranker를 초기화합니다.

        매개변수:
            endpoint: Clova Studio Reranker API 엔드포인트
            api_key: Clova Studio API 키 (CLOVASTUDIO_API_KEY)
            max_tokens: 최대 생성 토큰 수 (기본값: 1024, 최대: 4096)
            request_timeout: 요청 타임아웃 (초, 기본값: 60초)

        예외:
            ValueError: endpoint나 api_key가 비어있는 경우
        """
        if not endpoint:
            msg = "endpoint는 빈 문자열일 수 없습니다"
            raise ValueError(msg)
        if not api_key:
            msg = "api_key는 빈 문자열일 수 없습니다"
            raise ValueError(msg)

        self.endpoint = endpoint
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

        # HTTPX Client 초기화 (세션 재사용)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "ClovaStudioReranker":
        """
        Settings 객체로부터 ClovaStudioReranker를 생성합니다.

        매개변수:
            settings: ClovaStudioRerankerSettings 또는 호환 객체

        반환값:
            초기화된 ClovaStudioReranker 인스턴스

        예시:
            >>> from naver_connect_chatbot.config import settings
            >>> reranker = ClovaStudioReranker.from_settings(settings.reranker)
        """
        return cls(
            endpoint=settings.endpoint,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            max_tokens=getattr(settings, "max_tokens", 1024),
            request_timeout=getattr(settings, "request_timeout", 60.0),
        )

    @property
    def client(self) -> httpx.Client:
        """HTTPX Client를 lazy 생성하여 반환합니다."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout)
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """HTTPX AsyncClient를 lazy 생성하여 반환합니다."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._async_client

    def _build_headers(self) -> dict[str, str]:
        """
        API 요청 헤더를 생성합니다.

        새로운 Clova Studio API는 표준 Bearer 토큰 방식을 사용합니다:
        - Content-Type: application/json (필수)
        - Authorization: Bearer <api-key> (필수)

        반환값:
            HTTP 헤더 딕셔너리
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _validate_inputs(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> None:
        """
        입력 데이터의 유효성을 검증합니다.

        매개변수:
            query: 질문 텍스트
            documents: 문서 시퀀스

        예외:
            ValueError: 유효하지 않은 입력인 경우
        """
        if not query or not query.strip():
            msg = "query는 빈 문자열일 수 없습니다"
            raise ValueError(msg)

        if not documents:
            msg = "documents는 비어있을 수 없습니다"
            raise ValueError(msg)

        if len(documents) > 100:
            msg = "Clova Studio Reranker는 최대 100개의 문서까지 처리 가능합니다"
            logger.warning(msg, document_count=len(documents))

    def _parse_response(
        self,
        response_data: dict[str, Any],
        original_documents: Sequence[Document],
    ) -> list[Document]:
        """
        Clova Studio Reranker API 응답을 파싱하여 LangChain Document 리스트로 변환합니다.

        API 응답 구조:
        {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "result": "답변 텍스트",
                "citedDocuments": [
                    {"id": "doc_0", "doc": "문서내용..."},
                    {"id": "doc_1", "doc": "문서내용..."}
                ],
                "usage": {...}
            }
        }

        citedDocuments 배열의 순서가 관련도 순서를 나타냅니다.

        매개변수:
            response_data: API 응답 JSON 데이터
            original_documents: 원본 Document 시퀀스

        반환값:
            재정렬된 Document 리스트

        예외:
            ValueError: 응답 스키마가 예상과 다른 경우
        """
        try:
            result = response_data["result"]
            cited_documents = result["citedDocuments"]
            usage = result.get("usage", {})  # 토큰 사용량 (선택적)
        except KeyError as e:
            msg = f"API 응답 스키마가 올바르지 않습니다: {e}"
            raise ValueError(msg) from e

        # 사용량 정보 로깅 (디버그 레벨)
        if usage:
            logger.debug(
                "Reranker API 토큰 사용량",
                prompt_tokens=usage.get("promptTokens", 0),
                completion_tokens=usage.get("completionTokens", 0),
                total_tokens=usage.get("totalTokens", 0),
            )

        # 재정렬된 문서 리스트 생성
        reranked_docs: list[Document] = []

        for rank, item in enumerate(cited_documents, start=1):
            try:
                doc_id = item["id"]
                # doc_id는 "doc_{idx}" 형식이므로 인덱스 추출
                if not doc_id.startswith("doc_"):
                    logger.warning(
                        f"예상치 못한 문서 ID 형식: {doc_id}",
                        doc_id=doc_id,
                        rank=rank,
                    )
                    continue

                index = int(doc_id.split("_")[1])
            except (KeyError, ValueError, IndexError) as e:
                msg = f"문서 항목의 ID 파싱 실패: {e}"
                logger.error(msg, item=item, rank=rank)
                continue

            # 원본 문서 가져오기
            if index < 0 or index >= len(original_documents):
                msg = f"유효하지 않은 문서 인덱스: {index}"
                logger.error(msg, index=index, total_docs=len(original_documents))
                continue

            original_doc = original_documents[index]

            # 메타데이터 병합 (score는 없으므로 순위 기반으로 점수 생성)
            # 1등: 1.0, 2등: 0.9, 3등: 0.8, ...
            score = max(0.0, 1.0 - (rank - 1) * 0.1)
            reranked_doc = _merge_rerank_metadata(original_doc, score, rank)
            reranked_docs.append(reranked_doc)

        return reranked_docs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    def _call_reranker_api(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """
        Reranker API를 호출합니다 (retry 로직 포함).

        이 메서드는 tenacity를 사용하여 다음과 같은 retry 전략을 적용합니다:
        - 최대 3회 재시도
        - 지수 백오프 (2초 ~ 10초)
        - TimeoutException, NetworkError, HTTPStatusError (5xx만) 시 재시도
        - 4xx 클라이언트 오류는 즉시 실패 (재시도 안 함)
        - 재시도 전 WARNING 레벨로 로깅

        매개변수:
            payload: API 요청 페이로드
            headers: HTTP 헤더

        반환값:
            HTTP 응답 객체

        예외:
            httpx.HTTPStatusError: 4xx 클라이언트 오류 또는 모든 재시도 실패 후 5xx 오류
            httpx.TimeoutException: 모든 재시도 실패 후 타임아웃 오류
            httpx.NetworkError: 모든 재시도 실패 후 네트워크 오류
        """
        response = self.client.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    async def _acall_reranker_api(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """
        Reranker API를 비동기로 호출합니다 (retry 로직 포함).

        이 메서드는 tenacity를 사용하여 다음과 같은 retry 전략을 적용합니다:
        - 최대 3회 재시도
        - 지수 백오프 (2초 ~ 10초)
        - TimeoutException, NetworkError, HTTPStatusError (5xx만) 시 재시도
        - 4xx 클라이언트 오류는 즉시 실패 (재시도 안 함)
        - 재시도 전 WARNING 레벨로 로깅

        매개변수:
            payload: API 요청 페이로드
            headers: HTTP 헤더

        반환값:
            HTTP 응답 객체

        예외:
            httpx.HTTPStatusError: 4xx 클라이언트 오류 또는 모든 재시도 실패 후 5xx 오류
            httpx.TimeoutException: 모든 재시도 실패 후 타임아웃 오류
            httpx.NetworkError: 모든 재시도 실패 후 네트워크 오류
        """
        response = await self.async_client.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Clova Studio API를 호출하여 문서를 재정렬합니다.

        매개변수:
            query: 사용자 질문
            documents: 재정렬할 문서 시퀀스

        반환값:
            관련도 내림차순으로 정렬된 Document 리스트

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패

        Time Complexity:
            O(n + m), n은 문서 수, m은 API 응답 시간
        """
        # 입력 검증
        self._validate_inputs(query, documents)

        # 빈 문서 리스트는 즉시 반환
        if len(documents) == 0:
            return []

        # API 요청 페이로드 구성
        payload = {
            "query": query,
            "documents": _serialize_documents_for_api(documents),
            "maxTokens": self.max_tokens,
        }

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Reranker API 호출",
            request_id=request_id,
            query_length=len(query),
            document_count=len(documents),
            max_tokens=self.max_tokens,
        )

        try:
            response = self._call_reranker_api(payload, headers)
        except httpx.HTTPStatusError as e:
            msg = f"API 호출 실패 (HTTP {e.response.status_code})"
            logger.error(
                msg,
                request_id=request_id,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise RuntimeError(msg) from e
        except httpx.RequestError as e:
            msg = f"API 요청 중 오류 발생: {e}"
            logger.error(msg, request_id=request_id, error=str(e))
            raise RuntimeError(msg) from e

        # 응답 파싱
        try:
            response_data = response.json()
        except Exception as e:
            msg = f"응답 JSON 파싱 실패: {e}"
            logger.error(msg, request_id=request_id, response_text=response.text)
            raise RuntimeError(msg) from e

        # Document 리스트로 변환
        reranked_docs = self._parse_response(response_data, documents)

        # top_k 적용 (지정된 경우)
        if top_k is not None and top_k > 0:
            reranked_docs = reranked_docs[:top_k]

        logger.info(
            "문서 재정렬 완료",
            request_id=request_id,
            original_count=len(documents),
            reranked_count=len(reranked_docs),
            top_k=top_k,
        )

        return reranked_docs

    async def arerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        비동기 방식으로 Clova Studio API를 호출하여 문서를 재정렬합니다.

        매개변수:
            query: 사용자 질문
            documents: 재정렬할 문서 시퀀스
            top_k: 반환할 상위 문서 수

        반환값:
            관련도 내림차순으로 정렬된 Document 리스트

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패
        """
        # 입력 검증
        self._validate_inputs(query, documents)

        # 빈 문서 리스트는 즉시 반환
        if len(documents) == 0:
            return []

        # API 요청 페이로드 구성
        payload = {
            "query": query,
            "documents": _serialize_documents_for_api(documents),
            "maxTokens": self.max_tokens,
        }

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # 비동기 API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Reranker API 비동기 호출",
            request_id=request_id,
            query_length=len(query),
            document_count=len(documents),
            max_tokens=self.max_tokens,
        )

        try:
            response = await self._acall_reranker_api(payload, headers)
        except httpx.HTTPStatusError as e:
            msg = f"API 호출 실패 (HTTP {e.response.status_code})"
            logger.error(
                msg,
                request_id=request_id,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise RuntimeError(msg) from e
        except httpx.RequestError as e:
            msg = f"API 요청 중 오류 발생: {e}"
            logger.error(msg, request_id=request_id, error=str(e))
            raise RuntimeError(msg) from e

        # 응답 파싱
        try:
            response_data = response.json()
        except Exception as e:
            msg = f"응답 JSON 파싱 실패: {e}"
            logger.error(msg, request_id=request_id, response_text=response.text)
            raise RuntimeError(msg) from e

        # Document 리스트로 변환
        reranked_docs = self._parse_response(response_data, documents)

        # top_k 적용 (지정된 경우)
        if top_k is not None and top_k > 0:
            reranked_docs = reranked_docs[:top_k]

        logger.info(
            "비동기 문서 재정렬 완료",
            request_id=request_id,
            original_count=len(documents),
            reranked_count=len(reranked_docs),
            top_k=top_k,
        )

        return reranked_docs

    def close(self) -> None:
        """
        동기 HTTPX Client를 명시적으로 닫습니다.

        이 메서드는 멱등성(idempotent)을 보장하여 여러 번 호출해도 안전합니다.
        Context manager를 사용하지 않는 경우 명시적으로 호출해야 합니다.

        예시:
            >>> reranker = ClovaStudioReranker(...)
            >>> try:
            ...     results = reranker.rerank(query, docs)
            ... finally:
            ...     reranker.close()
        """
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing sync client: {e}")
            finally:
                self._client = None

    async def aclose(self) -> None:
        """
        비동기 HTTPX AsyncClient를 명시적으로 닫습니다.

        이 메서드는 멱등성(idempotent)을 보장하여 여러 번 호출해도 안전합니다.
        Async context manager를 사용하지 않는 경우 명시적으로 호출해야 합니다.

        예시:
            >>> reranker = ClovaStudioReranker(...)
            >>> try:
            ...     results = await reranker.arerank(query, docs)
            ... finally:
            ...     await reranker.aclose()
        """
        if hasattr(self, "_async_client") and self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async client: {e}")
            finally:
                self._async_client = None

    def __enter__(self) -> "ClovaStudioReranker":
        """
        동기 context manager 진입.

        예시:
            >>> with ClovaStudioReranker(...) as reranker:
            ...     results = reranker.rerank(query, docs)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        동기 context manager 종료.

        예외 발생 여부와 관계없이 리소스를 정리합니다.

        매개변수:
            exc_type: 예외 타입 (없으면 None)
            exc_val: 예외 값 (없으면 None)
            exc_tb: 예외 traceback (없으면 None)
        """
        self.close()

    async def __aenter__(self) -> "ClovaStudioReranker":
        """
        비동기 context manager 진입.

        예시:
            >>> async with ClovaStudioReranker(...) as reranker:
            ...     results = await reranker.arerank(query, docs)
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        비동기 context manager 종료.

        예외 발생 여부와 관계없이 리소스를 정리합니다.

        매개변수:
            exc_type: 예외 타입 (없으면 None)
            exc_val: 예외 값 (없으면 None)
            exc_tb: 예외 traceback (없으면 None)
        """
        await self.aclose()

    def __del__(self) -> None:
        """
        가비지 컬렉션 시 리소스 정리.

        참고:
            __del__은 GC 타이밍에 의존하므로 신뢰할 수 없습니다.
            명시적으로 close() 메서드를 호출하거나 context manager를 사용하세요.
            AsyncClient는 __del__에서 정리할 수 없으므로 반드시 aclose()를 호출해야 합니다.
        """
        try:
            self.close()
        except Exception:
            # __del__에서는 예외를 무시해야 함
            pass


# ============================================================================
# 모듈 공개 인터페이스
# ============================================================================

__all__ = [
    "BaseReranker",
    "ClovaStudioReranker",
]
