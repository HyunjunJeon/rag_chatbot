"""
Naver Clova Studio Summarization API 구현 모듈

이 모듈은 긴 텍스트를 짧고 간략하게 요약하는
Summarizer 인터페이스와 구현체를 제공합니다.

Clova Studio Summarization API 사양:
    - 엔드포인트: https://clovastudio.stream.ntruss.com/v1/api-tools/summarization/v2
    - 실제 API 문서: https://api.ncloud-docs.com/docs/clovastudio-summarization
    - 인증 헤더:
        * Content-Type: application/json (필수)
        * Authorization: Bearer <api-key> (필수)
    - 요청 페이로드:
        {
            "texts": list[str],  # 요약할 문장 목록 (~35,000자)
            "autoSentenceSplitter": bool,  # 문장 분리 허용 여부
            "segCount": int,  # 문단 수 (-1: 자동)
            "segMaxSize": int,  # 문단 최대 크기
            "segMinSize": int,  # 문단 최소 크기
            "includeAiFilters": bool  # AI Filter 적용 여부
        }
    - 응답 스키마:
        {
            "status": {
                "code": str,
                "message": str
            },
            "result": {
                "text": str,  # 요약 결과
                "inputTokens": int  # 입력 토큰 수
            }
        }
    - 특징:
        * 최대 35,000자까지 처리 가능 (한글 기준, 공백 포함)
        * 자동 문장 분리 및 문단 조절 기능
        * AI Filter를 통한 안전성 확보 옵션

사용 예:
    from naver_connect_chatbot.rag.summarization import ClovaStudioSummarizer
    from naver_connect_chatbot.config import settings
    
    summarizer = ClovaStudioSummarizer.from_settings(settings.summarization)
    result = summarizer.summarize(texts=["긴 문서 텍스트..."])
    print(f"요약: {result.text}")
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from naver_connect_chatbot.config import logger


# ============================================================================
# 결과 데이터 클래스
# ============================================================================


@dataclass(frozen=True)
class SummarizationResult:
    """
    요약 결과를 담는 불변 데이터 클래스입니다.
    
    속성:
        text: 요약된 텍스트
        input_tokens: 입력 토큰 수
        
    예시:
        >>> result = SummarizationResult(
        ...     text="클로바 스튜디오는 다양한 AI 기능을 제공합니다.",
        ...     input_tokens=187
        ... )
        >>> print(result.text)
        클로바 스튜디오는 다양한 AI 기능을 제공합니다.
    """
    text: str
    input_tokens: int


# ============================================================================
# Retry 조건 함수
# ============================================================================


def _should_retry_http_error(exception: BaseException) -> bool:
    """
    HTTP 오류가 재시도 가능한지 판단합니다.
    
    5xx 서버 오류만 재시도하고, 4xx 클라이언트 오류는 즉시 실패시킵니다.
    
    매개변수:
        exception: 발생한 예외
        
    반환값:
        재시도 가능 여부 (True: 재시도, False: 즉시 실패)
    """
    if isinstance(exception, httpx.HTTPStatusError):
        # 5xx 서버 오류만 재시도
        return 500 <= exception.response.status_code < 600
    # httpx.TimeoutException, httpx.NetworkError 등은 재시도
    return isinstance(exception, (httpx.TimeoutException, httpx.NetworkError))


# ============================================================================
# Summarizer 인터페이스
# ============================================================================


class BaseSummarizer(ABC):
    """
    요약을 위한 추상 베이스 클래스입니다.
    
    텍스트 리스트를 입력받아 요약된 결과를 SummarizationResult로 반환합니다.
    
    구현 클래스는 다음 메서드를 반드시 구현해야 합니다:
        - summarize: 동기 요약 메서드
        - asummarize: 비동기 요약 메서드 (선택적)
    """
    
    @abstractmethod
    def summarize(self, texts: list[str]) -> SummarizationResult:
        """
        텍스트 리스트를 입력받아 요약합니다.
        
        매개변수:
            texts: 요약할 문장 목록
            
        반환값:
            SummarizationResult 객체 (text, input_tokens 포함)
            
        예외:
            ValueError: texts가 비어있거나 너무 긴 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생
            
        예시:
            >>> summarizer = SomeSummarizer()
            >>> result = summarizer.summarize(["긴 문서 텍스트..."])
            >>> print(result.text)
        """
        raise NotImplementedError
    
    async def asummarize(self, texts: list[str]) -> SummarizationResult:
        """
        비동기 방식으로 텍스트를 요약합니다.
        
        기본 구현은 동기 메서드를 호출하므로, 진정한 비동기 처리가 필요한 경우
        하위 클래스에서 오버라이드해야 합니다.
        
        매개변수:
            texts: 요약할 문장 목록
            
        반환값:
            SummarizationResult 객체
            
        예외:
            ValueError: texts가 비어있거나 너무 긴 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생
        """
        # 기본 구현: 동기 메서드 호출
        return self.summarize(texts)


# ============================================================================
# Clova Studio Summarizer 구현체
# ============================================================================


class ClovaStudioSummarizer(BaseSummarizer):
    """
    Naver Clova Studio Summarization API를 활용한 요약 구현체입니다.
    
    HTTPX를 사용하여 REST API를 호출하고, 응답을 SummarizationResult로 변환합니다.
    
    속성:
        endpoint: Clova Studio Summarization API 엔드포인트 URL
        api_key: CLOVASTUDIO_API_KEY (Authorization Bearer 토큰)
        auto_sentence_splitter: 문장 분리 허용 여부
        seg_count: 문단 수
        seg_max_size: 문단 최대 크기
        seg_min_size: 문단 최소 크기
        include_ai_filters: AI Filter 적용 여부
        request_timeout: HTTP 요청 타임아웃 (초)
        client: HTTPX Client 인스턴스 (재사용)
        
    예시:
        >>> summarizer = ClovaStudioSummarizer(
        ...     endpoint="https://clovastudio.stream.ntruss.com/v1/api-tools/summarization/v2",
        ...     api_key="your-api-key",
        ...     auto_sentence_splitter=True,
        ... )
        >>> result = summarizer.summarize(["긴 문서 텍스트..."])
    """
    
    # 최대 입력 문자 수 (한글 기준, 공백 포함)
    MAX_INPUT_LENGTH = 35_000
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        auto_sentence_splitter: bool = True,
        seg_count: int = -1,
        seg_max_size: int = 1000,
        seg_min_size: int = 300,
        include_ai_filters: bool = False,
        request_timeout: float = 60.0,
    ) -> None:
        """
        ClovaStudioSummarizer를 초기화합니다.
        
        매개변수:
            endpoint: Clova Studio Summarization API 엔드포인트
            api_key: Clova Studio API 키 (CLOVASTUDIO_API_KEY)
            auto_sentence_splitter: 문장 분리 허용 여부 (기본값: True)
            seg_count: 문단 수 (기본값: -1, 자동)
            seg_max_size: 문단 최대 크기 (기본값: 1000)
            seg_min_size: 문단 최소 크기 (기본값: 300)
            include_ai_filters: AI Filter 적용 여부 (기본값: False)
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
        self.auto_sentence_splitter = auto_sentence_splitter
        self.seg_count = seg_count
        self.seg_max_size = seg_max_size
        self.seg_min_size = seg_min_size
        self.include_ai_filters = include_ai_filters
        self.request_timeout = request_timeout
        
        # HTTPX Client 초기화 (세션 재사용)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
    
    @classmethod
    def from_settings(cls, settings: Any) -> "ClovaStudioSummarizer":
        """
        Settings 객체로부터 ClovaStudioSummarizer를 생성합니다.
        
        매개변수:
            settings: ClovaStudioSummarizationSettings 또는 호환 객체
            
        반환값:
            초기화된 ClovaStudioSummarizer 인스턴스
            
        예시:
            >>> from naver_connect_chatbot.config import settings
            >>> summarizer = ClovaStudioSummarizer.from_settings(settings.summarization)
        """
        return cls(
            endpoint=settings.endpoint,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            auto_sentence_splitter=getattr(settings, "auto_sentence_splitter", True),
            seg_count=getattr(settings, "seg_count", -1),
            seg_max_size=getattr(settings, "seg_max_size", 1000),
            seg_min_size=getattr(settings, "seg_min_size", 300),
            include_ai_filters=getattr(settings, "include_ai_filters", False),
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
        
        Clova Studio API는 표준 Bearer 토큰 방식을 사용합니다:
        - Content-Type: application/json (필수)
        - Authorization: Bearer <api-key> (필수)
        
        반환값:
            HTTP 헤더 딕셔너리
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    def _validate_input(self, texts: list[str]) -> None:
        """
        입력 데이터의 유효성을 검증합니다.
        
        매개변수:
            texts: 입력 텍스트 리스트
            
        예외:
            ValueError: 유효하지 않은 입력인 경우
        """
        if not texts:
            msg = "texts는 비어있을 수 없습니다"
            raise ValueError(msg)
        
        # 전체 텍스트 길이 계산
        total_length = sum(len(text) for text in texts)
        
        if total_length > self.MAX_INPUT_LENGTH:
            msg = f"texts의 총 길이는 최대 {self.MAX_INPUT_LENGTH}자까지 처리 가능합니다 (입력: {total_length}자)"
            raise ValueError(msg)
    
    def _parse_response(self, response_data: dict[str, Any]) -> SummarizationResult:
        """
        Clova Studio Summarization API 응답을 파싱하여 SummarizationResult로 변환합니다.
        
        API 응답 구조:
        {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "text": str,
                "inputTokens": int
            }
        }
        
        매개변수:
            response_data: API 응답 JSON 데이터
            
        반환값:
            SummarizationResult 객체
            
        예외:
            ValueError: 응답 스키마가 예상과 다른 경우
        """
        try:
            result = response_data["result"]
            text = result["text"]
            input_tokens = result["inputTokens"]
        except KeyError as e:
            msg = f"API 응답 스키마가 올바르지 않습니다: {e}"
            raise ValueError(msg) from e
        
        # 토큰 사용량 로깅 (디버그 레벨)
        logger.debug(
            "Summarization API 토큰 사용량",
            input_tokens=input_tokens,
            summary_length=len(text),
        )
        
        return SummarizationResult(
            text=text,
            input_tokens=input_tokens,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    def _call_summarization_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """
        Summarization API를 호출합니다 (retry 로직 포함).
        
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
        retry=retry_if_exception(_should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    async def _acall_summarization_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """
        Summarization API를 비동기로 호출합니다 (retry 로직 포함).
        
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
    
    def summarize(self, texts: list[str]) -> SummarizationResult:
        """
        Clova Studio API를 호출하여 텍스트를 요약합니다.
        
        매개변수:
            texts: 요약할 문장 목록
            
        반환값:
            SummarizationResult 객체 (text, input_tokens)
            
        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패
            
        Time Complexity:
            O(n), n은 텍스트 총 길이 (API 응답 시간 제외)
        """
        # 입력 검증
        self._validate_input(texts)
        
        # API 요청 페이로드 구성
        payload = {
            "texts": texts,
            "autoSentenceSplitter": self.auto_sentence_splitter,
            "segCount": self.seg_count,
            "segMaxSize": self.seg_max_size,
            "segMinSize": self.seg_min_size,
            "includeAiFilters": self.include_ai_filters,
        }
        
        # 헤더 구성
        headers = self._build_headers()
        
        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())
        
        # 총 텍스트 길이 계산
        total_length = sum(len(text) for text in texts)
        
        # API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Summarization API 호출",
            request_id=request_id,
            text_count=len(texts),
            total_length=total_length,
        )
        
        try:
            response = self._call_summarization_api(payload, headers)
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
        
        # SummarizationResult로 변환
        result = self._parse_response(response_data)
        
        logger.info(
            "요약 완료",
            request_id=request_id,
            total_length=total_length,
            summary_length=len(result.text),
            input_tokens=result.input_tokens,
        )
        
        return result
    
    async def asummarize(self, texts: list[str]) -> SummarizationResult:
        """
        비동기 방식으로 Clova Studio API를 호출하여 텍스트를 요약합니다.
        
        매개변수:
            texts: 요약할 문장 목록
            
        반환값:
            SummarizationResult 객체
            
        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패
        """
        # 입력 검증
        self._validate_input(texts)
        
        # API 요청 페이로드 구성
        payload = {
            "texts": texts,
            "autoSentenceSplitter": self.auto_sentence_splitter,
            "segCount": self.seg_count,
            "segMaxSize": self.seg_max_size,
            "segMinSize": self.seg_min_size,
            "includeAiFilters": self.include_ai_filters,
        }
        
        # 헤더 구성
        headers = self._build_headers()
        
        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())
        
        # 총 텍스트 길이 계산
        total_length = sum(len(text) for text in texts)
        
        # 비동기 API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Summarization API 비동기 호출",
            request_id=request_id,
            text_count=len(texts),
            total_length=total_length,
        )
        
        try:
            response = await self._acall_summarization_api(payload, headers)
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
        
        # SummarizationResult로 변환
        result = self._parse_response(response_data)
        
        logger.info(
            "비동기 요약 완료",
            request_id=request_id,
            total_length=total_length,
            summary_length=len(result.text),
            input_tokens=result.input_tokens,
        )
        
        return result
    
    def close(self) -> None:
        """
        동기 HTTPX Client를 명시적으로 닫습니다.
        
        이 메서드는 멱등성(idempotent)을 보장하여 여러 번 호출해도 안전합니다.
        Context manager를 사용하지 않는 경우 명시적으로 호출해야 합니다.
        
        예시:
            >>> summarizer = ClovaStudioSummarizer(...)
            >>> try:
            ...     result = summarizer.summarize(texts)
            ... finally:
            ...     summarizer.close()
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
            >>> summarizer = ClovaStudioSummarizer(...)
            >>> try:
            ...     result = await summarizer.asummarize(texts)
            ... finally:
            ...     await summarizer.aclose()
        """
        if hasattr(self, "_async_client") and self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async client: {e}")
            finally:
                self._async_client = None
    
    def __enter__(self) -> "ClovaStudioSummarizer":
        """
        동기 context manager 진입.
        
        예시:
            >>> with ClovaStudioSummarizer(...) as summarizer:
            ...     result = summarizer.summarize(texts)
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
    
    async def __aenter__(self) -> "ClovaStudioSummarizer":
        """
        비동기 context manager 진입.
        
        예시:
            >>> async with ClovaStudioSummarizer(...) as summarizer:
            ...     result = await summarizer.asummarize(texts)
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
    "BaseSummarizer",
    "ClovaStudioSummarizer",
    "SummarizationResult",
]

