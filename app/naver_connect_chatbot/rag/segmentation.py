"""
Naver Clova Studio Segmentation API 구현 모듈

이 모듈은 문장 간 유사도를 파악하여 주제 단위로 글의 단락을 구분하는
Segmenter 인터페이스와 구현체를 제공합니다.

Clova Studio Segmentation API 사양:
    - 엔드포인트: https://clovastudio.stream.ntruss.com/v1/api-tools/segmentation
    - 실제 API 문서: https://api.ncloud-docs.com/docs/clovastudio-segmentation
    - 인증 헤더:
        * Content-Type: application/json (필수)
        * Authorization: Bearer <api-key> (필수)
    - 요청 페이로드:
        {
            "text": str,  # 문단 나누기를 수행할 문서 (~12만자)
            "alpha": float,  # threshold 값 (-1.5~1.5, -100: 자동)
            "segCnt": int,  # 원하는 문단 수 (-1: 자동)
            "postProcess": bool,  # 후처리 수행 여부
            "postProcessMaxSize": int,  # 후처리 최대 크기
            "postProcessMinSize": int  # 후처리 최소 크기
        }
    - 응답 스키마:
        {
            "status": {
                "code": str,
                "message": str
            },
            "result": {
                "topicSeg": list[list[str]],  # 주제별 문단 리스트
                "span": list[list[int]],  # 문단 인덱스
                "inputTokens": int  # 입력 토큰 수
            }
        }
    - 특징:
        * 최대 12만자까지 처리 가능 (한글 기준, 공백 포함)
        * 문장 간 유사도 기반으로 주제 단위 분할
        * 후처리를 통해 문단 길이 조절 가능

사용 예:
    from naver_connect_chatbot.rag.segmentation import ClovaStudioSegmenter
    from naver_connect_chatbot.config import settings

    segmenter = ClovaStudioSegmenter.from_settings(settings.segmentation)
    result = segmenter.segment(text="긴 문서 텍스트...")
    for idx, segment in enumerate(result.topic_segments):
        print(f"주제 {idx + 1}: {segment}")
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
class SegmentationResult:
    """
    문단 나누기 결과를 담는 불변 데이터 클래스입니다.

    속성:
        topic_segments: 주제별로 분리된 문단 리스트 (2차원 배열)
        span: 각 주제의 문단 인덱스 (2차원 배열)
        input_tokens: 입력 토큰 수

    예시:
        >>> result = SegmentationResult(
        ...     topic_segments=[
        ...         ["문장1", "문장2"],
        ...         ["문장3", "문장4", "문장5"]
        ...     ],
        ...     span=[[0, 1], [2, 3, 4]],
        ...     input_tokens=150
        ... )
        >>> len(result.topic_segments)
        2
    """

    topic_segments: list[list[str]]
    span: list[list[int]]
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
# Segmenter 인터페이스
# ============================================================================


class BaseSegmenter(ABC):
    """
    문단 나누기를 위한 추상 베이스 클래스입니다.

    텍스트를 입력받아 주제별로 문단을 분리하여 SegmentationResult를 반환합니다.

    구현 클래스는 다음 메서드를 반드시 구현해야 합니다:
        - segment: 동기 문단 나누기 메서드
        - asegment: 비동기 문단 나누기 메서드 (선택적)
    """

    @abstractmethod
    def segment(self, text: str) -> SegmentationResult:
        """
        텍스트를 입력받아 주제별로 문단을 분리합니다.

        매개변수:
            text: 문단 나누기를 수행할 텍스트

        반환값:
            SegmentationResult 객체 (topic_segments, span, input_tokens 포함)

        예외:
            ValueError: text가 빈 문자열이거나 너무 긴 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생

        예시:
            >>> segmenter = SomeSegmenter()
            >>> result = segmenter.segment("긴 문서 텍스트...")
            >>> len(result.topic_segments)
            5
        """
        raise NotImplementedError

    async def asegment(self, text: str) -> SegmentationResult:
        """
        비동기 방식으로 문단을 나눕니다.

        기본 구현은 동기 메서드를 호출하므로, 진정한 비동기 처리가 필요한 경우
        하위 클래스에서 오버라이드해야 합니다.

        매개변수:
            text: 문단 나누기를 수행할 텍스트

        반환값:
            SegmentationResult 객체

        예외:
            ValueError: text가 빈 문자열이거나 너무 긴 경우
            RuntimeError: API 호출 또는 응답 처리 중 오류 발생
        """
        # 기본 구현: 동기 메서드 호출
        return self.segment(text)


# ============================================================================
# Clova Studio Segmenter 구현체
# ============================================================================


class ClovaStudioSegmenter(BaseSegmenter):
    """
    Naver Clova Studio Segmentation API를 활용한 문단 나누기 구현체입니다.

    HTTPX를 사용하여 REST API를 호출하고, 응답을 SegmentationResult로 변환합니다.

    속성:
        endpoint: Clova Studio Segmentation API 엔드포인트 URL
        api_key: CLOVASTUDIO_API_KEY (Authorization Bearer 토큰)
        alpha: 문단 나누기 threshold 값
        seg_count: 원하는 문단 수
        post_process: 후처리 수행 여부
        post_process_max_size: 후처리 최대 크기
        post_process_min_size: 후처리 최소 크기
        request_timeout: HTTP 요청 타임아웃 (초)
        client: HTTPX Client 인스턴스 (재사용)

    예시:
        >>> segmenter = ClovaStudioSegmenter(
        ...     endpoint="https://clovastudio.stream.ntruss.com/v1/api-tools/segmentation",
        ...     api_key="your-api-key",
        ...     alpha=-100.0,
        ...     seg_count=-1,
        ... )
        >>> result = segmenter.segment("긴 문서 텍스트...")
    """

    # 최대 입력 문자 수 (한글 기준, 공백 포함)
    MAX_INPUT_LENGTH = 120_000

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        alpha: float = -100.0,
        seg_count: int = -1,
        post_process: bool = False,
        post_process_max_size: int = 1000,
        post_process_min_size: int = 300,
        request_timeout: float = 60.0,
    ) -> None:
        """
        ClovaStudioSegmenter를 초기화합니다.

        매개변수:
            endpoint: Clova Studio Segmentation API 엔드포인트
            api_key: Clova Studio API 키 (CLOVASTUDIO_API_KEY)
            alpha: threshold 값 (기본값: -100, 범위: -1.5~1.5 또는 -100)
            seg_count: 원하는 문단 수 (기본값: -1, 자동)
            post_process: 후처리 수행 여부 (기본값: False)
            post_process_max_size: 후처리 최대 크기 (기본값: 1000)
            post_process_min_size: 후처리 최소 크기 (기본값: 300)
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
        self.alpha = alpha
        self.seg_count = seg_count
        self.post_process = post_process
        self.post_process_max_size = post_process_max_size
        self.post_process_min_size = post_process_min_size
        self.request_timeout = request_timeout

        # HTTPX Client 초기화 (세션 재사용)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "ClovaStudioSegmenter":
        """
        Settings 객체로부터 ClovaStudioSegmenter를 생성합니다.

        매개변수:
            settings: ClovaStudioSegmentationSettings 또는 호환 객체

        반환값:
            초기화된 ClovaStudioSegmenter 인스턴스

        예시:
            >>> from naver_connect_chatbot.config import settings
            >>> segmenter = ClovaStudioSegmenter.from_settings(settings.segmentation)
        """
        return cls(
            endpoint=settings.endpoint,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            alpha=getattr(settings, "alpha", -100.0),
            seg_count=getattr(settings, "seg_count", -1),
            post_process=getattr(settings, "post_process", False),
            post_process_max_size=getattr(settings, "post_process_max_size", 1000),
            post_process_min_size=getattr(settings, "post_process_min_size", 300),
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

    def _validate_input(self, text: str) -> None:
        """
        입력 데이터의 유효성을 검증합니다.

        매개변수:
            text: 입력 텍스트

        예외:
            ValueError: 유효하지 않은 입력인 경우
        """
        if not text or not text.strip():
            msg = "text는 빈 문자열일 수 없습니다"
            raise ValueError(msg)

        if len(text) > self.MAX_INPUT_LENGTH:
            msg = f"text는 최대 {self.MAX_INPUT_LENGTH}자까지 처리 가능합니다 (입력: {len(text)}자)"
            raise ValueError(msg)

    def _parse_response(self, response_data: dict[str, Any]) -> SegmentationResult:
        """
        Clova Studio Segmentation API 응답을 파싱하여 SegmentationResult로 변환합니다.

        API 응답 구조:
        {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "topicSeg": list[list[str]],
                "span": list[list[int]],
                "inputTokens": int
            }
        }

        매개변수:
            response_data: API 응답 JSON 데이터

        반환값:
            SegmentationResult 객체

        예외:
            ValueError: 응답 스키마가 예상과 다른 경우
        """
        try:
            result = response_data["result"]
            topic_seg = result["topicSeg"]
            span = result["span"]
            input_tokens = result["inputTokens"]
        except KeyError as e:
            msg = f"API 응답 스키마가 올바르지 않습니다: {e}"
            raise ValueError(msg) from e

        # 토큰 사용량 로깅 (디버그 레벨)
        logger.debug(
            "Segmentation API 토큰 사용량",
            input_tokens=input_tokens,
            segment_count=len(topic_seg),
        )

        return SegmentationResult(
            topic_segments=topic_seg,
            span=span,
            input_tokens=input_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    def _call_segmentation_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """
        Segmentation API를 호출합니다 (retry 로직 포함).

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
    async def _acall_segmentation_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """
        Segmentation API를 비동기로 호출합니다 (retry 로직 포함).

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

    def segment(self, text: str) -> SegmentationResult:
        """
        Clova Studio API를 호출하여 텍스트를 주제별 문단으로 나눕니다.

        매개변수:
            text: 문단 나누기를 수행할 텍스트

        반환값:
            SegmentationResult 객체 (topic_segments, span, input_tokens)

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패

        Time Complexity:
            O(n), n은 텍스트 길이 (API 응답 시간 제외)
        """
        # 입력 검증
        self._validate_input(text)

        # API 요청 페이로드 구성
        payload = {
            "text": text,
            "alpha": self.alpha,
            "segCnt": self.seg_count,
            "postProcess": self.post_process,
            "postProcessMaxSize": self.post_process_max_size,
            "postProcessMinSize": self.post_process_min_size,
        }

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Segmentation API 호출",
            request_id=request_id,
            text_length=len(text),
            alpha=self.alpha,
            seg_count=self.seg_count,
        )

        try:
            response = self._call_segmentation_api(payload, headers)
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

        # SegmentationResult로 변환
        result = self._parse_response(response_data)

        logger.info(
            "문단 나누기 완료",
            request_id=request_id,
            text_length=len(text),
            segment_count=len(result.topic_segments),
            input_tokens=result.input_tokens,
        )

        return result

    async def asegment(self, text: str) -> SegmentationResult:
        """
        비동기 방식으로 Clova Studio API를 호출하여 텍스트를 주제별 문단으로 나눕니다.

        매개변수:
            text: 문단 나누기를 수행할 텍스트

        반환값:
            SegmentationResult 객체

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패
        """
        # 입력 검증
        self._validate_input(text)

        # API 요청 페이로드 구성
        payload = {
            "text": text,
            "alpha": self.alpha,
            "segCnt": self.seg_count,
            "postProcess": self.post_process,
            "postProcessMaxSize": self.post_process_max_size,
            "postProcessMinSize": self.post_process_min_size,
        }

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # 비동기 API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio Segmentation API 비동기 호출",
            request_id=request_id,
            text_length=len(text),
            alpha=self.alpha,
            seg_count=self.seg_count,
        )

        try:
            response = await self._acall_segmentation_api(payload, headers)
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

        # SegmentationResult로 변환
        result = self._parse_response(response_data)

        logger.info(
            "비동기 문단 나누기 완료",
            request_id=request_id,
            text_length=len(text),
            segment_count=len(result.topic_segments),
            input_tokens=result.input_tokens,
        )

        return result

    def close(self) -> None:
        """
        동기 HTTPX Client를 명시적으로 닫습니다.

        이 메서드는 멱등성(idempotent)을 보장하여 여러 번 호출해도 안전합니다.
        Context manager를 사용하지 않는 경우 명시적으로 호출해야 합니다.

        예시:
            >>> segmenter = ClovaStudioSegmenter(...)
            >>> try:
            ...     result = segmenter.segment(text)
            ... finally:
            ...     segmenter.close()
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
            >>> segmenter = ClovaStudioSegmenter(...)
            >>> try:
            ...     result = await segmenter.asegment(text)
            ... finally:
            ...     await segmenter.aclose()
        """
        if hasattr(self, "_async_client") and self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async client: {e}")
            finally:
                self._async_client = None

    def __enter__(self) -> "ClovaStudioSegmenter":
        """
        동기 context manager 진입.

        예시:
            >>> with ClovaStudioSegmenter(...) as segmenter:
            ...     result = segmenter.segment(text)
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

    async def __aenter__(self) -> "ClovaStudioSegmenter":
        """
        비동기 context manager 진입.

        예시:
            >>> async with ClovaStudioSegmenter(...) as segmenter:
            ...     result = await segmenter.asegment(text)
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
    "BaseSegmenter",
    "ClovaStudioSegmenter",
    "SegmentationResult",
]
