"""
Naver Clova Studio RAG Reasoning 구현 모듈

이 모듈은 Function calling 기반 RAG 추론을 위한 Clova Studio RAG Reasoning API를 제공합니다.

Clova Studio RAG Reasoning API 사양:
    - 엔드포인트: https://clovastudio.stream.ntruss.com/v1/api-tools/rag-reasoning
    - 실제 API 문서: https://api.ncloud-docs.com/docs/clovastudio-ragreasoning
    - 인증 헤더:
        * Content-Type: application/json (필수)
        * Authorization: Bearer <api-key> (필수)
    - 요청 페이로드:
        {
            "messages": [
                {"role": "user", "content": "질문"},
                {"role": "assistant", "content": "", "toolCalls": [...]},
                {"role": "tool", "content": "검색 결과", "toolCallId": "..."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "문서 검색 도구",
                        "parameters": {"type": "object", "properties": {...}}
                    }
                }
            ],
            "toolChoice": "auto",  # or "none", or {"type": "function", "function": {"name": "..."}}
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 1024,
            "temperature": 0.5,
            "repetitionPenalty": 1.1,
            "stop": [],
            "seed": 0,
            "includeAiFilters": true
        }
    - 응답 스키마:
        {
            "status": {
                "code": "20000",
                "message": "OK"
            },
            "result": {
                "message": {
                    "role": "assistant",
                    "content": "답변 텍스트 (인용 표기 포함)",
                    "thinkingContent": "의사결정 흐름",
                    "toolCalls": [
                        {
                            "id": "call_xxx",
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "arguments": {"query": "..."}
                            }
                        }
                    ]
                },
                "usage": {
                    "promptTokens": 135,
                    "completionTokens": 84,
                    "totalTokens": 219
                }
            }
        }
    - 특징:
        * Function calling 기반으로 검색 과정을 계획
        * Multi-turn 대화 지원 (toolCalls → tool → 최종 답변)
        * 인용 출처 표기 (<doc-123>...</doc-123>)
        * 최대 입력 토큰: 128,000
        * 최대 출력 토큰: 4,096

사용 예:
    from naver_connect_chatbot.rag.rag_reasoning import ClovaStudioRAGReasoning
    from naver_connect_chatbot.config import settings

    rag_reasoning = ClovaStudioRAGReasoning.from_settings(settings.rag_reasoning)

    # 단순 호출
    messages = [{"role": "user", "content": "VPC 삭제 방법은?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "문서 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "검색어"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    response = rag_reasoning.invoke(messages=messages, tools=tools)
    print(response["message"]["content"])
"""

import logging
import uuid
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
from naver_connect_chatbot.rag.utils import should_retry_http_error


# ============================================================================
# 유틸리티 함수
# ============================================================================


def convert_langchain_tool_to_rag_reasoning(tool: Any) -> dict[str, Any]:
    """
    LangChain Tool을 RAG Reasoning API 형식으로 변환합니다.

    LangChain의 BaseTool 또는 @tool 데코레이터로 정의한 함수를
    Clova Studio RAG Reasoning API에서 사용할 수 있는 형식으로 변환합니다.

    LangChain Tool 형식:
        - BaseTool 클래스 상속
        - @tool 데코레이터 사용
        - name, description, args_schema 속성

    RAG Reasoning Tool 형식:
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

    매개변수:
        tool: LangChain BaseTool 객체 또는 호환 객체

    반환값:
        RAG Reasoning API 형식의 tool 딕셔너리

    예외:
        ImportError: langchain_core를 import할 수 없는 경우
        ValueError: tool이 유효하지 않은 경우

    예시:
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def search_documents(query: str) -> str:
        ...     '''문서를 검색합니다.'''
        ...     return f"Search results for: {query}"
        >>>
        >>> rag_tool = convert_langchain_tool_to_rag_reasoning(search_documents)
        >>> print(rag_tool)
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "문서를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    """
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
    except ImportError as e:
        msg = (
            "langchain_core를 import할 수 없습니다. "
            "langchain 또는 langchain-core를 설치해주세요: "
            "uv add langchain-core"
        )
        raise ImportError(msg) from e

    # LangChain의 convert_to_openai_tool() 사용
    # OpenAI tool 형식과 RAG Reasoning 형식이 동일함
    try:
        openai_tool = convert_to_openai_tool(tool)
        return openai_tool
    except Exception as e:
        msg = f"LangChain tool을 RAG Reasoning 형식으로 변환하는 중 오류 발생: {e}"
        raise ValueError(msg) from e


def convert_langchain_tools_to_rag_reasoning(tools: list[Any]) -> list[dict[str, Any]]:
    """
    여러 LangChain Tool을 RAG Reasoning API 형식으로 일괄 변환합니다.

    매개변수:
        tools: LangChain BaseTool 객체 리스트

    반환값:
        RAG Reasoning API 형식의 tool 딕셔너리 리스트

    예외:
        ImportError: langchain_core를 import할 수 없는 경우
        ValueError: tools가 빈 리스트이거나 유효하지 않은 경우

    예시:
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def search_docs(query: str) -> str:
        ...     '''문서 검색'''
        ...     return f"Results: {query}"
        >>>
        >>> @tool
        >>> def get_weather(city: str) -> str:
        ...     '''날씨 조회'''
        ...     return f"Weather in {city}"
        >>>
        >>> rag_tools = convert_langchain_tools_to_rag_reasoning([search_docs, get_weather])
        >>> len(rag_tools)
        2
    """
    if not tools:
        msg = "tools는 비어있을 수 없습니다"
        raise ValueError(msg)

    converted_tools = []
    for idx, tool in enumerate(tools):
        try:
            converted = convert_langchain_tool_to_rag_reasoning(tool)
            converted_tools.append(converted)
        except Exception as e:
            msg = f"tools[{idx}] 변환 실패: {e}"
            logger.error(msg, tool_index=idx, error=str(e))
            raise ValueError(msg) from e

    return converted_tools


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    """
    Messages 배열의 유효성을 검증합니다.

    매개변수:
        messages: 대화 메시지 리스트

    예외:
        ValueError: 유효하지 않은 메시지 구조
    """
    if not messages:
        msg = "messages는 비어있을 수 없습니다"
        raise ValueError(msg)

    valid_roles = {"system", "user", "assistant", "tool"}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            msg_err = f"messages[{idx}]는 딕셔너리여야 합니다"
            raise ValueError(msg_err)

        role = msg.get("role")
        if role not in valid_roles:
            msg_err = f"messages[{idx}].role은 {valid_roles} 중 하나여야 합니다. 현재값: {role}"
            raise ValueError(msg_err)

        if "content" not in msg:
            msg_err = f"messages[{idx}]에 content 필드가 없습니다"
            raise ValueError(msg_err)

        # tool 역할은 toolCallId 필수
        if role == "tool" and "toolCallId" not in msg:
            msg_err = f"messages[{idx}]는 tool 역할이지만 toolCallId가 없습니다"
            raise ValueError(msg_err)


def _validate_tools(tools: list[dict[str, Any]]) -> None:
    """
    Tools 배열의 유효성을 검증합니다.

    매개변수:
        tools: 도구 정의 리스트

    예외:
        ValueError: 유효하지 않은 도구 구조
    """
    if not tools:
        msg = "tools는 비어있을 수 없습니다"
        raise ValueError(msg)

    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            msg_err = f"tools[{idx}]는 딕셔너리여야 합니다"
            raise ValueError(msg_err)

        if tool.get("type") != "function":
            msg_err = f"tools[{idx}].type은 'function'이어야 합니다. 현재값: {tool.get('type')}"
            raise ValueError(msg_err)

        function = tool.get("function")
        if not isinstance(function, dict):
            msg_err = f"tools[{idx}].function은 딕셔너리여야 합니다"
            raise ValueError(msg_err)

        if "name" not in function:
            msg_err = f"tools[{idx}].function에 name 필드가 없습니다"
            raise ValueError(msg_err)

        if "description" not in function:
            msg_err = f"tools[{idx}].function에 description 필드가 없습니다"
            raise ValueError(msg_err)

        if "parameters" not in function:
            msg_err = f"tools[{idx}].function에 parameters 필드가 없습니다"
            raise ValueError(msg_err)


# ============================================================================
# Clova Studio RAG Reasoning 구현체
# ============================================================================


class ClovaStudioRAGReasoning:
    """
    Naver Clova Studio RAG Reasoning API를 활용한 Function calling 기반 RAG 추론 구현체입니다.

    HTTPX를 사용하여 REST API를 호출하고, 응답을 파싱하여 반환합니다.

    속성:
        endpoint: Clova Studio RAG Reasoning API 엔드포인트 URL
        api_key: CLOVASTUDIO_API_KEY (Authorization Bearer 토큰)
        top_p: 생성 토큰 후보군 누적 확률 샘플링
        top_k: 생성 토큰 후보군 상위 K개 샘플링
        max_tokens: 최대 생성 토큰 수
        temperature: 생성 토큰 다양성 정도
        repetition_penalty: 같은 토큰 생성 패널티
        stop: 토큰 생성 중단 문자 리스트
        seed: 결과 일관성 수준 조정
        include_ai_filters: AI 필터 표시 여부
        request_timeout: HTTP 요청 타임아웃 (초)
        client: HTTPX Client 인스턴스 (재사용)

    예시:
        >>> rag_reasoning = ClovaStudioRAGReasoning(
        ...     endpoint="https://clovastudio.stream.ntruss.com/v1/api-tools/rag-reasoning",
        ...     api_key="your-api-key",
        ...     max_tokens=1024,
        ... )
        >>> messages = [{"role": "user", "content": "VPC 삭제 방법은?"}]
        >>> tools = [...]
        >>> response = rag_reasoning.invoke(messages=messages, tools=tools)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        top_p: float = 0.8,
        top_k: int = 0,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        repetition_penalty: float = 1.1,
        stop: list[str] | None = None,
        seed: int = 0,
        include_ai_filters: bool = True,
        request_timeout: float = 60.0,
    ) -> None:
        """
        ClovaStudioRAGReasoning을 초기화합니다.

        매개변수:
            endpoint: Clova Studio RAG Reasoning API 엔드포인트
            api_key: Clova Studio API 키 (CLOVASTUDIO_API_KEY)
            top_p: 생성 토큰 후보군 누적 확률 샘플링 (0 < topP ≤ 1, 기본값: 0.8)
            top_k: 생성 토큰 후보군 상위 K개 샘플링 (0 ≤ topK ≤ 128, 기본값: 0)
            max_tokens: 최대 생성 토큰 수 (1 ≤ maxTokens ≤ 4096, 기본값: 1024)
            temperature: 생성 토큰 다양성 정도 (0 < temperature ≤ 1, 기본값: 0.5)
            repetition_penalty: 같은 토큰 생성 패널티 (0 < penalty ≤ 2, 기본값: 1.1)
            stop: 토큰 생성 중단 문자 리스트 (기본값: [])
            seed: 결과 일관성 수준 (0: 랜덤, 1~4294967295: 고정, 기본값: 0)
            include_ai_filters: AI 필터 표시 여부 (기본값: True)
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
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.stop = stop if stop is not None else []
        self.seed = seed
        self.include_ai_filters = include_ai_filters
        self.request_timeout = request_timeout

        # HTTPX Client 초기화 (세션 재사용)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "ClovaStudioRAGReasoning":
        """
        Settings 객체로부터 ClovaStudioRAGReasoning을 생성합니다.

        매개변수:
            settings: ClovaStudioRAGReasoningSettings 또는 호환 객체

        반환값:
            초기화된 ClovaStudioRAGReasoning 인스턴스

        예시:
            >>> from naver_connect_chatbot.config import settings
            >>> rag_reasoning = ClovaStudioRAGReasoning.from_settings(settings.rag_reasoning)
        """
        return cls(
            endpoint=settings.endpoint,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            top_p=getattr(settings, "top_p", 0.8),
            top_k=getattr(settings, "top_k", 0),
            max_tokens=getattr(settings, "max_tokens", 1024),
            temperature=getattr(settings, "temperature", 0.5),
            repetition_penalty=getattr(settings, "repetition_penalty", 1.1),
            stop=getattr(settings, "stop", []),
            seed=getattr(settings, "seed", 0),
            include_ai_filters=getattr(settings, "include_ai_filters", True),
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

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        API 요청 페이로드를 구성합니다.

        매개변수:
            messages: 대화 메시지 리스트
            tools: 도구 정의 리스트
            tool_choice: 도구 호출 방식 ("auto", "none", 또는 특정 함수 지정)

        반환값:
            API 요청 페이로드 딕셔너리
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "topP": self.top_p,
            "topK": self.top_k,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "repetitionPenalty": self.repetition_penalty,
            "stop": self.stop,
            "seed": self.seed,
            "includeAiFilters": self.include_ai_filters,
        }

        if tool_choice is not None:
            payload["toolChoice"] = tool_choice

        return payload

    def _parse_response(
        self,
        response_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Clova Studio RAG Reasoning API 응답을 파싱합니다.

        API 응답 구조:
        {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "message": {
                    "role": "assistant",
                    "content": "답변 텍스트",
                    "thinkingContent": "의사결정 흐름",
                    "toolCalls": [...]
                },
                "usage": {
                    "promptTokens": 135,
                    "completionTokens": 84,
                    "totalTokens": 219
                }
            }
        }

        매개변수:
            response_data: API 응답 JSON 데이터

        반환값:
            파싱된 결과 딕셔너리 (message, usage 포함)

        예외:
            ValueError: 응답 스키마가 예상과 다른 경우
        """
        try:
            status = response_data.get("status", {})
            status_code = status.get("code")

            if status_code != "20000":
                msg = f"API 호출 실패: {status.get('message', 'Unknown error')}"
                raise ValueError(msg)

            result = response_data["result"]
            message = result["message"]
            usage = result.get("usage", {})
        except KeyError as e:
            msg = f"API 응답 스키마가 올바르지 않습니다: {e}"
            raise ValueError(msg) from e

        # 사용량 정보 로깅 (디버그 레벨)
        if usage:
            logger.debug(
                "RAG Reasoning API 토큰 사용량",
                prompt_tokens=usage.get("promptTokens", 0),
                completion_tokens=usage.get("completionTokens", 0),
                total_tokens=usage.get("totalTokens", 0),
            )

        return {
            "message": message,
            "usage": usage,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry_http_error),
        before_sleep=before_sleep_log(logger.bind(), logging.WARNING),
        reraise=True,
    )
    def _call_api(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        """
        RAG Reasoning API를 호출합니다 (retry 로직 포함).

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
    async def _acall_api(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        """
        RAG Reasoning API를 비동기로 호출합니다 (retry 로직 포함).

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

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        """
        Clova Studio RAG Reasoning API를 호출합니다.

        매개변수:
            messages: 대화 메시지 리스트
                [
                    {"role": "user", "content": "질문"},
                    {"role": "assistant", "content": "", "toolCalls": [...]},
                    {"role": "tool", "content": "검색 결과", "toolCallId": "..."}
                ]
            tools: 도구 정의 리스트
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_documents",
                            "description": "문서 검색",
                            "parameters": {...}
                        }
                    }
                ]
            tool_choice: 도구 호출 방식
                - "auto": 모델이 자동으로 함수 호출 (기본값)
                - "none": 함수 호출 없이 일반 답변 생성
                - {"type": "function", "function": {"name": "함수명"}}: 특정 함수 강제 호출

        반환값:
            {
                "message": {
                    "role": "assistant",
                    "content": "답변 텍스트",
                    "thinkingContent": "의사결정 흐름",
                    "toolCalls": [...]
                },
                "usage": {
                    "promptTokens": 135,
                    "completionTokens": 84,
                    "totalTokens": 219
                }
            }

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패

        Time Complexity:
            O(n), n은 API 응답 시간
        """
        # 입력 검증
        _validate_messages(messages)
        _validate_tools(tools)

        # API 요청 페이로드 구성
        payload = self._build_payload(messages, tools, tool_choice)

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio RAG Reasoning API 호출",
            request_id=request_id,
            message_count=len(messages),
            tool_count=len(tools),
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
        )

        try:
            response = self._call_api(payload, headers)
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

        # 결과 파싱
        result = self._parse_response(response_data)

        logger.info(
            "RAG Reasoning API 호출 완료",
            request_id=request_id,
            has_tool_calls="toolCalls" in result["message"],
            content_length=len(result["message"].get("content", "")),
        )

        return result

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        """
        비동기 방식으로 Clova Studio RAG Reasoning API를 호출합니다.

        매개변수:
            messages: 대화 메시지 리스트
            tools: 도구 정의 리스트
            tool_choice: 도구 호출 방식 (기본값: "auto")

        반환값:
            파싱된 결과 딕셔너리 (message, usage 포함)

        예외:
            ValueError: 입력이 유효하지 않거나 응답 파싱 실패
            RuntimeError: API 호출 실패
        """
        # 입력 검증
        _validate_messages(messages)
        _validate_tools(tools)

        # API 요청 페이로드 구성
        payload = self._build_payload(messages, tools, tool_choice)

        # 헤더 구성
        headers = self._build_headers()

        # 요청 ID 생성 (로깅용)
        request_id = str(uuid.uuid4())

        # 비동기 API 호출 (retry 로직 포함)
        logger.debug(
            "Clova Studio RAG Reasoning API 비동기 호출",
            request_id=request_id,
            message_count=len(messages),
            tool_count=len(tools),
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
        )

        try:
            response = await self._acall_api(payload, headers)
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

        # 결과 파싱
        result = self._parse_response(response_data)

        logger.info(
            "비동기 RAG Reasoning API 호출 완료",
            request_id=request_id,
            has_tool_calls="toolCalls" in result["message"],
            content_length=len(result["message"].get("content", "")),
        )

        return result

    def close(self) -> None:
        """
        동기 HTTPX Client를 명시적으로 닫습니다.

        이 메서드는 멱등성(idempotent)을 보장하여 여러 번 호출해도 안전합니다.
        Context manager를 사용하지 않는 경우 명시적으로 호출해야 합니다.

        예시:
            >>> rag_reasoning = ClovaStudioRAGReasoning(...)
            >>> try:
            ...     result = rag_reasoning.invoke(messages, tools)
            ... finally:
            ...     rag_reasoning.close()
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
            >>> rag_reasoning = ClovaStudioRAGReasoning(...)
            >>> try:
            ...     result = await rag_reasoning.ainvoke(messages, tools)
            ... finally:
            ...     await rag_reasoning.aclose()
        """
        if hasattr(self, "_async_client") and self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async client: {e}")
            finally:
                self._async_client = None

    def __enter__(self) -> "ClovaStudioRAGReasoning":
        """
        동기 context manager 진입.

        예시:
            >>> with ClovaStudioRAGReasoning(...) as rag_reasoning:
            ...     result = rag_reasoning.invoke(messages, tools)
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

    async def __aenter__(self) -> "ClovaStudioRAGReasoning":
        """
        비동기 context manager 진입.

        예시:
            >>> async with ClovaStudioRAGReasoning(...) as rag_reasoning:
            ...     result = await rag_reasoning.ainvoke(messages, tools)
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
    "ClovaStudioRAGReasoning",
    "convert_langchain_tool_to_rag_reasoning",
    "convert_langchain_tools_to_rag_reasoning",
]
