"""
LangChain 에이전트 응답 파싱 유틸리티.

이 모듈은 다양한 형식의 LangChain 에이전트 응답을 Pydantic 모델로 파싱하는
통합 함수를 제공합니다. 200+ 라인의 중복 코드를 단일 소스로 통합합니다.
"""

import json
from typing import Any, Type, TypeVar, Optional, List
from pydantic import BaseModel, ValidationError

from langchain_core.messages import BaseMessage, ToolMessage, filter_messages

from naver_connect_chatbot.config import logger

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def parse_agent_response(
    response: Any,
    model_type: Type[_ModelT],
    *,
    fallback: Optional[_ModelT] = None,
) -> _ModelT:
    """
    LangChain 에이전트 응답을 지정된 Pydantic 모델로 파싱합니다.

    지원하는 응답 형식 (우선순위 순):
    1. BaseModel 인스턴스 (이미 올바른 타입 또는 변환 가능)
    2. LangChain 메시지 (AIMessage, ToolMessage - content 추출)
    3. dict with "messages" key (AgentState - ToolMessage 추출)
    4. dict with "output" key (Legacy Agent)
    5. 일반 dict (모델 데이터)
    6. JSON 문자열

    매개변수:
        response: 에이전트 응답 (다양한 형식 지원)
        model_type: 반환할 Pydantic 모델 타입
        fallback: 파싱 실패 시 반환할 기본값 (None이면 예외 발생)

    반환값:
        파싱된 Pydantic 모델 인스턴스

    예외:
        ValueError: 파싱 실패 시 (fallback이 None인 경우)

    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from pydantic import BaseModel
        >>>
        >>> class Result(BaseModel):
        ...     status: str
        ...     confidence: float
        >>>
        >>> agent = create_agent(...)
        >>> response = agent.invoke({"messages": [...]})
        >>> result = parse_agent_response(response, Result)
        >>> print(result.status)
    """
    try:
        # 1. 이미 올바른 타입인 경우
        if isinstance(response, model_type):
            logger.debug(f"Response is already {model_type.__name__}")
            return response

        # 2. LangChain 메시지 객체 (content 추출) - BaseModel 체크보다 먼저!
        if hasattr(response, "content") and isinstance(response, BaseMessage):
            logger.debug("Extracting from message content")
            result = _parse_value(response.content, model_type)
            if result is not None:
                return result

        # 3. BaseModel이지만 다른 타입인 경우 (변환 시도)
        if isinstance(response, BaseModel):
            logger.debug(f"Converting {type(response).__name__} to {model_type.__name__}")
            return model_type.model_validate(response.model_dump())

        # 4. dict 형태인 경우
        if isinstance(response, dict):
            # 3a. AgentState (messages 키 포함)
            if "messages" in response:
                logger.debug("Extracting from AgentState messages")
                result = _extract_from_messages(response["messages"], model_type)
                if result is not None:
                    return result

            # 3b. Legacy Agent (output 키 포함)
            if "output" in response:
                logger.debug("Extracting from legacy agent output")
                result = _parse_value(response["output"], model_type)
                if result is not None:
                    return result

            # 4c. dict 자체가 모델 데이터인 경우
            try:
                logger.debug(f"Parsing dict as {model_type.__name__}")
                return model_type.model_validate(response)
            except ValidationError:
                pass  # 다음 방법 시도

        # 5. 문자열 (JSON 파싱 시도)
        if isinstance(response, str):
            logger.debug("Parsing string as JSON")
            result = _parse_json_string(response, model_type)
            if result is not None:
                return result

        # 파싱 실패
        error_msg = (
            f"Unable to parse response into {model_type.__name__}. "
            f"Response type: {type(response).__name__}"
        )
        logger.warning(error_msg)

        if fallback is not None:
            logger.info(f"Using fallback value for {model_type.__name__}")
            return fallback

        raise ValueError(error_msg)

    except Exception as e:
        if fallback is not None:
            logger.warning(
                f"Error parsing response: {e}. Using fallback value",
                exc_info=True
            )
            return fallback
        raise


def _extract_from_messages(
    messages: List[BaseMessage],
    model_type: Type[_ModelT]
) -> Optional[_ModelT]:
    """
    메시지 리스트에서 ToolMessage를 찾아 모델로 파싱합니다.

    매개변수:
        messages: LangChain 메시지 리스트
        model_type: 파싱할 Pydantic 모델 타입

    반환값:
        파싱된 모델 또는 None (ToolMessage가 없거나 파싱 실패 시)
    """
    try:
        # ToolMessage만 필터링
        tool_messages = filter_messages(messages, include_types=[ToolMessage])

        if not tool_messages:
            logger.debug("No ToolMessage found in messages")
            return None

        # 가장 최근의 ToolMessage 사용
        last_tool_msg = tool_messages[-1]
        tool_content = last_tool_msg.content

        logger.debug(f"Found ToolMessage with content type: {type(tool_content).__name__}")
        return _parse_value(tool_content, model_type)

    except Exception as e:
        logger.warning(f"Error extracting from messages: {e}")
        return None


def _parse_value(value: Any, model_type: Type[_ModelT]) -> Optional[_ModelT]:
    """
    단일 값을 Pydantic 모델로 파싱합니다.

    지원하는 값 타입: BaseModel, dict, JSON 문자열
    """
    # BaseModel 인스턴스
    if isinstance(value, model_type):
        return value

    if isinstance(value, BaseModel):
        try:
            return model_type.model_validate(value.model_dump())
        except ValidationError as e:
            logger.debug(f"Failed to convert BaseModel: {e}")
            return None

    # dict
    if isinstance(value, dict):
        try:
            return model_type.model_validate(value)
        except ValidationError as e:
            logger.debug(f"Failed to validate dict: {e}")
            return None

    # JSON 문자열
    if isinstance(value, str):
        return _parse_json_string(value, model_type)

    return None


def _parse_json_string(json_str: str, model_type: Type[_ModelT]) -> Optional[_ModelT]:
    """
    JSON 문자열을 Pydantic 모델로 파싱합니다.
    """
    try:
        data = json.loads(json_str)
        return model_type.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"Failed to parse JSON string: {e}")
        return None


def extract_tool_messages(messages: List[BaseMessage]) -> List[ToolMessage]:
    """
    메시지 리스트에서 모든 ToolMessage를 추출합니다.

    매개변수:
        messages: LangChain 메시지 리스트

    반환값:
        ToolMessage 리스트 (시간순)

    예시:
        >>> messages = agent.invoke({"messages": [...]})["messages"]
        >>> tool_msgs = extract_tool_messages(messages)
        >>> for msg in tool_msgs:
        ...     print(msg.content)
    """
    return filter_messages(messages, include_types=[ToolMessage])


def get_last_tool_message(messages: List[BaseMessage]) -> Optional[ToolMessage]:
    """
    메시지 리스트에서 마지막 ToolMessage를 반환합니다.

    매개변수:
        messages: LangChain 메시지 리스트

    반환값:
        마지막 ToolMessage 또는 None
    """
    tool_messages = extract_tool_messages(messages)
    return tool_messages[-1] if tool_messages else None


__all__ = [
    "parse_agent_response",
    "extract_tool_messages",
    "get_last_tool_message",
]
