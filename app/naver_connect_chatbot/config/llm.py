"""
LLM 설정 모듈

이 모듈은 OpenAI, OpenRouter, Naver Cloud 세 가지 LLM 제공자를
ChatOpenAI를 활용하여 OpenAI Compatible 형태로 통합합니다.

사용 예:
    from naver_connect_chatbot.config.llm import get_chat_model, LLMProvider
    
    # OpenAI 사용
    openai_model = get_chat_model(LLMProvider.OPENAI)
    response = await openai_model.ainvoke("Hello!")
    
    # OpenRouter 사용
    openrouter_model = get_chat_model(LLMProvider.OPENROUTER)
    
    # Naver Cloud 사용
    naver_model = get_chat_model(LLMProvider.NAVER_CLOUD)
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

from naver_connect_chatbot.config.settings.llm import (
    OpenAISettings,
    OpenRouterSettings,
)
from naver_connect_chatbot.config.settings.naver_cloud import (
    NaverCloudOpenAICompatibleSettings,
)

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.main import Settings


class LLMProvider(str, Enum):
    """
    LLM 제공자 타입
    
    각 제공자는 ChatOpenAI를 통해 OpenAI 호환 API로 접근됩니다.
    """
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    NAVER_CLOUD = "naver_cloud"


# ============================================================================
# 팩토리 함수
# ============================================================================


def get_chat_model(
    provider: LLMProvider = LLMProvider.OPENAI,
    settings_obj: "Settings | None" = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    LLM 제공자에 따른 ChatOpenAI 인스턴스 생성
    
    이 팩토리 함수는 제공자 타입에 따라 적절한 설정으로
    ChatOpenAI 인스턴스를 생성합니다. 모든 제공자는 동일한
    ChatOpenAI 클래스를 사용하되, base_url과 헤더만 다르게 설정됩니다.
    
    매개변수:
        provider: LLM 제공자 타입 (OPENAI, OPENROUTER, NAVER_CLOUD)
        settings_obj: Settings 인스턴스 (None이면 전역 settings 사용)
        **kwargs: ChatOpenAI 추가 파라미터 (설정 오버라이드용)
            - model: 모델명 오버라이드
            - temperature: 온도 오버라이드
            - max_tokens: 최대 토큰 수 오버라이드
            - 기타 ChatOpenAI가 지원하는 모든 파라미터
    
    반환값:
        ChatOpenAI 인스턴스
    
    예외:
        ValueError: 지원하지 않는 제공자이거나 필수 설정이 누락된 경우
    
    예시:
        >>> from naver_connect_chatbot.config.llm import get_chat_model, LLMProvider
        >>> 
        >>> # 기본 설정으로 OpenAI 모델 생성
        >>> model = get_chat_model(LLMProvider.OPENAI)
        >>> 
        >>> # 설정 오버라이드
        >>> model = get_chat_model(
        ...     LLMProvider.OPENAI,
        ...     model="gpt-4o",
        ...     temperature=0.9
        ... )
    """
    # 순환 import 방지를 위해 여기서 import
    from naver_connect_chatbot.config.settings.main import settings
    
    if settings_obj is None:
        settings_obj = settings
    
    if provider == LLMProvider.OPENAI:
        return _create_openai_chat_model(settings_obj.openai, **kwargs)
    elif provider == LLMProvider.OPENROUTER:
        return _create_openrouter_chat_model(settings_obj.openrouter, **kwargs)
    elif provider == LLMProvider.NAVER_CLOUD:
        return _create_naver_cloud_chat_model(settings_obj.openai_compatible, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_openai_chat_model(
    config: OpenAISettings,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    OpenAI ChatModel 인스턴스 생성
    
    매개변수:
        config: OpenAI 설정
        **kwargs: ChatOpenAI 추가 파라미터
    
    반환값:
        ChatOpenAI 인스턴스
    
    예외:
        ValueError: OpenAI가 비활성화되어 있거나 API 키가 누락된 경우
    """
    if not config.enabled or not config.api_key:
        raise ValueError("OpenAI is not enabled or API key is missing")
    
    # 오버라이드 가능한 파라미터들
    excluded_keys = {"model", "temperature", "max_tokens"}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}
    
    # 기본 파라미터 구성
    chat_params: dict[str, Any] = {
        "model": kwargs.get("model", config.model_name),
        "api_key": config.api_key.get_secret_value() if config.api_key else None,
        "temperature": kwargs.get("temperature", config.temperature),
    }
    
    # max_tokens가 None이나 -1이 아닌 경우에만 설정 (-1은 무제한 의미)
    max_tokens_value = kwargs.get("max_tokens", config.max_tokens)
    if max_tokens_value is not None and max_tokens_value != -1:
        chat_params["max_tokens"] = max_tokens_value
    
    return ChatOpenAI(**chat_params, **extra_kwargs)


def _create_openrouter_chat_model(
    config: OpenRouterSettings,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    OpenRouter ChatModel 인스턴스 생성
    
    OpenRouter는 OpenAI 호환 API를 제공하므로 ChatOpenAI를
    base_url 설정만으로 사용할 수 있습니다.
    
    매개변수:
        config: OpenRouter 설정
        **kwargs: ChatOpenAI 추가 파라미터
    
    반환값:
        ChatOpenAI 인스턴스
    
    예외:
        ValueError: OpenRouter가 비활성화되어 있거나 API 키가 누락된 경우
    """
    if not config.enabled or not config.api_key:
        raise ValueError("OpenRouter is not enabled or API key is missing")
    
    # 오버라이드 가능한 파라미터들
    excluded_keys = {"model", "base_url", "temperature", "max_tokens"}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}
    
    # 기본 파라미터 구성
    chat_params: dict[str, Any] = {
        "model": kwargs.get("model", config.model_name),
        "base_url": kwargs.get("base_url", config.base_url),
        "api_key": config.api_key.get_secret_value() if config.api_key else None,
        "temperature": kwargs.get("temperature", config.temperature),
    }
    
    # max_tokens가 None이나 -1이 아닌 경우에만 설정 (-1은 무제한 의미)
    max_tokens_value = kwargs.get("max_tokens", config.max_tokens)
    if max_tokens_value is not None and max_tokens_value != -1:
        chat_params["max_tokens"] = max_tokens_value
    
    return ChatOpenAI(**chat_params, **extra_kwargs)


def _create_naver_cloud_chat_model(
    config: NaverCloudOpenAICompatibleSettings,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Naver Cloud ChatModel 인스턴스 생성 (OpenAI 호환 모드)
    
    Naver Cloud Clova Studio는 OpenAI 호환 API를 제공하지만,
    인증을 위해 커스텀 헤더(X-NCP-CLOVASTUDIO-API-KEY, X-NCP-APIGW-API-KEY)를
    필요로 합니다. 이 헤더들은 default_headers 파라미터를 통해 전달됩니다.
    
    매개변수:
        config: Naver Cloud OpenAI Compatible 설정
        **kwargs: ChatOpenAI 추가 파라미터
    
    반환값:
        ChatOpenAI 인스턴스
    
    예외:
        ValueError: Naver Cloud가 비활성화되어 있거나 필수 설정이 누락된 경우
    """
    if not config.enabled or not config.api_key or not config.base_url:
        raise ValueError(
            "Naver Cloud OpenAI Compatible is not enabled or required settings are missing"
        )
    
    # Naver Cloud 커스텀 헤더 설정
    default_headers = {
        "X-NCP-CLOVASTUDIO-API-KEY": config.api_key.get_secret_value() if config.api_key else "",
    }

    # API Gateway 키가 있으면 추가
    if config.api_gateway_key:
        default_headers["X-NCP-APIGW-API-KEY"] = config.api_gateway_key.get_secret_value()
    
    # 오버라이드 가능한 파라미터들
    excluded_keys = {"model", "base_url", "api_key", "default_headers", "max_tokens"}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}
    
    # 기본 파라미터 구성
    chat_params: dict[str, Any] = {
        "model": kwargs.get("model", config.default_model),
        "base_url": config.base_url,
        "api_key": config.api_key.get_secret_value() if config.api_key else None,  # 기본 인증용 (OpenAI SDK 호환성)
        "default_headers": default_headers,  # Naver Cloud 전용 헤더
    }
    
    # max_tokens가 -1이 아닌 경우에만 설정
    max_tokens_value = kwargs.get("max_tokens", config.max_tokens if hasattr(config, "max_tokens") else None)
    if max_tokens_value is not None and max_tokens_value != -1:
        chat_params["max_tokens"] = max_tokens_value
    
    return ChatOpenAI(**chat_params, **extra_kwargs)


__all__ = [
    "LLMProvider",
    "OpenAISettings",
    "OpenRouterSettings",
    "get_chat_model",
]
