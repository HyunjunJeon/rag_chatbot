"""
LLM 설정 모듈

이 모듈은 langchain_google_genai의 ChatGoogleGenerativeAI를 사용하여 LLM 인스턴스를 생성합니다.

Gemini 3.1 Pro 특성:
    - thinking_level 기본값이 "high"이므로 별도 thinking 설정 불필요
    - temperature는 1.0 유지 권장 (변경 시 예상치 못한 동작 가능)
    - with_structured_output() 네이티브 지원
    - Google Search grounding 도구 지원

사용 예:
    from naver_connect_chatbot.config.llm import get_chat_model

    # 기본 설정으로 LLM 생성 (thinking_level=high 자동 적용)
    llm = get_chat_model()
    response = await llm.ainvoke("안녕하세요!")

    # thinking_level 조정이 필요한 경우 (분류 등 가벼운 작업)
    llm = get_chat_model(thinking_level="low")
"""

from typing import TYPE_CHECKING, Any

from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.main import Settings


def get_chat_model(
    settings_obj: "Settings | None" = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """
    ChatGoogleGenerativeAI 인스턴스를 생성합니다.

    이 팩토리 함수는 설정에서 읽은 값으로 Gemini LLM을 초기화하며,
    kwargs를 통해 일부 설정을 오버라이드할 수 있습니다.

    Gemini 3.1 Pro는 기본적으로 thinking_level="high"를 사용합니다.
    분류나 간단한 작업에는 thinking_level="low" 또는 "minimal"을 전달하세요.

    매개변수:
        settings_obj: Settings 인스턴스 (None이면 전역 settings 사용)
        **kwargs: ChatGoogleGenerativeAI 파라미터 오버라이드
            - model: 모델명 오버라이드
            - temperature: 온도 오버라이드 (Gemini 3은 1.0 유지 권장)
            - max_output_tokens: 최대 출력 토큰 수 오버라이드
            - thinking_level: "minimal", "low", "medium", "high" (기본 high)
            - 기타 ChatGoogleGenerativeAI가 지원하는 모든 파라미터

        [하위 호환성] 기존 ClovaX 팩토리의 kwargs도 자동 매핑:
            - use_reasoning: True → thinking_level 유지 (기본 high)
            - reasoning_effort: "low"/"medium"/"high" → thinking_level 매핑

    반환값:
        ChatGoogleGenerativeAI 인스턴스

    예외:
        ValueError: API 키가 설정되지 않은 경우

    예시:
        >>> from naver_connect_chatbot.config.llm import get_chat_model
        >>>
        >>> # 기본 설정으로 모델 생성 (thinking_level=high)
        >>> llm = get_chat_model()
        >>>
        >>> # 분류 작업용 (가벼운 thinking)
        >>> llm = get_chat_model(thinking_level="low")
        >>>
        >>> # 하위 호환성: 기존 reasoning_effort 인터페이스도 동작
        >>> llm = get_chat_model(use_reasoning=True, reasoning_effort="medium")
    """
    # 순환 import 방지를 위해 여기서 import
    from naver_connect_chatbot.config.settings.main import settings

    if settings_obj is None:
        settings_obj = settings

    config = settings_obj.gemini_llm

    # API 키 검증
    if not config.api_key:
        raise ValueError(
            "GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일에 GOOGLE_API_KEY를 설정하세요."
        )

    # 기본 파라미터 구성
    chat_params: dict[str, Any] = {
        "model": kwargs.get("model", config.model),
        "google_api_key": config.api_key.get_secret_value(),
        "temperature": kwargs.get("temperature", config.temperature),
    }

    # max_output_tokens 설정
    max_tokens_value = kwargs.get("max_output_tokens", config.max_output_tokens)
    if max_tokens_value is not None and max_tokens_value > 0:
        chat_params["max_output_tokens"] = max_tokens_value

    # thinking_level 설정 (Gemini 3 기본값: high)
    # 직접 thinking_level이 전달된 경우 우선 적용
    if "thinking_level" in kwargs:
        chat_params["thinking_level"] = kwargs["thinking_level"]
    # 하위 호환성: reasoning_effort → thinking_level 매핑
    elif "reasoning_effort" in kwargs:
        chat_params["thinking_level"] = kwargs["reasoning_effort"]

    # kwargs에서 이미 처리한 키들 제외
    excluded_keys = {
        "model",
        "temperature",
        "max_output_tokens",
        "max_tokens",
        "thinking_level",
        "thinking",
        "use_reasoning",
        "reasoning_effort",
    }
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

    return ChatGoogleGenerativeAI(**chat_params, **extra_kwargs)


__all__ = [
    "get_chat_model",
]
