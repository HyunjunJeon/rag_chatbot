"""
LLM 설정 모듈

이 모듈은 langchain_naver의 ChatClovaX를 사용하여 LLM 인스턴스를 생성합니다.

사용 예:
    from naver_connect_chatbot.config.llm import get_chat_model

    # 기본 설정으로 LLM 생성
    llm = get_chat_model()
    response = await llm.ainvoke("안녕하세요!")

    # 설정 오버라이드
    llm = get_chat_model(model="HCX-003", temperature=0.9)
"""

from typing import TYPE_CHECKING, Any

from langchain_naver import ChatClovaX

if TYPE_CHECKING:
    from naver_connect_chatbot.config.settings.main import Settings


def get_chat_model(
    settings_obj: "Settings | None" = None,
    **kwargs: Any,
) -> ChatClovaX:
    """
    ChatClovaX 인스턴스를 생성합니다.

    이 팩토리 함수는 설정에서 읽은 값으로 ChatClovaX를 초기화하며,
    kwargs를 통해 일부 설정을 오버라이드할 수 있습니다.

    매개변수:
        settings_obj: Settings 인스턴스 (None이면 전역 settings 사용)
        **kwargs: ChatClovaX 파라미터 오버라이드
            - model: 모델명 오버라이드
            - temperature: 온도 오버라이드
            - max_tokens: 최대 토큰 수 오버라이드
            - thinking: Thinking 모드 설정 (예: {"effort": "low"})
            - 기타 ChatClovaX가 지원하는 모든 파라미터

    반환값:
        ChatClovaX 인스턴스

    예외:
        ValueError: API 키가 설정되지 않은 경우

    예시:
        >>> from naver_connect_chatbot.config.llm import get_chat_model
        >>>
        >>> # 기본 설정으로 모델 생성
        >>> llm = get_chat_model()
        >>>
        >>> # 설정 오버라이드
        >>> llm = get_chat_model(
        ...     model="HCX-007",
        ...     temperature=0.9,
        ...     thinking={"effort": "high"}
        ... )
    """
    # 순환 import 방지를 위해 여기서 import
    from naver_connect_chatbot.config.settings.main import settings

    if settings_obj is None:
        settings_obj = settings

    config = settings_obj.clova_llm

    # API 키 검증
    if not config.api_key:
        raise ValueError(
            "CLOVASTUDIO_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 CLOVASTUDIO_API_KEY를 설정하세요."
        )

    # reasoning 관련 제어 플래그 (기본값: reasoning 비활성)
    use_reasoning_flag = kwargs.get("use_reasoning")
    reasoning_effort_override = kwargs.get("reasoning_effort")
    use_reasoning = bool(use_reasoning_flag) or reasoning_effort_override is not None

    # 기본 파라미터 구성
    chat_params: dict[str, Any] = {
        "model": kwargs.get("model", config.model),
        "api_key": config.api_key.get_secret_value(),
        "temperature": kwargs.get("temperature", config.temperature),
    }

    # max_tokens 설정
    max_tokens_value = kwargs.get("max_tokens", config.max_tokens)
    if max_tokens_value is not None and max_tokens_value > 1:
        chat_params["max_tokens"] = max_tokens_value

    # thinking / reasoning 설정 (config의 thinking_effort 또는 kwargs의 thinking/reasoning_effort)
    if "thinking" in kwargs:
        # kwargs에서 직접 전달된 경우 (우선순위 높음)
        chat_params["thinking"] = kwargs["thinking"]
    else:
        # use_reasoning이 활성화된 경우에만 settings의 thinking_effort 또는 override 적용
        effective_effort = reasoning_effort_override or config.thinking_effort
        if use_reasoning and effective_effort:
            # ChatClovaX는 reasoning_effort 파라미터를 통해 Reasoning 모드를 제어합니다.
            chat_params["reasoning_effort"] = effective_effort

    # kwargs에서 이미 처리한 키들 제외
    excluded_keys = {
        "model",
        "temperature",
        "max_tokens",
        "thinking",
        "use_reasoning",
        "reasoning_effort",
    }
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

    return ChatClovaX(**chat_params, **extra_kwargs)


__all__ = [
    "get_chat_model",
]
