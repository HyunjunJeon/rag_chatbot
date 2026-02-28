"""
Gemini LLM 설정 모듈

이 모듈은 langchain_google_genai의 ChatGoogleGenerativeAI를 위한
설정 클래스를 제공합니다. ClovaX Embeddings/Reranker는 기존 clova.py에서 유지합니다.

Gemini 3.1 Pro 모델 스펙:
    - 컨텍스트 윈도우: 1M 토큰 입력, 64K 토큰 출력
    - thinking_level 기본값: "high" (별도 설정 불필요)
    - temperature 기본값: 1.0 (변경 시 예상치 못한 동작 가능)
    - 지원 도구: Google Search, Function Calling, Structured Output

환경변수 예시:
    GOOGLE_API_KEY=your-google-api-key
    GEMINI_MODEL=gemini-3.1-pro-preview
    GEMINI_TEMPERATURE=1.0
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class GeminiLLMSettings(BaseSettings):
    """
    Gemini LLM 설정 (langchain_google_genai.ChatGoogleGenerativeAI)

    환경변수 prefix: GEMINI_
    예: GEMINI_MODEL=gemini-3.1-pro-preview

    ChatGoogleGenerativeAI는 다음 파라미터를 사용합니다:
    - google_api_key: GOOGLE_API_KEY 환경변수에서 자동 로드
    - model: 모델명 (기본값: gemini-3.1-pro-preview)
    - temperature: 응답의 창의성 조절 (Gemini 3 권장: 1.0)
    - max_output_tokens: 생성할 최대 토큰 수 (최대 64K)
    - thinking: Thinking 모드 설정 (기본 high, 별도 설정 불필요)
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="google_api_key",
        description="GOOGLE_API_KEY - Google AI API 키",
    )
    model: str = Field(
        default="gemini-3.1-pro-preview",
        description="사용할 Gemini 모델명 (gemini-2.5-pro, gemini-3.1-pro-preview 등)",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="응답의 창의성 조절. Gemini 3 모델은 1.0 유지를 강력 권장",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="생성할 최대 토큰 수 (None이면 모델 기본값 사용, 최대 64K)",
    )


__all__ = [
    "GeminiLLMSettings",
]
