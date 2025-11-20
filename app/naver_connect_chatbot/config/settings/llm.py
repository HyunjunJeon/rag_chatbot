"""
LLM 설정 클래스 모듈

OpenAI와 OpenRouter의 설정 클래스를 정의합니다.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class OpenAISettings(BaseSettings):
    """
    OpenAI Chat Model 설정
    
    환경변수 prefix: OPENAI_
    예: OPENAI_API_KEY=sk-...
    
    OpenAI의 공식 Chat Completions API를 사용합니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )
    
    api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API 키"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="사용할 모델명"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="응답의 창의성 조절 (0.0: 결정적, 2.0: 창의적)"
    )
    max_tokens: int | None = Field(
        default=None,
        ge=-1,
        description="생성할 최대 토큰 수 (None 또는 -1이면 제한 없음)"
    )
    enabled: bool = Field(
        default=False,
        description="OpenAI 활성화 여부"
    )


class OpenRouterSettings(BaseSettings):
    """
    OpenRouter Chat Model 설정
    
    환경변수 prefix: OPENROUTER_
    예: OPENROUTER_API_KEY=sk-or-...
    
    OpenRouter는 다양한 LLM을 통합된 API로 제공하는 서비스입니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )
    
    api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API 키"
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API Base URL"
    )
    model_name: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="사용할 모델명 (예: anthropic/claude-3.5-sonnet, openai/gpt-4o)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="응답의 창의성 조절 (0.0: 결정적, 2.0: 창의적)"
    )
    max_tokens: int | None = Field(
        default=None,
        ge=-1,
        description="생성할 최대 토큰 수 (None 또는 -1이면 제한 없음)"
    )
    enabled: bool = Field(
        default=False,
        description="OpenRouter 활성화 여부"
    )


__all__ = [
    "OpenAISettings",
    "OpenRouterSettings",
]

