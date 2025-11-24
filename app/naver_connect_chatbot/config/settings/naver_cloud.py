"""
Clova Studio Reranker API 설정 모듈

이 모듈은 Clova Studio의 Reranker API 설정을 관리합니다.

참고:
- Reranker는 검색 결과 재순위화를 위한 별도 API입니다
- API 문서: https://api.ncloud-docs.com/docs/clovastudio-reranker
- LLM/Embeddings와 동일한 CLOVASTUDIO_API_KEY를 사용합니다
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class ClovaStudioRerankerSettings(BaseSettings):
    """
    Clova Studio Reranker API 설정
    
    환경변수 prefix: CLOVASTUDIO_RERANKER_
    예: CLOVASTUDIO_RERANKER_ENDPOINT=https://clovastudio.stream.ntruss.com/v1/api-tools/reranker
    
    API 스펙:
    - 요청: POST /v1/api-tools/reranker
    - 헤더: Authorization: Bearer {CLOVASTUDIO_API_KEY}
    - 바디: documents, query, maxTokens (optional)
    - 최대 입력 토큰: 128,000
    - 최대 출력 토큰: 4,096
    """
    model_config = SettingsConfigDict(
        env_prefix="CLOVASTUDIO_RERANKER_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Authorization Bearer 토큰 (LLM/Embeddings와 공유)"
    )
    endpoint: str | None = Field(
        default=None,
        description="Reranker API 엔드포인트 URL"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="최대 생성 토큰 수 (기본값: 1024, 최대: 4096)"
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )


__all__ = [
    "ClovaStudioRerankerSettings",
]
