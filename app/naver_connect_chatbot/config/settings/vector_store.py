"""
벡터 저장소 설정 모듈

이 모듈은 벡터 데이터베이스 설정을 관리합니다.
현재는 Qdrant를 지원하며, 향후 다른 벡터 DB 설정도 추가할 수 있습니다.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class QdrantVectorStoreSettings(BaseSettings):
    """
    Qdrant 벡터 저장소 설정
    
    환경변수 prefix: QDRANT_
    예: QDRANT_URL=http://localhost:6333
    """
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    url: str | None = Field(
        default=None,
        description="Qdrant 인스턴스 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API 키 (선택적)"
    )
    collection_name: str = Field(
        default="default",
        description="사용할 컬렉션 이름"
    )
    embedding_dimensions: int = Field(
        default=1024,
        description="임베딩 차원 수"
    )


__all__ = [
    "QdrantVectorStoreSettings",
]

