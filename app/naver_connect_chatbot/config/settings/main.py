"""
애플리케이션 전체 설정 통합 모듈

이 모듈은 모든 도메인별 설정을 통합하여 하나의 Settings 클래스로 제공합니다.
.env 파일에서 환경변수를 자동으로 로드하며, 각 서브 설정은 자체 prefix를 가진
환경변수를 통해 개별적으로 구성됩니다.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT
from .clova import (
    ClovaXLLMSettings,
    ClovaXEmbeddingsSettings,
    ClovaStudioRerankerSettings,
    ClovaStudioRAGReasoningSettings,
)
from .logging import LoggingSettings
from .retriever import (
    AdvancedHybridSettings,
    MultiQuerySettings,
    RetrieverSettings,
)
from .slack import SlackSettings
from .vector_store import QdrantVectorStoreSettings
from ..monitoring import LangfuseSettings


class Settings(BaseSettings):
    """
    애플리케이션 전체 설정

    .env 파일에서 환경변수를 자동으로 로드하며, 각 서브 설정은
    자체 prefix를 가진 환경변수를 통해 개별적으로 구성됩니다.

    .env 예시:
    ```
    # Clova X Settings (LLM + Embeddings via langchain_naver)
    CLOVASTUDIO_API_KEY=your_clovastudio_api_key
    CLOVA_MODEL=HCX-007
    CLOVA_EMBEDDINGS_MODEL=bge-m3
    CLOVA_TEMPERATURE=0.7
    CLOVA_MAX_TOKENS=
    CLOVA_THINKING_EFFORT=low

    # Naver Cloud Reranker (별도 API)
    NAVER_CLOUD_RERANKER_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT=30.0
    NAVER_CLOUD_RERANKER_DEFAULT_TOP_K=10
    NAVER_CLOUD_RERANKER_ENABLED=false

    # Qdrant Vector Store
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=optional_api_key
    QDRANT_COLLECTION_NAME=my_collection
    QDRANT_EMBEDDING_DIMENSIONS=1024

    # Retriever Settings
    RETRIEVER_DEFAULT_K=10
    RETRIEVER_DEFAULT_SPARSE_WEIGHT=0.5
    RETRIEVER_DEFAULT_DENSE_WEIGHT=0.5
    RETRIEVER_DEFAULT_RRF_C=60

    # MultiQuery Settings
    MULTI_QUERY_NUM_QUERIES=5
    MULTI_QUERY_DEFAULT_STRATEGY=rrf
    MULTI_QUERY_RRF_K=60
    MULTI_QUERY_INCLUDE_ORIGINAL=true

    # Advanced Hybrid Settings
    ADVANCED_HYBRID_BASE_HYBRID_WEIGHT=0.6
    ADVANCED_HYBRID_MULTI_QUERY_WEIGHT=0.4

    # Logging
    LOG_LEVEL=INFO
    LOG_ENABLE_CONSOLE=true
    LOG_ENABLE_FILE=true
    LOG_LOG_DIR=logs

    # Slack
    SLACK_BOT_TOKEN=xoxb-...
    SLACK_SIGNING_SECRET=...
    SLACK_APP_TOKEN=xapp-...
    ```
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Clova X 설정 (langchain_naver)
    clova_llm: ClovaXLLMSettings = Field(
        default_factory=ClovaXLLMSettings,
        description="Clova X LLM 설정 (langchain_naver.ChatClovaX)",
    )
    clova_embeddings: ClovaXEmbeddingsSettings = Field(
        default_factory=ClovaXEmbeddingsSettings,
        description="Clova X Embeddings 설정 (langchain_naver.ClovaXEmbeddings)",
    )

    # Clova Studio Reranker (별도 API)
    reranker: ClovaStudioRerankerSettings = Field(
        default_factory=ClovaStudioRerankerSettings, description="Clova Studio Reranker 설정"
    )

    # Clova Studio RAG Reasoning (별도 API)
    rag_reasoning: ClovaStudioRAGReasoningSettings = Field(
        default_factory=ClovaStudioRAGReasoningSettings,
        description="Clova Studio RAG Reasoning 설정",
    )

    # 벡터 저장소 설정
    qdrant_vector_store: QdrantVectorStoreSettings = Field(
        default_factory=QdrantVectorStoreSettings, description="Qdrant 벡터 저장소 설정"
    )

    # Retriever 관련 설정
    retriever: RetrieverSettings = Field(
        default_factory=RetrieverSettings, description="Retriever 기본 설정"
    )
    multi_query: MultiQuerySettings = Field(
        default_factory=MultiQuerySettings, description="MultiQuery 설정"
    )
    advanced_hybrid: AdvancedHybridSettings = Field(
        default_factory=AdvancedHybridSettings, description="Advanced Hybrid 설정"
    )

    # 로깅 설정
    logging: LoggingSettings = Field(default_factory=LoggingSettings, description="로깅 설정")

    # Slack 설정
    slack: SlackSettings = Field(default_factory=SlackSettings, description="Slack App 설정")

    # LangFuse Monitoring
    langfuse: LangfuseSettings = Field(
        default_factory=LangfuseSettings, description="LangFuse monitoring configuration"
    )


# ============================================================================
# 전역 settings 인스턴스
# ============================================================================

settings = Settings()


__all__ = [
    "Settings",
    "settings",
]
