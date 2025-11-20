"""
애플리케이션 전체 설정 통합 모듈

이 모듈은 모든 도메인별 설정을 통합하여 하나의 Settings 클래스로 제공합니다.
.env 파일에서 환경변수를 자동으로 로드하며, 각 서브 설정은 자체 prefix를 가진
환경변수를 통해 개별적으로 구성됩니다.

사용 예:
    from naver_connect_chatbot.config import settings
    
    # 각 설정에 접근
    print(settings.chat.model_name)
    print(settings.qdrant_vector_store.url)
    print(settings.retriever.default_k)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT
from .llm import OpenAISettings, OpenRouterSettings
from .logging import LoggingSettings
from .naver_cloud import (
    NaverCloudChatSettings,
    NaverCloudEmbeddingsSettings,
    NaverCloudOpenAICompatibleSettings,
    NaverCloudRAGReasoningSettings,
    NaverCloudRerankerSettings,
    NaverCloudSegmentationSettings,
    NaverCloudSummarizationSettings,
)
from .retriever import (
    AdvancedHybridSettings,
    MultiQuerySettings,
    RetrieverSettings,
)
from .slack import SlackSettings
from .vector_store import QdrantVectorStoreSettings


class Settings(BaseSettings):
    """
    애플리케이션 전체 설정
    
    .env 파일에서 환경변수를 자동으로 로드하며, 각 서브 설정은
    자체 prefix를 가진 환경변수를 통해 개별적으로 구성됩니다.
    
    .env 예시:
    ```
    # Naver Cloud Embeddings
    NAVER_CLOUD_EMBEDDINGS_MODEL_URL=https://example.com/embeddings
    NAVER_CLOUD_EMBEDDINGS_API_KEY=your_api_key
    
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
    MULTI_QUERY_NUM_QUERIES=4
    MULTI_QUERY_DEFAULT_STRATEGY=rrf
    MULTI_QUERY_RRF_K=60
    MULTI_QUERY_INCLUDE_ORIGINAL=true
    
    # Advanced Hybrid Settings
    ADVANCED_HYBRID_BASE_HYBRID_WEIGHT=0.4
    ADVANCED_HYBRID_MULTI_QUERY_WEIGHT=0.6
    
    # LLM Settings (OpenAI, OpenRouter)
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL_NAME=gpt-4o-mini
    OPENAI_TEMPERATURE=0.7
    OPENAI_ENABLED=true
    
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL_NAME=anthropic/claude-3.5-sonnet
    OPENROUTER_TEMPERATURE=0.7
    OPENROUTER_ENABLED=true
    
    # Reranker Settings
    NAVER_CLOUD_RERANKER_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_RERANKER_API_KEY=your_api_key
    NAVER_CLOUD_RERANKER_API_GATEWAY_KEY=optional_gateway_key
    NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT=30.0
    NAVER_CLOUD_RERANKER_DEFAULT_TOP_K=10
    NAVER_CLOUD_RERANKER_ENABLED=false
    
    # Chat Models (Chat Completions V3)
    NAVER_CLOUD_CHAT_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_CHAT_API_KEY=your_api_key
    NAVER_CLOUD_CHAT_API_GATEWAY_KEY=optional_gateway_key
    NAVER_CLOUD_CHAT_MODEL_NAME=HCX-003
    NAVER_CLOUD_CHAT_TEMPERATURE=0.7
    NAVER_CLOUD_CHAT_MAX_TOKENS=1024
    NAVER_CLOUD_CHAT_ENABLED=true
    
    # Segmentation
    NAVER_CLOUD_SEGMENTATION_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_SEGMENTATION_API_KEY=your_api_key
    NAVER_CLOUD_SEGMENTATION_ENABLED=false
    
    # Summarization
    NAVER_CLOUD_SUMMARIZATION_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_SUMMARIZATION_API_KEY=your_api_key
    NAVER_CLOUD_SUMMARIZATION_DEFAULT_LENGTH=medium
    NAVER_CLOUD_SUMMARIZATION_ENABLED=false
    
    # RAG Reasoning
    NAVER_CLOUD_RAG_REASONING_ENDPOINT=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_RAG_REASONING_API_KEY=your_api_key
    NAVER_CLOUD_RAG_REASONING_ENABLED=false
    
    # OpenAI Compatible
    NAVER_CLOUD_OPENAI_COMPATIBLE_BASE_URL=https://clovastudio.apigw.ntruss.com/...
    NAVER_CLOUD_OPENAI_COMPATIBLE_API_KEY=your_api_key
    NAVER_CLOUD_OPENAI_COMPATIBLE_DEFAULT_MODEL=HCX-003
    NAVER_CLOUD_OPENAI_COMPATIBLE_ENABLED=false
    
    # Logging
    LOG_LEVEL=INFO
    LOG_ENABLE_CONSOLE=true
    LOG_ENABLE_FILE=true
    LOG_LOG_DIR=logs
    ```
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM 설정
    openai: OpenAISettings = Field(
        default_factory=OpenAISettings,
        description="OpenAI Chat Model 설정"
    )
    openrouter: OpenRouterSettings = Field(
        default_factory=OpenRouterSettings,
        description="OpenRouter Chat Model 설정"
    )

    # Naver Cloud 관련 설정
    naver_cloud_embeddings: NaverCloudEmbeddingsSettings = Field(
        default_factory=NaverCloudEmbeddingsSettings,
        description="Naver Cloud 임베딩 서비스 설정"
    )
    chat: NaverCloudChatSettings = Field(
        default_factory=NaverCloudChatSettings,
        description="Clova Studio Chat Completions V3 설정"
    )
    segmentation: NaverCloudSegmentationSettings = Field(
        default_factory=NaverCloudSegmentationSettings,
        description="Clova Studio 문서 분할 설정"
    )
    summarization: NaverCloudSummarizationSettings = Field(
        default_factory=NaverCloudSummarizationSettings,
        description="Clova Studio 문서 요약 설정"
    )
    rag_reasoning: NaverCloudRAGReasoningSettings = Field(
        default_factory=NaverCloudRAGReasoningSettings,
        description="Clova Studio RAG Reasoning 설정"
    )
    reranker: NaverCloudRerankerSettings = Field(
        default_factory=NaverCloudRerankerSettings,
        description="Clova Studio Reranker 설정"
    )
    openai_compatible: NaverCloudOpenAICompatibleSettings = Field(
        default_factory=NaverCloudOpenAICompatibleSettings,
        description="Clova Studio OpenAI 호환 API 설정"
    )

    # 벡터 저장소 설정
    qdrant_vector_store: QdrantVectorStoreSettings = Field(
        default_factory=QdrantVectorStoreSettings,
        description="Qdrant 벡터 저장소 설정"
    )

    # Retriever 관련 설정
    retriever: RetrieverSettings = Field(
        default_factory=RetrieverSettings,
        description="Retriever 기본 설정"
    )
    multi_query: MultiQuerySettings = Field(
        default_factory=MultiQuerySettings,
        description="MultiQuery 설정"
    )
    advanced_hybrid: AdvancedHybridSettings = Field(
        default_factory=AdvancedHybridSettings,
        description="Advanced Hybrid 설정"
    )

    # 로깅 설정
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="로깅 설정"
    )

    # Slack 설정
    slack: SlackSettings = Field(
        default_factory=SlackSettings,
        description="Slack App 설정"
    )


# ============================================================================
# 전역 settings 인스턴스
# ============================================================================

settings = Settings()


__all__ = [
    "Settings",
    "settings",
]

