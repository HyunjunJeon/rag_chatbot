"""
Configuration module for the Naver Connect Chatbot.

이 모듈은 설정 클래스, Enum, LLM 팩토리, 그리고 로거를 애플리케이션 전역에 노출합니다.
모든 설정은 settings 객체를 통해 접근하고, 로깅은 logger 객체를 통해 사용합니다.

사용 예:
    from naver_connect_chatbot.config import settings, logger, get_chat_model, LLMProvider
    
    logger.info("Application started")
    print(settings.chat.model_name)
    print(settings.qdrant_vector_store.url)
    
    # LLM 사용
    openai_model = get_chat_model(LLMProvider.OPENAI)
"""

# LLM
from naver_connect_chatbot.config.llm import (
    LLMProvider,
    OpenAISettings,
    OpenRouterSettings,
    get_chat_model,
)

# Settings
from naver_connect_chatbot.config.settings import (
    PROJECT_ROOT,
    AdvancedHybridSettings,
    HybridMethodType,
    LoggingSettings,
    MultiQuerySettings,
    NaverCloudChatSettings,
    NaverCloudEmbeddingsSettings,
    NaverCloudOpenAICompatibleSettings,
    NaverCloudRAGReasoningSettings,
    NaverCloudRerankerSettings,
    NaverCloudSegmentationSettings,
    NaverCloudSummarizationSettings,
    QdrantVectorStoreSettings,
    RetrieverSettings,
    RetrieverStrategy,
    Settings,
    SlackSettings,
    settings,
)

# Adaptive RAG Settings
from naver_connect_chatbot.config.settings.rag_settings import (
    AdaptiveRAGSettings,
    adaptive_rag_settings,
    get_adaptive_rag_settings,
    update_adaptive_rag_settings,
)

# Logger
from naver_connect_chatbot.config.log import logger

__all__ = [
    # Base
    "PROJECT_ROOT",
    # Main Settings
    "Settings",
    "settings",
    # LLM
    "LLMProvider",
    "OpenAISettings",
    "OpenRouterSettings",
    "get_chat_model",
    # Naver Cloud Settings
    "NaverCloudEmbeddingsSettings",
    "NaverCloudChatSettings",
    "NaverCloudSegmentationSettings",
    "NaverCloudSummarizationSettings",
    "NaverCloudRAGReasoningSettings",
    "NaverCloudRerankerSettings",
    "NaverCloudOpenAICompatibleSettings",
    # Vector Store Settings
    "QdrantVectorStoreSettings",
    # Retriever Settings
    "RetrieverSettings",
    "MultiQuerySettings",
    "AdvancedHybridSettings",
    # Adaptive RAG Settings
    "AdaptiveRAGSettings",
    "adaptive_rag_settings",
    "get_adaptive_rag_settings",
    "update_adaptive_rag_settings",
    # Logging Settings
    "LoggingSettings",
    # Slack Settings
    "SlackSettings",
    # Enum 타입
    "RetrieverStrategy",
    "HybridMethodType",
    # Logger
    "logger",
]
