"""
Configuration module for the Naver Connect Chatbot.

이 모듈은 설정 클래스, Enum, LLM/Embeddings 팩토리, 그리고 로거를 애플리케이션 전역에 노출합니다.
모든 설정은 settings 객체를 통해 접근하고, 로깅은 logger 객체를 통해 사용합니다.

사용 예:
    from naver_connect_chatbot.config import settings, logger, get_chat_model, get_embeddings

    logger.info("Application started")
    print(settings.clova_llm.model)
    print(settings.qdrant_vector_store.url)

    # LLM 및 Embeddings 사용 (langchain_naver)
    llm = get_chat_model()
    embeddings = get_embeddings()
"""

# LLM and Embeddings
from naver_connect_chatbot.config.llm import get_chat_model
from naver_connect_chatbot.config.embedding import get_embeddings

# Settings
from naver_connect_chatbot.config.settings import (
    PROJECT_ROOT,
    AdvancedHybridSettings,
    ClovaXLLMSettings,
    ClovaXEmbeddingsSettings,
    ClovaStudioRerankerSettings,
    ClovaStudioRAGReasoningSettings,
    HybridMethodType,
    LoggingSettings,
    MultiQuerySettings,
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

# Monitoring
from naver_connect_chatbot.config.monitoring import LangfuseSettings

__all__ = [
    # Base
    "PROJECT_ROOT",
    # Main Settings
    "Settings",
    "settings",
    # LLM and Embeddings (langchain_naver)
    "get_chat_model",
    "get_embeddings",
    # Clova X Settings
    "ClovaXLLMSettings",
    "ClovaXEmbeddingsSettings",
    # Clova Studio Reranker
    "ClovaStudioRerankerSettings",
    # Clova Studio RAG Reasoning
    "ClovaStudioRAGReasoningSettings",
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
    # Monitoring
    "LangfuseSettings",
    # Logger
    "logger",
]
