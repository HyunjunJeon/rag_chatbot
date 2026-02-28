"""
Settings 패키지

이 패키지는 애플리케이션의 모든 설정을 관리합니다.
각 도메인별로 모듈이 분리되어 있으며, 모든 설정 클래스를
이 __init__.py를 통해 통합적으로 접근할 수 있습니다.

구조:
- base.py: 공통 상수 (PROJECT_ROOT)
- enums.py: Enum 타입 정의
- clova.py: Clova X LLM/Embeddings/Reranker/Segmentation/Summarization/RAG Reasoning 설정
- vector_store.py: 벡터 저장소 설정
- retriever.py: Retriever 관련 설정 (3개 클래스)
- logging.py: 로깅 설정
- slack.py: Slack 설정
- main.py: 통합 Settings 클래스 + 전역 settings 인스턴스
"""

# Base
from .base import PROJECT_ROOT

# Enums
from .enums import HybridMethodType, RetrieverStrategy

# Gemini LLM Settings (langchain_google_genai)
from .gemini import GeminiLLMSettings

# Clova X Settings (langchain_naver) — Embeddings/Reranker용 유지
from .clova import (
    ClovaXLLMSettings,
    ClovaXEmbeddingsSettings,
    ClovaStudioRerankerSettings,
    ClovaStudioSegmentationSettings,
    ClovaStudioSummarizationSettings,
    ClovaStudioRAGReasoningSettings,
)

# Logging Settings
from .logging import LoggingSettings

# Retriever Settings
from .retriever import (
    AdvancedHybridSettings,
    MultiQuerySettings,
    RetrieverSettings,
)

# Vector Store Settings
from .vector_store import QdrantVectorStoreSettings

# Slack Settings
from .slack import SlackSettings

# Main Settings
from .main import Settings, settings

__all__ = [
    # Base
    "PROJECT_ROOT",
    # Main Settings
    "Settings",
    "settings",
    # Gemini LLM Settings
    "GeminiLLMSettings",
    # Clova X Settings
    "ClovaXLLMSettings",
    "ClovaXEmbeddingsSettings",
    "ClovaStudioRerankerSettings",
    "ClovaStudioSegmentationSettings",
    "ClovaStudioSummarizationSettings",
    "ClovaStudioRAGReasoningSettings",
    # Vector Store Settings
    "QdrantVectorStoreSettings",
    # Retriever Settings
    "RetrieverSettings",
    "MultiQuerySettings",
    "AdvancedHybridSettings",
    # Logging Settings
    "LoggingSettings",
    # Slack Settings
    "SlackSettings",
    # Enum 타입
    "RetrieverStrategy",
    "HybridMethodType",
]
