"""
Retriever 모듈

이 패키지는 다양한 검색 전략을 제공합니다:
- HybridRetriever: Dense + Sparse 하이브리드 검색
- MultiQueryRetriever: LLM 기반 다중 쿼리 검색
- KiwiBM25Retriever: 한국어 형태소 분석 기반 BM25 검색
- QdrantVDBRetriever: Qdrant 벡터 DB 검색

팩토리 함수들은 naver_connect_chatbot.rag.retriever_factory 에서 import 하세요.
"""

from naver_connect_chatbot.rag.retriever.hybrid_retriever import (
    HybridRetriever,
    HybridMethod,
)
from naver_connect_chatbot.rag.retriever.kiwi_bm25_retriever import (
    KiwiBM25Retriever,
)
from naver_connect_chatbot.rag.retriever.multi_query_retriever import (
    MultiQueryRetriever,
)
from naver_connect_chatbot.rag.retriever.qdrant_sdk_retriever import (
    QdrantVDBRetriever,
)

__all__ = [
    # 검색기 클래스
    "HybridRetriever",
    "HybridMethod",
    "KiwiBM25Retriever",
    "MultiQueryRetriever",
    "QdrantVDBRetriever",
]
