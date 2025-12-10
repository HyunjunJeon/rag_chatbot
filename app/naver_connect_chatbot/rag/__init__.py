"""
RAG (Retrieval-Augmented Generation) 패키지

이 패키지는 RAG 시스템의 핵심 구성 요소들을 제공합니다:
- Reranker: 검색 결과 재정렬
- Segmenter: 텍스트 문단 나누기
- Summarizer: 텍스트 요약
- RAG Reasoning: Function calling 기반 RAG 추론
- Schema Registry: VectorDB 스키마 정보 캐시 및 제공
"""

from .rerank import BaseReranker, ClovaStudioReranker
from .segmentation import BaseSegmenter, ClovaStudioSegmenter, SegmentationResult
from .summarization import BaseSummarizer, ClovaStudioSummarizer, SummarizationResult
from .rag_reasoning import (
    ClovaStudioRAGReasoning,
    convert_langchain_tool_to_rag_reasoning,
    convert_langchain_tools_to_rag_reasoning,
)
from .schema_registry import (
    SchemaRegistry,
    VectorDBSchema,
    DataSourceInfo,
    get_schema_registry,
    get_data_source_context,
)

__all__ = [
    # Reranker
    "BaseReranker",
    "ClovaStudioReranker",
    # Segmentation
    "BaseSegmenter",
    "ClovaStudioSegmenter",
    "SegmentationResult",
    # Summarization
    "BaseSummarizer",
    "ClovaStudioSummarizer",
    "SummarizationResult",
    # RAG Reasoning
    "ClovaStudioRAGReasoning",
    "convert_langchain_tool_to_rag_reasoning",
    "convert_langchain_tools_to_rag_reasoning",
    # Schema Registry
    "SchemaRegistry",
    "VectorDBSchema",
    "DataSourceInfo",
    "get_schema_registry",
    "get_data_source_context",
]
