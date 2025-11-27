"""
Retriever 설정 모듈

이 모듈은 RAG 시스템의 검색(retrieval) 기능 관련 설정을 관리합니다.
- RetrieverSettings: 기본 검색 설정
- MultiQuerySettings: 다중 쿼리 검색 설정
- AdvancedHybridSettings: 고급 하이브리드 검색 설정
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class RetrieverSettings(BaseSettings):
    """
    Retriever 기본 설정
    
    환경변수 prefix: RETRIEVER_
    예: RETRIEVER_DEFAULT_K=10
    """
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVER_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    default_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="검색 시 반환할 문서 수"
    )
    default_sparse_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sparse 검색기 가중치 (0.0 ~ 1.0)"
    )
    default_dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Dense 검색기 가중치 (0.0 ~ 1.0)"
    )
    default_rrf_c: int = Field(
        default=60,
        ge=1,
        description="RRF(Reciprocal Rank Fusion) 상수"
    )
    bm25_index_path: str = Field(
        default="sparse_index/unified_bm25",
        description="저장된 BM25 인덱스 경로 (프로젝트 루트 기준)"
    )


class MultiQuerySettings(BaseSettings):
    """
    MultiQuery Retriever 설정
    
    환경변수 prefix: MULTI_QUERY_
    예: MULTI_QUERY_NUM_QUERIES=4
    """
    model_config = SettingsConfigDict(
        env_prefix="MULTI_QUERY_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    num_queries: int = Field(
        default=4,
        ge=1,
        le=10,
        description="LLM으로 생성할 쿼리 개수"
    )
    default_strategy: Literal["rrf", "max", "sum"] = Field(
        default="rrf",
        description="결과 병합 전략"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="MultiQuery RRF 상수"
    )
    include_original: bool = Field(
        default=True,
        description="원본 쿼리를 생성된 쿼리 목록에 포함할지 여부"
    )


class AdvancedHybridSettings(BaseSettings):
    """
    Advanced Hybrid (Final Hybrid) 설정
    
    환경변수 prefix: ADVANCED_HYBRID_
    예: ADVANCED_HYBRID_BASE_WEIGHT=0.4
    """
    model_config = SettingsConfigDict(
        env_prefix="ADVANCED_HYBRID_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    base_hybrid_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Base Hybrid 가중치 (Final Hybrid 구성 시)"
    )
    multi_query_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="MultiQuery 가중치 (Final Hybrid 구성 시)"
    )


__all__ = [
    "RetrieverSettings",
    "MultiQuerySettings",
    "AdvancedHybridSettings",
]

