"""
Naver Cloud Clova Studio API 설정 모듈

이 모듈은 Naver Cloud Clova Studio의 다양한 AI 서비스 설정을 관리합니다.
- Embeddings: 텍스트 임베딩 생성
- Chat: Chat Completions V3 (Thinking, Function Calling, Structured Output 지원)
- Segmentation: 문서 분할
- Summarization: 문서 요약
- RAG Reasoning: RAG 기반 추론
- Reranker: 검색 결과 재순위화
- OpenAI Compatible: OpenAI API 호환 인터페이스
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class NaverCloudEmbeddingsSettings(BaseSettings):
    """
    Naver Cloud 임베딩 서비스 설정
    
    환경변수 prefix: NAVER_CLOUD_EMBEDDINGS_
    예: NAVER_CLOUD_EMBEDDINGS_MODEL_URL=https://...
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_EMBEDDINGS_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    model_url: str | None = Field(
        default=None,
        description="BGE-M3 임베딩 서비스의 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="서비스 API 키"
    )


class NaverCloudChatSettings(BaseSettings):
    """
    Naver Cloud Clova Studio Chat Completions V3 설정
    
    환경변수 prefix: NAVER_CLOUD_CHAT_
    예: NAVER_CLOUD_CHAT_ENDPOINT=https://...
    
    이 설정은 Chat Completions V3 API의 기본 파라미터를 관리합니다.
    Thinking 모드, Function Calling, Structured Output 등을 지원합니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_CHAT_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    endpoint: str | None = Field(
        default=None,
        description="Chat Completions V3 API 엔드포인트 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    model_name: str = Field(
        default="HCX-003",
        description="사용할 모델명 (예: HCX-003, HCX-DASH-001 등)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="응답의 창의성 조절 (0.0: 결정적, 1.0: 창의적)"
    )
    max_tokens: int = Field(
        default=1024,
        ge=-1,
        le=8192,
        description="생성할 최대 토큰 수 (-1이면 무제한)"
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Top-K 샘플링 (0이면 비활성화)"
    )
    top_p: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Top-P (nucleus) 샘플링"
    )
    repeat_penalty: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="반복 페널티 (1.0: 페널티 없음, 최대 10.0)"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )
    enabled: bool = Field(
        default=False,
        description="Chat API 기능 활성화 여부"
    )


class NaverCloudSegmentationSettings(BaseSettings):
    """
    Naver Cloud Clova Studio Segmentation API 설정
    
    환경변수 prefix: NAVER_CLOUD_SEGMENTATION_
    예: NAVER_CLOUD_SEGMENTATION_ENDPOINT=https://...
    
    문서를 의미 단위로 분할하는 API 설정입니다.
    Contextual Chunking 구현 시 활용할 수 있습니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_SEGMENTATION_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    endpoint: str | None = Field(
        default=None,
        description="Segmentation API 엔드포인트 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    default_alpha: int = Field(
        default=-1,
        ge=-1,
        le=100,
        description="하이브리드 분할 파라미터 (-1: auto, 0-100: 수동 설정)"
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )
    enabled: bool = Field(
        default=False,
        description="Segmentation API 기능 활성화 여부"
    )


class NaverCloudSummarizationSettings(BaseSettings):
    """
    Naver Cloud Clova Studio Summarization API 설정
    
    환경변수 prefix: NAVER_CLOUD_SUMMARIZATION_
    예: NAVER_CLOUD_SUMMARIZATION_ENDPOINT=https://...
    
    문서를 요약하는 API 설정입니다.
    Contextual Chunking 시 컨텍스트 생성에 활용할 수 있습니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_SUMMARIZATION_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    endpoint: str | None = Field(
        default=None,
        description="Summarization API 엔드포인트 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    default_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="기본 요약 길이 (short: 짧게, medium: 보통, long: 길게)"
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )
    enabled: bool = Field(
        default=False,
        description="Summarization API 기능 활성화 여부"
    )


class NaverCloudRAGReasoningSettings(BaseSettings):
    """
    Naver Cloud Clova Studio RAG Reasoning API 설정
    
    환경변수 prefix: NAVER_CLOUD_RAG_REASONING_
    예: NAVER_CLOUD_RAG_REASONING_ENDPOINT=https://...
    
    RAG 기반 추론을 수행하는 API 설정입니다.
    검색된 문서를 기반으로 고급 추론을 수행할 수 있습니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_RAG_REASONING_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    endpoint: str | None = Field(
        default=None,
        description="RAG Reasoning API 엔드포인트 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )
    enabled: bool = Field(
        default=False,
        description="RAG Reasoning API 기능 활성화 여부"
    )


class NaverCloudRerankerSettings(BaseSettings):
    """
    Naver Cloud Clova Studio Reranker 설정
    
    환경변수 prefix: NAVER_CLOUD_RERANKER_
    예: NAVER_CLOUD_RERANKER_ENDPOINT=https://...
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_RERANKER_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    endpoint: str | None = Field(
        default=None,
        description="Clova Studio Reranker API 엔드포인트 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )
    default_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="top_k 미지정 시 기본 반환 문서 수"
    )
    enabled: bool = Field(
        default=False,
        description="Reranker 기능 활성화 여부"
    )
    id_key: str | None = Field(
        default=None,
        description="문서 고유 ID로 사용할 메타데이터 키 (None이면 해시 사용)"
    )


class NaverCloudOpenAICompatibleSettings(BaseSettings):
    """
    Naver Cloud Clova Studio OpenAI Compatible API 설정
    
    환경변수 prefix: NAVER_CLOUD_OPENAI_COMPATIBLE_
    예: NAVER_CLOUD_OPENAI_COMPATIBLE_BASE_URL=https://...
    
    OpenAI API와 호환되는 엔드포인트 설정입니다.
    LangChain의 OpenAI 클라이언트를 그대로 사용하여 Clova Studio 모델을 호출할 수 있습니다.
    """
    model_config = SettingsConfigDict(
        env_prefix="NAVER_CLOUD_OPENAI_COMPATIBLE_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    base_url: str | None = Field(
        default=None,
        description="OpenAI 호환 베이스 URL"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-CLOVASTUDIO-API-KEY 헤더 값"
    )
    api_gateway_key: SecretStr | None = Field(
        default=None,
        description="X-NCP-APIGW-API-KEY 헤더 값 (선택적)"
    )
    default_model: str = Field(
        default="HCX-007",
        description="기본 모델명"
    )
    enabled: bool = Field(
        default=False,
        description="OpenAI Compatible API 기능 활성화 여부"
    )


__all__ = [
    "NaverCloudEmbeddingsSettings",
    "NaverCloudChatSettings",
    "NaverCloudSegmentationSettings",
    "NaverCloudSummarizationSettings",
    "NaverCloudRAGReasoningSettings",
    "NaverCloudRerankerSettings",
    "NaverCloudOpenAICompatibleSettings",
]

