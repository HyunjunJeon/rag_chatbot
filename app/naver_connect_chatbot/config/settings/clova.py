"""
Clova X LLM 및 Embeddings 설정 모듈

이 모듈은 langchain_naver 패키지의 ChatClovaX와 ClovaXEmbeddings를 위한
설정 클래스를 제공합니다.

환경변수 예시:
    CLOVASTUDIO_API_KEY=your-api-key
    CLOVA_MODEL=HCX-007
    CLOVA_EMBEDDINGS_MODEL=bge-m3
    CLOVA_TEMPERATURE=0.7
    CLOVA_MAX_TOKENS=
    CLOVA_THINKING_EFFORT=low
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class ClovaXLLMSettings(BaseSettings):
    """
    Clova X LLM 설정 (langchain_naver.ChatClovaX)
    
    환경변수 prefix: CLOVA_
    예: CLOVA_MODEL=HCX-007
    
    langchain_naver의 ChatClovaX는 다음 파라미터를 사용합니다:
    - api_key: CLOVASTUDIO_API_KEY 환경변수에서 자동 로드
    - model: 모델명 (기본값: HCX-005, 권장: HCX-007)
    - temperature: 응답의 창의성 조절
    - max_tokens: 생성할 최대 토큰 수
    - thinking: Thinking 모드 설정 (예: {"effort": "low"})
    """
    
    model_config = SettingsConfigDict(
        env_prefix="CLOVA_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )
    
    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Clova Studio API 키"
    )
    model: str = Field(
        default="HCX-007",
        description="사용할 Clova X 모델명 (HCX-003, HCX-007 등)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="응답의 창의성 조절 (0.0: 결정적, 1.0: 창의적)"
    )
    max_tokens: int | None = Field(
        default=None,
        description="생성할 최대 토큰 수 (None이면 제한 없음)"
    )
    thinking_effort: str | None = Field(
        default=None,
        description="Thinking 모드 effort: 'none', 'low', 'medium', 'high' 또는 None"
    )
    
    @field_validator("thinking_effort")
    @classmethod
    def validate_thinking_effort(cls, v: str | None) -> str | None:
        """Thinking effort 값 검증"""
        if v is None:
            return v
        
        valid_values = {None, "none", "low", "medium", "high"}
        if v.lower() not in valid_values:
            raise ValueError(
                f"thinking_effort는 {valid_values} 중 하나여야 합니다. 입력값: {v}"
            )
        if v is None:
            return "none"
        return v.lower()


class ClovaXEmbeddingsSettings(BaseSettings):
    """
    Clova X Embeddings 설정 (langchain_naver.ClovaXEmbeddings)
    
    환경변수 prefix: CLOVA_
    예: CLOVA_EMBEDDINGS_MODEL=bge-m3
    
    langchain_naver의 ClovaXEmbeddings는 다음 파라미터를 사용합니다:
    - api_key: CLOVASTUDIO_API_KEY 환경변수에서 자동 로드
    - model: 임베딩 모델명 (기본값: clir-emb-dolphin, 지원: bge-m3)
    """
    
    model_config = SettingsConfigDict(
        env_prefix="CLOVA_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )
    
    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Clova Studio API 키"
    )
    embeddings_model: str = Field(
        default="bge-m3",
        description="임베딩 모델명 (clir-emb-dolphin, bge-m3 등)"
    )


class ClovaStudioRerankerSettings(BaseSettings):
    """
    Clova Studio Reranker API 설정
    
    환경변수 prefix: CLOVASTUDIO_RERANKER_
    예: CLOVASTUDIO_RERANKER_ENDPOINT=https://clovastudio.stream.ntruss.com/v1/api-tools/reranker
    
    API 스펙:
    - 요청: POST /v1/api-tools/reranker
    - 헤더: Authorization: Bearer {CLOVASTUDIO_API_KEY}
    - 바디: {"query": str, "documents": [{"id": str, "doc": str}], "maxTokens": int}
    - 최대 입력 토큰: 128,000
    - 최대 출력 토큰: 4,096
    """
    model_config = SettingsConfigDict(
        env_prefix="CLOVASTUDIO_RERANKER_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Authorization Bearer 토큰 (LLM/Embeddings와 공유)"
    )
    endpoint: str | None = Field(
        default="https://clovastudio.stream.ntruss.com/v1/api-tools/reranker",
        description="Reranker API 엔드포인트 URL"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=4096,
        description="최대 생성 토큰 수 (기본값: 4096, 최대: 4096)"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초) 기본 60초, 최대 300초"
    )


class ClovaStudioSegmentationSettings(BaseSettings):
    """
    Clova Studio Segmentation API 설정
    
    환경변수 prefix: CLOVASTUDIO_SEGMENTATION_
    예: CLOVASTUDIO_SEGMENTATION_ENDPOINT=https://clovastudio.stream.ntruss.com/v1/api-tools/segmentation
    
    API 스펙:
    - 요청: POST /v1/api-tools/segmentation
    - 헤더: Authorization: Bearer {CLOVASTUDIO_API_KEY}
    - 바디: {"text": str, "alpha": float, "segCnt": int, "postProcess": bool, ...}
    - 최대 입력: ~120,000자 (한글 기준, 공백 포함)
    - 문장 간 유사도를 파악하여 주제 단위로 글의 단락을 구분
    """
    model_config = SettingsConfigDict(
        env_prefix="CLOVASTUDIO_SEGMENTATION_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Authorization Bearer 토큰 (LLM/Embeddings와 공유)"
    )
    endpoint: str = Field(
        default="https://clovastudio.stream.ntruss.com/v1/api-tools/segmentation",
        description="Segmentation API 엔드포인트 URL"
    )
    alpha: float = Field(
        default=-100.0,
        ge=-100.0,
        le=1.5,
        description="문단 나누기 threshold 값. 클수록 나눠지는 문단 수 증가 (범위: -1.5~1.5, -100은 자동)"
    )
    seg_count: int = Field(
        default=-1,
        ge=-1,
        description="원하는 문단 나누기 수 (범위: 1 이상, -1이면 모델이 최적 문단 수로 분리)"
    )
    post_process: bool = Field(
        default=False,
        description="문단 나누기 수행 후 원하는 길이로 문단을 합치거나 나누는 후처리 수행 여부"
    )
    post_process_max_size: int = Field(
        default=1000,
        ge=1,
        description="후처리 적용 시 문단에 포함되는 문자열의 최대 글자 수"
    )
    post_process_min_size: int = Field(
        default=300,
        ge=0,
        description="후처리 적용 시 문단에 포함되는 문자열의 최소 글자 수 (0~maxSize, -1이면 자동)"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )


class ClovaStudioSummarizationSettings(BaseSettings):
    """
    Clova Studio Summarization API 설정
    
    환경변수 prefix: CLOVASTUDIO_SUMMARIZATION_
    예: CLOVASTUDIO_SUMMARIZATION_ENDPOINT=https://clovastudio.stream.ntruss.com/v1/api-tools/summarization/v2
    
    API 스펙:
    - 요청: POST /v1/api-tools/summarization/v2
    - 헤더: Authorization: Bearer {CLOVASTUDIO_API_KEY}
    - 바디: {"texts": list[str], "autoSentenceSplitter": bool, ...}
    - 최대 입력: ~35,000자 (한글 기준, 공백 포함)
    - 긴 텍스트를 짧고 간략하게 요약
    """
    model_config = SettingsConfigDict(
        env_prefix="CLOVASTUDIO_SUMMARIZATION_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Authorization Bearer 토큰 (LLM/Embeddings와 공유)"
    )
    endpoint: str = Field(
        default="https://clovastudio.stream.ntruss.com/v1/api-tools/summarization/v2",
        description="Summarization API 엔드포인트 URL"
    )
    auto_sentence_splitter: bool = Field(
        default=True,
        description="요약할 문장 목록의 문장 분리 허용 여부"
    )
    seg_count: int = Field(
        default=-1,
        ge=-1,
        description="요약할 문장 목록을 분리할 문단 수 (1 이상, -1이면 모델이 최적 문단 수로 분리)"
    )
    seg_max_size: int = Field(
        default=1000,
        ge=1,
        le=3000,
        description="문단 분리 시 한 문단에 포함될 최대 문자열의 글자 수 (1~3000자)"
    )
    seg_min_size: int = Field(
        default=300,
        ge=0,
        description="문단 분리 시 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)"
    )
    include_ai_filters: bool = Field(
        default=False,
        description="AI Filter 적용 여부"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )


class ClovaStudioRAGReasoningSettings(BaseSettings):
    """
    Clova Studio RAG Reasoning API 설정
    
    환경변수 prefix: CLOVASTUDIO_RAG_REASONING_
    예: CLOVASTUDIO_RAG_REASONING_ENDPOINT=https://clovastudio.stream.ntruss.com/v1/api-tools/rag-reasoning
    
    API 스펙:
    - 요청: POST /v1/api-tools/rag-reasoning
    - 헤더: Authorization: Bearer {CLOVASTUDIO_API_KEY}
    - 바디: {"messages": list[dict], "tools": list[dict], "topP": float, ...}
    - 최대 입력 토큰: 128,000
    - 최대 출력 토큰: 4,096
    - Function calling 기반 RAG 추론 모델로 검색 과정을 계획하고 문서 출처를 인용
    
    참고:
    - tools와 toolChoice는 런타임에 동적으로 설정 (설정 클래스에 미포함)
    - messages는 system, user, assistant, tool 역할 지원
    """
    model_config = SettingsConfigDict(
        env_prefix="CLOVASTUDIO_RAG_REASONING_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
        validate_default=False,
    )

    api_key: SecretStr | None = Field(
        default=None,
        alias="clovastudio_api_key",
        description="CLOVASTUDIO_API_KEY - Authorization Bearer 토큰 (LLM/Embeddings와 공유)"
    )
    endpoint: str = Field(
        default="https://clovastudio.stream.ntruss.com/v1/api-tools/rag-reasoning",
        description="RAG Reasoning API 엔드포인트 URL"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="생성 토큰 후보군을 누적 확률을 기반으로 샘플링 (0 < topP ≤ 1)"
    )
    top_k: int = Field(
        default=40,
        ge=0,
        le=128,
        description="생성 토큰 후보군에서 확률이 높은 K개를 후보로 지정하여 샘플링 (0 ≤ topK ≤ 128)"
    )
    max_tokens: int = Field(
        default=4096,
        ge=-1,
        le=4096,
        description="최대 생성 토큰 수 (1024 ≤ maxTokens ≤ 4096, -1이면 최대값 4096 사용)"
    )
    temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="생성 토큰에 대한 다양성 정도 (0.00 < temperature ≤ 1.00)"
    )
    
    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """max_tokens 값 검증 및 변환 (-1이면 최대값 4096으로 설정)"""
        if v == -1:
            return 4096
        if v < 1024 or v > 4096:
            raise ValueError(
                f"max_tokens는 1024~4096 범위여야 합니다 (또는 -1로 최대값 지정). 입력값: {v}"
            )
        return v
    repetition_penalty: float = Field(
        default=1.1,
        ge=0.0,
        le=2.0,
        description="같은 토큰을 생성하는 것에 대한 패널티 정도 (0.0 < repetitionPenalty ≤ 2.0)"
    )
    stop: list[str] = Field(
        default_factory=list,
        description="토큰 생성 중단 문자 리스트 (기본값: [])"
    )
    seed: int = Field(
        default=0,
        ge=0,
        le=4294967295,
        description="모델 반복 실행 시 결괏값의 일관성 수준 조정 (0: 랜덤, 1~4294967295: 고정)"
    )
    include_ai_filters: bool = Field(
        default=True,
        description="AI 필터 결과 표시 여부 (욕설, 비하/차별/혐오, 성희롱/음란 등)"
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP 요청 타임아웃 (초)"
    )


__all__ = [
    "ClovaXLLMSettings",
    "ClovaXEmbeddingsSettings",
    "ClovaStudioRerankerSettings",
    "ClovaStudioSegmentationSettings",
    "ClovaStudioSummarizationSettings",
    "ClovaStudioRAGReasoningSettings",
]

