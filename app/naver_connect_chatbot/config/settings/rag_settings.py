"""
Adaptive RAG 구성을 위한 설정 정의 모듈.

에이전트 활성화 플래그, 재시도 제한, 품질 임계값, 모델 선택 등
Adaptive RAG 시스템 전반에 필요한 설정을 한곳에서 정의합니다.
"""

from pydantic import BaseModel, Field


class AdaptiveRAGSettings(BaseModel):
    """
    Adaptive RAG 시스템을 제어하기 위한 세부 설정 컬렉션.
    
    속성:
        # Agent Activation
        enable_intent_classification: 의도 분류 단계를 사용할지 여부
        enable_query_analysis: 질의 분석 및 정제 단계를 활성화할지 여부
        enable_document_evaluation: 문서 적합성 평가 단계를 활성화할지 여부
        enable_answer_validation: 답변 품질 검증 단계를 활성화할지 여부
        enable_correction: 교정 루프를 활성화할지 여부
        
        # Retry Limits
        max_retrieval_retries: 검색 재시도 허용 횟수 상한
        max_correction_retries: 교정 반복 허용 횟수 상한
        
        # Quality Thresholds
        min_quality_score: 허용 가능한 최소 답변 품질 점수 (0.0 ~ 1.0)
        min_document_relevance: 허용 가능한 최소 문서 관련성 점수
        min_intent_confidence: 의도 분류에 요구되는 최소 신뢰도
        
        # Model Selection
        intent_model: 의도 분류용 경량 모델
        evaluation_model: 문서/답변 평가에 사용할 모델
        main_model: 생성 및 복잡한 작업에 사용할 주 모델
        
        # Timeouts
        agent_timeout: 각 에이전트 동작 타임아웃(초)
        retrieval_timeout: 검색 단계 타임아웃(초)
        
        # Feature Flags
        use_multi_query: 다중 질의 검색 기능 사용 여부
        use_reranking: 재순위화 기능 사용 여부
        use_caching: 결과 캐싱 기능 사용 여부
    """
    
    # Agent Activation
    enable_intent_classification: bool = Field(
        default=True,
        description="Enable intent classification for adaptive strategies"
    )
    enable_query_analysis: bool = Field(
        default=True,
        description="Enable query quality analysis and improvement"
    )
    enable_document_evaluation: bool = Field(
        default=True,
        description="Enable document relevance and sufficiency evaluation"
    )
    enable_answer_validation: bool = Field(
        default=True,
        description="Enable answer validation for hallucinations and quality"
    )
    enable_correction: bool = Field(
        default=True,
        description="Enable correction loop for answer improvement"
    )
    
    # Retry Limits
    max_retrieval_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of retrieval retry attempts"
    )
    max_correction_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of correction iteration attempts"
    )
    
    # Quality Thresholds
    min_quality_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable answer quality score"
    )
    min_document_relevance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable document relevance score"
    )
    min_intent_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for intent classification"
    )
    
    # Model Selection (for cost optimization)
    intent_model: str = Field(
        default="gpt-4o-mini",
        description="Model for intent classification (faster/cheaper)"
    )
    evaluation_model: str = Field(
        default="gpt-4o-mini",
        description="Model for document and answer evaluation"
    )
    main_model: str = Field(
        default="gpt-4o",
        description="Model for answer generation and complex tasks"
    )
    
    # Timeouts (seconds)
    agent_timeout: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Timeout for agent operations in seconds"
    )
    retrieval_timeout: int = Field(
        default=10,
        ge=5,
        le=60,
        description="Timeout for retrieval operations in seconds"
    )
    
    # Feature Flags
    use_multi_query: bool = Field(
        default=True,
        description="Enable multi-query retrieval for complex questions"
    )
    use_reranking: bool = Field(
        default=True,
        description="Enable reranking of retrieved documents"
    )
    use_caching: bool = Field(
        default=False,
        description="Enable caching of intermediate results"
    )
    
    class Config:
        """Pydantic 설정 옵션."""
        env_prefix = "ADAPTIVE_RAG_"
        case_sensitive = False


# Default instance
adaptive_rag_settings = AdaptiveRAGSettings()


def get_adaptive_rag_settings() -> AdaptiveRAGSettings:
    """
    Adaptive RAG 설정 인스턴스를 반환합니다.
    
    반환값:
        AdaptiveRAGSettings: 현재 설정 인스턴스
        
    예시:
        >>> settings = get_adaptive_rag_settings()
        >>> print(settings.enable_intent_classification)
        True
    """
    return adaptive_rag_settings


def update_adaptive_rag_settings(**kwargs) -> None:
    """
    Adaptive RAG 설정 값을 갱신합니다.
    
    매개변수:
        **kwargs: 변경할 설정 키와 값 쌍
        
    예시:
        >>> update_adaptive_rag_settings(
        ...     max_retrieval_retries=3,
        ...     min_quality_score=0.9
        ... )
    """
    global adaptive_rag_settings
    
    for key, value in kwargs.items():
        if hasattr(adaptive_rag_settings, key):
            setattr(adaptive_rag_settings, key, value)

