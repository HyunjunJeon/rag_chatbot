"""
Adaptive RAG용 질의 분석 및 다중 쿼리 생성 에이전트 구현.

질의 품질을 평가하고 검색 최적화를 위한 다중 쿼리를 생성합니다.

Gemini 3.1 Pro는 with_structured_output()을 네이티브로 지원하므로,
별도의 ParserWrapper 없이 직접 Pydantic 모델을 반환받습니다.

Version History:
    - v1.0: 통합 프롬프트 (query_analysis.yaml) 사용
    - v2.0: 분리된 프롬프트 지원 추가 (query_quality_analysis.yaml + query_expansion.yaml)
    - v3.0: Gemini with_structured_output() 직접 사용 (ParserWrapper 제거)
"""

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.prompts import get_prompt


# =============================================================================
# 분리된 프롬프트용 모델
# =============================================================================


class QueryQualityResult(BaseModel):
    """
    질의 품질 평가 결과를 담는 모델입니다.
    query_quality_analysis.yaml 프롬프트의 출력 형식입니다.
    """

    clarity_score: float = Field(description="Clarity score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    specificity_score: float = Field(description="Specificity score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    searchability_score: float = Field(
        description="Searchability score (0.0 ~ 1.0)", ge=0.0, le=1.0
    )
    issues: list[str] = Field(description="Identified issues", default_factory=list)
    recommendations: list[str] = Field(
        description="Recommendations for improvement", default_factory=list
    )


class QueryExpansionResult(BaseModel):
    """
    쿼리 확장 및 필터 추출 결과를 담는 모델입니다.
    query_expansion.yaml 프롬프트의 출력 형식입니다.
    """

    improved_queries: list[str] = Field(
        description="List of diverse search queries for comprehensive retrieval",
        default_factory=list,
    )
    retrieval_filters: "QueryRetrievalFilters" = Field(
        default_factory=lambda: QueryRetrievalFilters(),
        description="Metadata-based retrieval filters extracted from the question",
    )


# =============================================================================
# 기존 통합 모델 (하위 호환성 유지)
# =============================================================================


class QueryRetrievalFilters(BaseModel):
    """
    질문에서 추출한 검색 필터입니다.

    질문 내용에서 명시적으로 언급된 도메인 힌트만 추출합니다.
    언급되지 않은 필드는 null로 남겨 전체 검색을 허용합니다.
    """

    doc_type: list[str] | None = Field(
        default=None,
        description="Document types to search: 'slack_qa', 'pdf', 'notebook', 'mission'. Extract only if explicitly mentioned (e.g., 'Slack에서', '강의자료에서', '미션에서').",
    )
    course: list[str] | None = Field(
        default=None,
        description="Course names to search (list format). For ambiguous terms, include all matching courses. "
        "Example: 'CV' → ['CV 이론', 'level2_cv', 'Computer Vision']",
    )
    course_topic: list[str] | None = Field(
        default=None,
        description="Specific topic(s) within a course if mentioned (e.g., 'PyTorch', 'Transformer', 'CNN'). Always provide as a list.",
    )
    generation: str | None = Field(
        default=None, description="Bootcamp generation if mentioned (e.g., '1기', '2기', '3기')."
    )
    filter_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the extracted filters (0.0 ~ 1.0). "
        "Set below 0.5 when the query is highly ambiguous and multiple interpretations are equally valid. "
        "Example: 'CV 관련 질문' with no context → confidence 0.3",
    )

    @field_validator("doc_type", "course", "course_topic", mode="before")
    @classmethod
    def normalize_list_fields(cls, v):
        """단일 문자열을 리스트로 변환합니다."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v


class QueryAnalysis(BaseModel):
    """
    질의 품질 평가 및 다중 검색 쿼리 생성 결과를 담는 통합 모델입니다.

    속성:
        clarity_score: 질의의 명확도 (0.0 ~ 1.0)
        specificity_score: 질의의 구체성 (0.0 ~ 1.0)
        searchability_score: 검색 친화도 (0.0 ~ 1.0)
        improved_queries: 다양한 관점의 검색 쿼리 목록 (Multi-Query)
        issues: 원본 질의의 문제점
        recommendations: 개선 권장 사항
        retrieval_filters: 질문에서 추출한 메타데이터 기반 검색 필터
    """

    clarity_score: float = Field(description="Clarity score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    specificity_score: float = Field(description="Specificity score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    searchability_score: float = Field(
        description="Searchability score (0.0 ~ 1.0)", ge=0.0, le=1.0
    )
    improved_queries: list[str] = Field(
        description="List of diverse search queries for comprehensive retrieval (Multi-Query)",
        default_factory=list,
    )
    issues: list[str] = Field(description="Identified issues", default_factory=list)
    recommendations: list[str] = Field(
        description="Recommendations for improvement", default_factory=list
    )
    retrieval_filters: QueryRetrievalFilters = Field(
        default_factory=QueryRetrievalFilters,
        description="Metadata-based retrieval filters extracted from the question",
    )


def analyze_query(
    question: str,
    intent: str,
    llm: Runnable,
    data_source_context: str | None = None,
) -> QueryAnalysis:
    """
    사용자 질의를 분석하고 다중 검색 쿼리를 생성합니다.

    매개변수:
        question: 분석할 사용자 질문
        intent: 분류된 질문 의도
        llm: 분석에 사용할 언어 모델
        data_source_context: VectorDB 데이터 소스 정보 (프롬프트에 주입)

    반환값:
        QueryAnalysis 결과
    """
    try:
        prompt_template: ChatPromptTemplate = get_prompt("query_analysis", return_type="template")
        system_prompt = (
            prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        )

        if data_source_context:
            system_prompt = system_prompt.replace("{data_source_context}", data_source_context)
        else:
            system_prompt = system_prompt.replace(
                "{data_source_context}",
                "## Available Data Sources\n데이터 소스 정보를 사용할 수 없습니다.\n"
                "일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission",
            )

        full_prompt = f"{system_prompt}\n\nquestion: {question}\nintent: {intent}"

        structured_llm = llm.with_structured_output(QueryAnalysis)
        result = structured_llm.invoke(full_prompt)

        if isinstance(result, QueryAnalysis):
            return result

        logger.warning(f"Unexpected result type: {type(result)}")
        return _default_query_analysis(question)

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return _default_query_analysis(question, error=str(e))


async def aanalyze_query(
    question: str,
    intent: str,
    llm: Runnable,
    data_source_context: str | None = None,
) -> QueryAnalysis:
    """
    사용자 질의를 비동기로 분석하고 다중 검색 쿼리를 생성합니다.

    매개변수:
        question: 분석할 사용자 질문
        intent: 분류된 질문 의도
        llm: 분석에 사용할 언어 모델
        data_source_context: VectorDB 데이터 소스 정보 (프롬프트에 주입)

    반환값:
        QueryAnalysis 결과
    """
    try:
        prompt_template: ChatPromptTemplate = get_prompt("query_analysis")
        system_prompt = (
            prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        )

        if data_source_context:
            system_prompt = system_prompt.replace("{data_source_context}", data_source_context)
        else:
            system_prompt = system_prompt.replace(
                "{data_source_context}",
                "## Available Data Sources\n데이터 소스 정보를 사용할 수 없습니다.\n"
                "일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission",
            )

        full_prompt = f"{system_prompt}\n\nquestion: {question}\nintent: {intent}"

        structured_llm = llm.with_structured_output(QueryAnalysis)
        result = await structured_llm.ainvoke(full_prompt)

        if isinstance(result, QueryAnalysis):
            return result

        logger.warning(f"Unexpected result type: {type(result)}")
        return _default_query_analysis(question)

    except Exception as e:
        logger.error(f"Async query analysis failed: {e}")
        return _default_query_analysis(question, error=str(e))


def _default_query_analysis(question: str, error: str | None = None) -> QueryAnalysis:
    """기본 QueryAnalysis 결과를 반환합니다."""
    issues = ["Unable to analyze query"]
    if error:
        issues.append(f"Error: {error}")

    return QueryAnalysis(
        clarity_score=0.5,
        specificity_score=0.5,
        searchability_score=0.5,
        improved_queries=[question],
        issues=issues,
        recommendations=["Use the original query"],
    )


# =============================================================================
# 분리된 프롬프트 지원 함수
# =============================================================================


async def aanalyze_query_split(
    question: str,
    intent: str,
    llm: Runnable,
    data_source_context: str | None = None,
) -> QueryAnalysis:
    """
    분리된 프롬프트를 사용하여 질의를 분석합니다.

    품질 평가와 쿼리 확장을 두 단계로 나누어 처리합니다:
    1. query_quality_analysis.yaml: 품질 평가만 수행
    2. query_expansion.yaml: 품질 점수를 기반으로 쿼리 확장 및 필터 추출

    매개변수:
        question: 분석할 사용자 질문
        intent: 분류된 질문 의도
        llm: 분석에 사용할 언어 모델
        data_source_context: VectorDB 데이터 소스 정보

    반환값:
        QueryAnalysis 결과 (기존 통합 모델과 동일한 형식)
    """
    try:
        quality_result = await _analyze_quality(question, intent, llm)
        logger.debug(
            f"Quality analysis: clarity={quality_result.clarity_score}, specificity={quality_result.specificity_score}"
        )

        expansion_result = await _expand_query(
            question=question,
            intent=intent,
            llm=llm,
            clarity=quality_result.clarity_score,
            specificity=quality_result.specificity_score,
            data_source_context=data_source_context,
        )

        return _merge_results(quality_result, expansion_result)

    except Exception as e:
        logger.error(f"Split query analysis failed: {e}")
        logger.info("Falling back to combined prompt")
        return await aanalyze_query(question, intent, llm, data_source_context)


async def _analyze_quality(
    question: str,
    intent: str,
    llm: Runnable,
) -> QueryQualityResult:
    """품질 평가만 수행합니다 (query_quality_analysis.yaml 사용)."""
    prompt_template: ChatPromptTemplate = get_prompt("query_quality_analysis")
    system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

    full_prompt = f"{system_prompt}\n\nQuestion: {question}\nIntent: {intent}\n\nAnalyze the quality of this question."

    structured_llm = llm.with_structured_output(QueryQualityResult)
    result = await structured_llm.ainvoke(full_prompt)

    if isinstance(result, QueryQualityResult):
        return result

    return QueryQualityResult(
        clarity_score=0.5,
        specificity_score=0.5,
        searchability_score=0.5,
        issues=["Unable to analyze quality"],
        recommendations=["Use the original query"],
    )


async def _expand_query(
    question: str,
    intent: str,
    llm: Runnable,
    clarity: float,
    specificity: float,
    data_source_context: str | None = None,
) -> QueryExpansionResult:
    """쿼리 확장 및 필터 추출을 수행합니다 (query_expansion.yaml 사용)."""
    prompt_template: ChatPromptTemplate = get_prompt("query_expansion")
    system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

    if data_source_context:
        system_prompt = system_prompt.replace("{data_source_context}", data_source_context)
    else:
        system_prompt = system_prompt.replace(
            "{data_source_context}",
            "## Available Data Sources\n데이터 소스 정보를 사용할 수 없습니다.\n"
            "일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission",
        )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Question: {question}\n"
        f"Intent: {intent}\n"
        f"Quality Scores: clarity={clarity}, specificity={specificity}\n\n"
        f"Generate diverse search queries and extract retrieval filters."
    )

    structured_llm = llm.with_structured_output(QueryExpansionResult)
    result = await structured_llm.ainvoke(full_prompt)

    if isinstance(result, QueryExpansionResult):
        return result

    return QueryExpansionResult(
        improved_queries=[question],
        retrieval_filters=QueryRetrievalFilters(),
    )


def _merge_results(
    quality: QueryQualityResult,
    expansion: QueryExpansionResult,
) -> QueryAnalysis:
    """품질 평가 결과와 쿼리 확장 결과를 통합 모델로 병합합니다."""
    return QueryAnalysis(
        clarity_score=quality.clarity_score,
        specificity_score=quality.specificity_score,
        searchability_score=quality.searchability_score,
        improved_queries=expansion.improved_queries,
        issues=quality.issues,
        recommendations=quality.recommendations,
        retrieval_filters=expansion.retrieval_filters,
    )
