"""
Adaptive RAG용 질의 분석 및 다중 쿼리 생성 에이전트 구현.

질의 품질을 평가하고 검색 최적화를 위한 다중 쿼리를 생성합니다.

Note:
    CLOVA HCX-007은 tools와 reasoning을 동시에 지원하지 않으므로,
    with_structured_output() 패턴을 사용하여 구조화된 출력을 생성합니다.

Version History:
    - v1.0: 통합 프롬프트 (query_analysis.yaml) 사용
    - v2.0: 분리된 프롬프트 지원 추가 (query_quality_analysis.yaml + query_expansion.yaml)
"""

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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
    searchability_score: float = Field(description="Searchability score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    issues: list[str] = Field(description="Identified issues", default_factory=list)
    recommendations: list[str] = Field(description="Recommendations for improvement", default_factory=list)


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
    course_topic: list[str] | str | None = Field(
        default=None,
        description="Specific topic(s) within a course if mentioned (e.g., 'PyTorch', 'Transformer', 'CNN'). Can be a single string or a list of strings.",
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
    searchability_score: float = Field(description="Searchability score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    improved_queries: list[str] = Field(
        description="List of diverse search queries for comprehensive retrieval (Multi-Query)",
        default_factory=list,
    )
    issues: list[str] = Field(description="Identified issues", default_factory=list)
    recommendations: list[str] = Field(description="Recommendations for improvement", default_factory=list)
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

    예시:
        >>> from naver_connect_chatbot.config import get_chat_model
        >>> llm = get_chat_model()
        >>> result = analyze_query("What is PyTorch?", "SIMPLE_QA", llm)
        >>> print(result.improved_queries)
        ['PyTorch 개요', 'PyTorch 딥러닝 프레임워크', ...]
    """
    try:
        # 프롬프트 템플릿 로드
        prompt_template: ChatPromptTemplate = get_prompt("query_analysis", "template")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

        # 데이터 소스 컨텍스트 주입
        if data_source_context:
            system_prompt = system_prompt.replace("{data_source_context}", data_source_context)
        else:
            system_prompt = system_prompt.replace(
                "{data_source_context}",
                "## Available Data Sources\n데이터 소스 정보를 사용할 수 없습니다.\n"
                "일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission",
            )

        # 전체 프롬프트 구성
        full_prompt = f"{system_prompt}\n\nquestion: {question}\nintent: {intent}"

        # with_structured_output 사용하여 LLM 직접 호출
        structured_llm = _get_structured_llm(llm, QueryAnalysis)
        result = structured_llm.invoke(full_prompt)

        if isinstance(result, QueryAnalysis):
            return result

        # fallback
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
        # 프롬프트 템플릿 로드
        prompt_template: ChatPromptTemplate = get_prompt("query_analysis")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

        # 데이터 소스 컨텍스트 주입
        if data_source_context:
            system_prompt = system_prompt.replace("{data_source_context}", data_source_context)
        else:
            system_prompt = system_prompt.replace(
                "{data_source_context}",
                "## Available Data Sources\n데이터 소스 정보를 사용할 수 없습니다.\n"
                "일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission",
            )

        # 전체 프롬프트 구성
        full_prompt = f"{system_prompt}\n\nquestion: {question}\nintent: {intent}"

        # with_structured_output 사용하여 LLM 직접 호출
        structured_llm = _get_structured_llm(llm, QueryAnalysis)
        result = await structured_llm.ainvoke(full_prompt)

        if isinstance(result, QueryAnalysis):
            return result

        # fallback
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


def _get_structured_llm(llm: Runnable, schema: type[BaseModel]) -> Runnable:
    """
    LLM에 structured output을 적용합니다.

    CLOVA HCX-007 API는 OpenAI의 tool calling/function calling 파라미터를
    완전히 지원하지 않으므로, 항상 PydanticOutputParser를 사용합니다.

    매개변수:
        llm: 언어 모델
        schema: 출력 스키마 (Pydantic 모델)

    반환값:
        구조화된 출력을 생성하는 Runnable

    Note:
        with_structured_output()은 CLOVA API에서 parallel_tool_calls 에러를
        발생시키므로 사용하지 않습니다. 대신 프롬프트에서 JSON 형식을 요청하고
        PydanticOutputParser로 파싱합니다.
    """
    from langchain_core.output_parsers import JsonOutputParser
    import json
    import re

    parser = JsonOutputParser(pydantic_object=schema)

    # 파서와 함께 호출하는 래퍼 반환
    class ParserWrapper:
        def __init__(self, llm, parser, schema):
            self._llm = llm
            self._parser = parser
            self._schema = schema

        def _extract_json(self, content: str) -> str:
            """응답에서 JSON 부분만 추출합니다."""
            # 코드펜스 내 JSON 추출 (중첩 객체 포함)
            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
            if json_match:
                return json_match.group(1)

            # 직접 JSON 객체 추출 (중첩 객체 포함)
            # 가장 바깥쪽 중괄호를 찾음
            start_idx = content.find("{")
            if start_idx == -1:
                return content

            # 중괄호 카운팅으로 JSON 끝 찾기
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if brace_count == 0:
                return content[start_idx : end_idx + 1]

            return content

        def _parse_with_fallback(self, content: str):
            """파싱 시도 후 실패 시 fallback 반환."""
            try:
                json_str = self._extract_json(content)
                # JsonOutputParser는 dict를 반환하므로 Pydantic 모델로 변환
                parsed = self._parser.parse(json_str)
                if isinstance(parsed, dict):
                    return self._schema(**parsed)
                return parsed
            except Exception as parse_error:
                logger.warning(f"JSON parsing failed: {parse_error}, trying direct parse")
                try:
                    # 직접 JSON 파싱 시도
                    data = json.loads(self._extract_json(content))
                    return self._schema(**data)
                except Exception:
                    logger.error(f"All parsing attempts failed for content: {content[:300]}")
                    raise

        def invoke(self, prompt):
            result = self._llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            return self._parse_with_fallback(content)

        async def ainvoke(self, prompt):
            result = await self._llm.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            return self._parse_with_fallback(content)

    return ParserWrapper(llm, parser, schema)


# # Deprecated: 이전 버전과의 호환성을 위해 유지
# def create_query_analyzer(llm: Runnable, data_source_context: str | None = None):
#     """
#     Deprecated: analyze_query() 또는 aanalyze_query()를 직접 사용하세요.

#     이전 버전 호환성을 위한 래퍼 클래스를 반환합니다.
#     """
#     logger.warning("create_query_analyzer() is deprecated. Use analyze_query() or aanalyze_query() directly.")

#     class QueryAnalyzerWrapper:
#         def __init__(self, llm, data_source_context):
#             self._llm = llm
#             self._data_source_context = data_source_context

#         def invoke(self, input_dict):
#             question = input_dict.get("question", "")
#             intent = input_dict.get("intent", "SIMPLE_QA")
#             return analyze_query(question, intent, self._llm, self._data_source_context)

#         async def ainvoke(self, input_dict):
#             question = input_dict.get("question", "")
#             intent = input_dict.get("intent", "SIMPLE_QA")
#             return await aanalyze_query(question, intent, self._llm, self._data_source_context)

#     return QueryAnalyzerWrapper(llm, data_source_context)


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

    사용 시기:
        - 품질 평가와 쿼리 확장을 독립적으로 최적화하고 싶을 때
        - 각 단계의 프롬프트를 개별적으로 튜닝하고 싶을 때
        - 디버깅 시 각 단계의 출력을 확인하고 싶을 때
    """
    try:
        # Step 1: 품질 평가
        quality_result = await _analyze_quality(question, intent, llm)
        logger.debug(
            f"Quality analysis: clarity={quality_result.clarity_score}, specificity={quality_result.specificity_score}"
        )

        # Step 2: 쿼리 확장 (품질 점수 기반)
        expansion_result = await _expand_query(
            question=question,
            intent=intent,
            llm=llm,
            clarity=quality_result.clarity_score,
            specificity=quality_result.specificity_score,
            data_source_context=data_source_context,
        )

        # 결과 병합
        return _merge_results(quality_result, expansion_result)

    except Exception as e:
        logger.error(f"Split query analysis failed: {e}")
        # 실패 시 기존 통합 프롬프트로 폴백
        logger.info("Falling back to combined prompt")
        return await aanalyze_query(question, intent, llm, data_source_context)


async def _analyze_quality(
    question: str,
    intent: str,
    llm: Runnable,
) -> QueryQualityResult:
    """
    품질 평가만 수행합니다 (query_quality_analysis.yaml 사용).
    """
    prompt_template: ChatPromptTemplate = get_prompt("query_quality_analysis")
    system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

    full_prompt = f"{system_prompt}\n\nQuestion: {question}\nIntent: {intent}\n\nAnalyze the quality of this question."

    structured_llm = _get_structured_llm(llm, QueryQualityResult)
    result = await structured_llm.ainvoke(full_prompt)

    if isinstance(result, QueryQualityResult):
        return result

    # fallback
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
    """
    쿼리 확장 및 필터 추출을 수행합니다 (query_expansion.yaml 사용).
    """
    prompt_template: ChatPromptTemplate = get_prompt("query_expansion")
    system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""

    # 데이터 소스 컨텍스트 주입
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

    structured_llm = _get_structured_llm(llm, QueryExpansionResult)
    result = await structured_llm.ainvoke(full_prompt)

    if isinstance(result, QueryExpansionResult):
        return result

    # fallback
    return QueryExpansionResult(
        improved_queries=[question],
        retrieval_filters=QueryRetrievalFilters(),
    )


def _merge_results(
    quality: QueryQualityResult,
    expansion: QueryExpansionResult,
) -> QueryAnalysis:
    """
    품질 평가 결과와 쿼리 확장 결과를 통합 모델로 병합합니다.
    """
    return QueryAnalysis(
        clarity_score=quality.clarity_score,
        specificity_score=quality.specificity_score,
        searchability_score=quality.searchability_score,
        improved_queries=expansion.improved_queries,
        issues=quality.issues,
        recommendations=quality.recommendations,
        retrieval_filters=expansion.retrieval_filters,
    )
