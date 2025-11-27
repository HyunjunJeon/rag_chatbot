"""
Adaptive RAG용 질의 분석 및 다중 쿼리 생성 에이전트 구현.

질의 품질을 평가하고 검색 최적화를 위한 다중 쿼리를 생성합니다.
"""

import json
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_prompt
from naver_connect_chatbot.config import logger


class QueryRetrievalFilters(BaseModel):
    """
    질문에서 추출한 검색 필터입니다.

    질문 내용에서 명시적으로 언급된 도메인 힌트만 추출합니다.
    언급되지 않은 필드는 null로 남겨 전체 검색을 허용합니다.
    """
    doc_type: list[str] | None = Field(
        default=None,
        description="Document types to search: 'slack_qa', 'pdf', 'notebook', 'mission'. Extract only if explicitly mentioned (e.g., 'Slack에서', '강의자료에서', '미션에서')."
    )
    course: str | None = Field(
        default=None,
        description="Course name filter if explicitly mentioned (e.g., '데이터분석', 'CV', 'NLP')."
    )
    course_topic: str | None = Field(
        default=None,
        description="Specific topic within a course if mentioned (e.g., 'PyTorch', 'Transformer', 'CNN')."
    )
    generation: str | None = Field(
        default=None,
        description="Bootcamp generation if mentioned (e.g., '1기', '2기', '3기')."
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
    clarity_score: float = Field(
        description="Clarity score (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    specificity_score: float = Field(
        description="Specificity score (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    searchability_score: float = Field(
        description="Searchability score (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    improved_queries: list[str] = Field(
        description="List of diverse search queries for comprehensive retrieval (Multi-Query)",
        default_factory=list
    )
    issues: list[str] = Field(
        description="Identified issues",
        default_factory=list
    )
    recommendations: list[str] = Field(
        description="Recommendations for improvement",
        default_factory=list
    )
    retrieval_filters: QueryRetrievalFilters = Field(
        default_factory=QueryRetrievalFilters,
        description="Metadata-based retrieval filters extracted from the question"
    )


@tool(args_schema=QueryAnalysis)
def emit_query_analysis_result(
    clarity_score: Annotated[float, Field(description="Clarity score (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    specificity_score: Annotated[float, Field(description="Specificity score (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    searchability_score: Annotated[float, Field(description="Searchability score (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    improved_queries: Annotated[list[str], Field(description="List of improved query variations")],
    issues: Annotated[list[str], Field(description="Identified issues")],
    recommendations: Annotated[list[str], Field(description="Recommendations for improvement")],
    retrieval_filters: Annotated[dict | None, Field(description="Metadata filters extracted from the question")] = None,
) -> QueryAnalysis:
    """
    Emit structured query analysis results.

    Call this tool after analyzing a query to return the final results as a QueryAnalysis model.
    This ensures consistent structured output via ToolMessage.

    Args:
        clarity_score: Query clarity score (0.0 ~ 1.0)
        specificity_score: Query specificity score (0.0 ~ 1.0)
        searchability_score: Search-friendliness score (0.0 ~ 1.0)
        improved_queries: List of improved query variations
        issues: List of identified issues in the original query
        recommendations: List of recommendations for query improvement
        retrieval_filters: Optional metadata filters (doc_type, course, course_topic, generation)

    Returns:
        QueryAnalysis instance with all analysis results
    """
    # retrieval_filters가 dict인 경우 QueryRetrievalFilters로 변환
    filters = QueryRetrievalFilters()
    if retrieval_filters and isinstance(retrieval_filters, dict):
        filters = QueryRetrievalFilters(**retrieval_filters)

    return QueryAnalysis(
        clarity_score=clarity_score,
        specificity_score=specificity_score,
        searchability_score=searchability_score,
        improved_queries=improved_queries,
        issues=issues,
        recommendations=recommendations,
        retrieval_filters=filters,
    )


def create_query_analyzer(llm: Runnable) -> Any:
    """
    질의 분석 에이전트를 생성합니다.
    
    명확성, 구체성, 검색 친화도를 평가하고 개선 방향을 제안합니다.
    에이전트는 emit_query_analysis 도구를 사용하여 구조화된 결과를
    ToolMessage 형태로 반환합니다.
    
    매개변수:
        llm: 분석에 사용할 언어 모델
    
    반환값:
        질의를 분석하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> analyzer = create_query_analyzer(llm)
        >>> result = analyzer.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "question: What is it?\nintent: CLARIFICATION_NEEDED"
        ...     }]
        ... })
    """
    try:        
        prompt_template = get_prompt("query_analysis")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        
        # 시스템 프롬프트에 도구 사용 가이드 추가
        schema_text = json.dumps(
            QueryAnalysis.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        enhanced_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: After analyzing the query, you MUST call the emit_query_analysis tool "
            "with all required parameters (clarity_score, specificity_score, searchability_score, "
            "improved_queries, issues, recommendations) to return your structured analysis."
            "\n\nReturn outputs that match this JSON schema exactly (no extra fields, no prose):\n"
            f"{schema_text}"
        )
        
        agent = create_agent(
            model=llm,
            tools=[emit_query_analysis_result],
            system_prompt=enhanced_prompt,
            name="query_analyzer",
        )
        
        logger.debug("Query analyzer agent created successfully with emit_query_analysis tool")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create query analyzer: {e}")
        raise


def analyze_query(
    question: str,
    intent: str,
    llm: Runnable
) -> QueryAnalysis:
    """
    단일 질의를 분석하는 편의 함수입니다.
    
    에이전트가 emit_query_analysis 도구를 호출하면 ToolMessage로
    결과가 반환되며, 이를 파싱하여 QueryAnalysis 객체를 추출합니다.
    
    매개변수:
        question: 분석 대상 사용자 질문
        intent: 분류된 질문 의도
        llm: 사용할 언어 모델
    
    반환값:
        QueryAnalysis 결과
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> result = analyze_query("What is it?", "CLARIFICATION_NEEDED", llm)
        >>> print(result.improved_queries)
        ['What is PyTorch?', 'What is the concept being discussed?']
    """
    analyzer = create_query_analyzer(llm)
    response = analyzer.invoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\nintent: {intent}"
        }]
    })
    
    # 에이전트 응답에서 ToolMessage를 찾아 QueryAnalysis를 추출합니다.
    try:
        # response가 dict이고 messages가 있는 경우
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            for msg in reversed(messages):  # 마지막 메시지부터 확인
                # ToolMessage 타입 확인 (클래스명 또는 type 속성으로)
                is_tool_msg = (
                    msg.__class__.__name__ == "ToolMessage" or
                    (hasattr(msg, "type") and msg.type == "tool")
                )
                
                if is_tool_msg:
                    tool_content = msg.content
                    # content가 이미 QueryAnalysis 인스턴스인 경우
                    if isinstance(tool_content, QueryAnalysis):
                        logger.debug("Successfully extracted QueryAnalysis from ToolMessage")
                        return tool_content
                    # content가 dict인 경우 QueryAnalysis로 변환
                    elif isinstance(tool_content, dict):
                        logger.debug("Converting dict content to QueryAnalysis")
                        return QueryAnalysis(**tool_content)
                    # content가 문자열인 경우 (Pydantic repr 형식) 파싱 시도
                    elif isinstance(tool_content, str):
                        try:
                            # QueryAnalysis의 __repr__ 형식 파싱
                            logger.debug("Attempting to parse string content as QueryAnalysis")
                            import re
                            import ast
                            
                            # 문자열에서 각 필드 추출
                            data = {}
                            
                            # clarity_score, specificity_score, searchability_score 추출
                            for score_name in ['clarity_score', 'specificity_score', 'searchability_score']:
                                match = re.search(rf'{score_name}=([\d.]+)', tool_content)
                                if match:
                                    data[score_name] = float(match.group(1))
                            
                            # improved_queries, issues, recommendations 추출 (리스트 형식)
                            for list_name in ['improved_queries', 'issues', 'recommendations']:
                                match = re.search(rf'{list_name}=(\[.*?\](?=\s+\w+=|$))', tool_content, re.DOTALL)
                                if match:
                                    try:
                                        data[list_name] = ast.literal_eval(match.group(1))
                                    except Exception:
                                        # 복잡한 리스트 파싱 실패 시 빈 리스트
                                        data[list_name] = []
                            
                            if data:
                                logger.debug(f"Successfully parsed QueryAnalysis from string: {data}")
                                return QueryAnalysis(**data)
                        except Exception as e:
                            logger.warning(f"Failed to parse string content: {e}")
        
        # 직접 QueryAnalysis가 반환된 경우
        if isinstance(response, QueryAnalysis):
            return response
        
        # dict에 output 키가 있는 경우
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
            if isinstance(output, QueryAnalysis):
                return output
            elif isinstance(output, dict):
                return QueryAnalysis(**output)
        
        # 폴백: 기본 분석 결과를 반환합니다.
        logger.warning(
            f"Unable to extract QueryAnalysis from response. "
            f"Response type: {type(response)}, "
            f"Has messages: {isinstance(response, dict) and 'messages' in response}"
        )
        
    except Exception as e:
        logger.error(f"Error extracting QueryAnalysis from response: {e}")
    
    # 최종 폴백
    return QueryAnalysis(
        clarity_score=0.5,
        specificity_score=0.5,
        searchability_score=0.5,
        improved_queries=[question],
        issues=["Unable to analyze query"],
        recommendations=["Use the original query"]
    )
