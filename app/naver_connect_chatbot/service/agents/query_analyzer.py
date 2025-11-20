"""
Adaptive RAG용 질의 분석 에이전트 구현.

질의 품질을 평가하고 개선 방향을 제안합니다.
"""

from typing import Any, List
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_query_analysis_prompt
from naver_connect_chatbot.config import logger


class QueryAnalysis(BaseModel):
    """
    질의 품질 평가 및 개선 제안을 담는 모델입니다.
    
    속성:
        clarity_score: 질의의 명확도 (0.0 ~ 1.0)
        specificity_score: 질의의 구체성 (0.0 ~ 1.0)
        searchability_score: 검색 친화도 (0.0 ~ 1.0)
        improved_queries: 개선된 질의 후보 목록
        issues: 원본 질의의 문제점
        recommendations: 개선 권장 사항
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
    improved_queries: List[str] = Field(
        description="List of improved query variations",
        default_factory=list
    )
    issues: List[str] = Field(
        description="Identified issues",
        default_factory=list
    )
    recommendations: List[str] = Field(
        description="Recommendations for improvement",
        default_factory=list
    )


def create_query_analyzer(llm: Runnable) -> Any:
    """
    질의 분석 에이전트를 생성합니다.
    
    명확성, 구체성, 검색 친화도를 평가하고 개선 방향을 제안합니다.
    
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
        # 분석 프롬프트를 불러옵니다.
        system_prompt = get_query_analysis_prompt()
        
        # 구조화된 출력을 반환하도록 LLM을 구성합니다.
        structured_llm = llm.with_structured_output(QueryAnalysis)
        
        # 도구 없이 분석만 수행하는 에이전트를 생성합니다.
        agent = create_agent(
            model=structured_llm,
            tools=[],
            system_prompt=system_prompt,
        )
        
        logger.debug("Query analyzer agent created successfully")
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
    
    # 응답에서 구조화된 출력을 추출합니다.
    if hasattr(response, "content") and isinstance(response.content, QueryAnalysis):
        return response.content
    elif isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        # 폴백: 기본 분석 결과를 반환합니다.
        logger.warning(f"Unexpected response format from query analyzer: {type(response)}")
        return QueryAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            searchability_score=0.5,
            improved_queries=[question],
            issues=["Unable to analyze query"],
            recommendations=["Use the original query"]
        )

