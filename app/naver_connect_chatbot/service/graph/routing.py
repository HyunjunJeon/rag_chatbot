"""
Adaptive RAG 워크플로의 라우팅 로직.

상태 조건에 따라 다음 노드를 결정하는 조건부 라우팅 함수를 제공합니다.
"""

from typing import Literal
from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.config import logger


def route_by_intent(
    state: AdaptiveRAGState
) -> Literal["analyze_query", "generate_answer"]:
    """
    분류된 의도에 따라 다음 노드를 결정합니다.
    
    의도에 따라 다른 처리가 필요할 수 있습니다.
    - SIMPLE_QA: 질의 분석으로 이동
    - COMPLEX_REASONING: 질의 분석으로 이동
    - EXPLORATORY: 질의 분석으로 이동
    - CLARIFICATION_NEEDED: 질의 분석으로 이동
    
    현재는 모든 의도가 질의 분석으로 흐르지만 추후 확장 가능합니다.
    
    매개변수:
        state: 현재 워크플로 상태
    
    반환값:
        다음 노드 이름
    """
    intent = state.get("intent", "SIMPLE_QA")
    logger.debug(f"Routing by intent: {intent}")
    
    # 현재는 모든 의도를 질의 분석으로 보냅니다.
    # 향후 매우 단순한 질의는 생략할 수 있습니다.
    return "analyze_query"


def check_document_sufficiency(
    state: AdaptiveRAGState
) -> Literal["generate_answer", "refine_query", "generate_best_effort"]:
    """
    검색된 문서가 충분한지 확인합니다.
    
    다음 중 어떤 경로로 진행할지 결정합니다.
    - 현재 문서로 답변 생성
    - 질의를 정제 후 재검색
    - 재시도 한계를 넘긴 경우 최선의 답변 생성
    
    매개변수:
        state: 현재 워크플로 상태
    
    반환값:
        다음 노드 이름
    """
    sufficient = state.get("sufficient_context", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    logger.debug(
        f"Checking document sufficiency: sufficient={sufficient}, "
        f"retry_count={retry_count}, max_retries={max_retries}"
    )
    
    if sufficient:
        logger.info("Documents are sufficient, proceeding to answer generation")
        return "generate_answer"
    elif retry_count >= max_retries:
        logger.warning(
            f"Max retrieval retries ({max_retries}) exceeded, "
            "generating best-effort answer"
        )
        return "generate_best_effort"
    else:
        logger.info(
            f"Documents insufficient (retry {retry_count + 1}/{max_retries}), "
            "refining query"
        )
        return "refine_query"




def check_query_quality(
    state: AdaptiveRAGState
) -> Literal["retrieve", "skip_to_generate"]:
    """
    질의 품질이 검색에 충분한지 확인합니다.
    
    품질이 매우 높으면 추가 처리를 생략할 수 있습니다.
    
    매개변수:
        state: 현재 워크플로 상태
    
    반환값:
        다음 노드 이름
    """
    query_analysis = state.get("query_analysis", {})
    
    # 품질 점수를 확인합니다.
    clarity = query_analysis.get("clarity_score", 0.0)
    specificity = query_analysis.get("specificity_score", 0.0)
    searchability = query_analysis.get("searchability_score", 0.0)
    
    avg_quality = (clarity + specificity + searchability) / 3
    
    logger.debug(f"Query quality: {avg_quality:.2f}")
    
    # 현재는 모든 질의가 검색 단계를 거칩니다.
    # (향후 특수 케이스는 생략 가능)
    return "retrieve"

