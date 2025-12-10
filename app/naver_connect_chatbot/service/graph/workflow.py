"""
Adaptive RAG 워크플로 구성 모듈.

LangGraph StateGraph API를 사용해 전체 워크플로를 구축하고 설정합니다.
"""

from functools import partial
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from langgraph.graph.state import CompiledStateGraph
from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.graph.nodes import (
    classify_intent_node,
    analyze_query_node,
    retrieve_node,
    rerank_node,
    generate_answer_node,
    generate_ood_response_node,
    clarify_node,
    finalize_node,
)
from naver_connect_chatbot.config import logger


# OOD 감지 임계값 상수
OOD_DOMAIN_RELEVANCE_THRESHOLD = 0.3


def route_after_intent(state: AdaptiveRAGState) -> str:
    """
    Intent 분류 후 다음 노드를 결정합니다.

    OUT_OF_DOMAIN 질문은 OOD 응답 노드로,
    그 외 질문은 Query Analysis 노드로 라우팅합니다.

    Args:
        state: 현재 워크플로 상태

    Returns:
        "generate_ood_response" | "analyze_query"
    """
    intent = state.get("intent", "SIMPLE_QA")
    domain_relevance = state.get("domain_relevance", 1.0)

    # OUT_OF_DOMAIN 감지: intent가 OUT_OF_DOMAIN이거나 domain_relevance가 낮은 경우
    if intent == "OUT_OF_DOMAIN" or domain_relevance < OOD_DOMAIN_RELEVANCE_THRESHOLD:
        logger.info(
            f"Routing to OOD response: intent={intent}, "
            f"domain_relevance={domain_relevance:.2f}"
        )
        return "generate_ood_response"

    logger.info(f"Routing to analyze_query: intent={intent}")
    return "analyze_query"


def should_clarify(
    state: AdaptiveRAGState,
    enable_clarification: bool = False,
    clarification_threshold: float = 0.5,
) -> str:
    """
    필터 신뢰도를 기반으로 명확화 필요 여부를 결정합니다.

    Args:
        state: 현재 워크플로 상태
        enable_clarification: 명확화 기능 활성화 여부
        clarification_threshold: 명확화 트리거 임계값 (기본 0.5)

    Returns:
        "clarify" 또는 "continue"

    Note:
        confidence <= threshold일 때 clarify로 라우팅합니다.
        (경계값 포함 - off-by-one 버그 방지)
    """
    if not enable_clarification:
        return "continue"

    confidence = state.get("filter_confidence", 1.0)
    # 경계값 포함: confidence가 threshold와 정확히 같을 때도 clarify
    if confidence <= clarification_threshold:
        logger.info(
            f"Filter confidence ({confidence:.2f}) at or below threshold "
            f"({clarification_threshold}), routing to clarify"
        )
        return "clarify"
    return "continue"


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,
    *,
    reasoning_llm: Runnable | None = None,
    check_pointers: BaseCheckpointSaver | None = None,
    debug: bool = False,
    enable_clarification: bool = False,
    clarification_threshold: float = 0.5,
) -> CompiledStateGraph:
    """
    Adaptive RAG 워크플로 그래프를 구성합니다.

    새로운 워크플로 구조:
    1. Intent Classification - 의도 분류
    2. Multi-Query Generation - 다중 쿼리 생성 (Query Analysis 통합)
    3. (선택) Clarification - 필터 신뢰도 낮으면 사용자에게 명확화 요청
    4. Hybrid Retrieval - Dense + Sparse 검색
    5. Reranking - Clova Studio Reranker (Post-Retriever)
    6. Answer Generation - Reasoning 모드 활용
    7. Finalize - 완료

    매개변수:
        retriever: 하이브리드 검색기 (Multi-Query 기본 활성화)
        llm: CLOVA HCX-007 모델 (Reasoning 지원)
        reasoning_llm: 답변 생성용 Reasoning LLM (선택)
        check_pointers: LangGraph 체크포인터 (선택)
        debug: 디버그 모드 활성화 여부 (선택)
        enable_clarification: 명확화 기능 활성화 (기본 False)
        clarification_threshold: 명확화 트리거 신뢰도 임계값 (기본 0.5)

    반환값:
        컴파일된 LangGraph 워크플로

    예시:
        >>> from naver_connect_chatbot.config import get_llm
        >>> from naver_connect_chatbot.rag import build_advanced_hybrid_retriever
        >>>
        >>> llm = get_llm()  # CLOVA HCX-007
        >>> retriever = build_advanced_hybrid_retriever(...)
        >>>
        >>> graph = build_adaptive_rag_graph(retriever, llm)
        >>> result = await graph.ainvoke({
        ...     "question": "PyTorch란 무엇인가요?",
        ... })
    """
    # Intent classification과 query analysis에 사용할 LLM
    classification_llm = llm

    # Answer generation 전용 LLM
    answer_llm = reasoning_llm

    logger.info("Building NaverConnectBoostCampChatbot workflow graph (with OOD detection)")

    workflow = StateGraph(state_schema=AdaptiveRAGState)

    # 노드 등록
    workflow.add_node("classify_intent", partial(classify_intent_node, llm=classification_llm))
    workflow.add_node(
        "analyze_query",
        partial(analyze_query_node, llm=classification_llm),
    )
    workflow.add_node("retrieve", partial(retrieve_node, retriever=retriever))
    workflow.add_node(
        "rerank",
        rerank_node,
    )
    workflow.add_node(
        "generate_answer",
        partial(generate_answer_node, llm=answer_llm),
    )
    workflow.add_node("finalize", finalize_node)

    # OOD 응답 노드 (항상 등록)
    workflow.add_node("generate_ood_response", generate_ood_response_node)

    # Clarification 노드 (활성화된 경우에만 사용)
    if enable_clarification:
        workflow.add_node("clarify", clarify_node)

    workflow.set_entry_point("classify_intent")

    # Intent 분류 후 조건부 라우팅: OUT_OF_DOMAIN → OOD 응답, 그 외 → Query 분석
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "generate_ood_response": "generate_ood_response",
            "analyze_query": "analyze_query",
        },
    )

    # OOD 응답 노드는 바로 finalize로 이동
    workflow.add_edge("generate_ood_response", "finalize")

    # Clarification 분기 처리
    if enable_clarification:
        # 조건부 라우팅: filter_confidence에 따라 clarify 또는 retrieve로 분기
        workflow.add_conditional_edges(
            "analyze_query",
            partial(
                should_clarify,
                enable_clarification=enable_clarification,
                clarification_threshold=clarification_threshold,
            ),
            {
                "clarify": "clarify",
                "continue": "retrieve",
            },
        )
        # clarify 노드는 finalize로 직접 연결 (사용자 응답 대기)
        workflow.add_edge("clarify", "finalize")
    else:
        workflow.add_edge("analyze_query", "retrieve")

    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(
        checkpointer=check_pointers,
        debug=debug,
        name="NaverConnectBoostCampChatbot",
    )
