"""
Adaptive RAG 워크플로 구성 모듈.

LangGraph StateGraph API를 사용해 전체 워크플로를 구축하고 설정합니다.

OOD 감지 전략:
    - Hard OOD (domain_relevance < 0.2): 확실한 OOD → 즉시 거부 (인사, 날씨, 주식 등)
    - Soft OOD (0.2 ≤ domain_relevance < 0.5): 검색 먼저 시도 후 판단
    - In-domain (domain_relevance ≥ 0.5): 정상 QA 처리
"""

from functools import partial
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.graph.nodes import (
    classify_intent_node,
    analyze_query_node,
    agent_node,
    post_process_node,
    generate_ood_response_node,
    clarify_node,
    finalize_node,
)
from naver_connect_chatbot.service.tool.retrieval_tool import create_qdrant_search_tool
from naver_connect_chatbot.rag.web_search import create_google_search_tool
from naver_connect_chatbot.config import logger


# OOD 감지 임계값 상수 (2단계)
OOD_HARD_THRESHOLD = 0.2  # 이 이하: 확실한 OOD (인사, 날씨, 주식 등)
OOD_SOFT_THRESHOLD = 0.5  # hard~soft 사이: 검색 먼저 시도 후 판단


def route_after_intent(state: AdaptiveRAGState) -> str:
    """
    Intent 분류 후 다음 노드를 결정합니다.

    2단계 OOD 라우팅:
    - Hard OOD: intent가 OUT_OF_DOMAIN이고 domain_relevance < 0.2 → 즉시 OOD 응답
    - Soft OOD 또는 애매한 경우 → 검색 먼저 시도 (analyze_query로 진행)

    Args:
        state: 현재 워크플로 상태

    Returns:
        "generate_ood_response" | "analyze_query"
    """
    intent = state.get("intent", "SIMPLE_QA")
    domain_relevance = state.get("domain_relevance", 1.0)

    # Hard OOD: 확실히 도메인 밖 (greeting, 날씨, 주식 등)
    if intent == "OUT_OF_DOMAIN" and domain_relevance < OOD_HARD_THRESHOLD:
        logger.info(
            f"Hard OOD → routing to OOD response: intent={intent}, "
            f"domain_relevance={domain_relevance:.2f}"
        )
        return "generate_ood_response"

    # Soft OOD 또는 애매한 경우: 검색 먼저 시도
    if intent == "OUT_OF_DOMAIN" and domain_relevance < OOD_SOFT_THRESHOLD:
        logger.info(
            f"Soft OOD → routing to analyze_query (search first): intent={intent}, "
            f"domain_relevance={domain_relevance:.2f}"
        )

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


def should_continue(state: AdaptiveRAGState, max_tool_iterations: int = 3) -> str:
    """
    Agent 노드 실행 후 다음 단계를 결정합니다.

    마지막 AIMessage에 tool_calls가 있고 아직 최대 반복 횟수에 도달하지 않았으면
    "tools"로 라우팅하여 도구를 실행합니다. 그렇지 않으면 "post_process"로 이동합니다.

    Args:
        state: 현재 워크플로 상태
        max_tool_iterations: 최대 도구 호출 반복 횟수 (기본값: 3)

    Returns:
        "tools" | "post_process"
    """
    messages = state.get("messages", [])
    if not messages:
        return "post_process"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_call_count = state.get("tool_call_count", 0)
        if tool_call_count < max_tool_iterations:
            return "tools"
        logger.warning(
            f"Max tool iterations reached ({tool_call_count}/{max_tool_iterations}), "
            "routing to post_process"
        )

    return "post_process"


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,
    *,
    reasoning_llm: Runnable | None = None,
    reranker_settings: Any | None = None,
    gemini_llm_settings: Any | None = None,
    check_pointers: BaseCheckpointSaver | None = None,
    debug: bool = False,
    enable_clarification: bool = False,
    clarification_threshold: float = 0.5,
    max_tool_iterations: int = 3,
) -> CompiledStateGraph:
    """
    Tool-based Adaptive RAG 워크플로 그래프를 구성합니다.

    워크플로 구조:
    1. Intent Classification — 의도 분류 (OOD 감지 포함)
    2. [Hard OOD?] → OOD 응답 → Finalize
    3. [Soft OOD / In-domain] → Query Analysis → (Clarify?) → Agent ⇄ Tools 루프
    4. Post Process — 최종 답변 추출 + post-retrieval OOD 감지
    5. Finalize

    매개변수:
        retriever: 하이브리드 검색기
        llm: Gemini LLM (thinking_level=low, 분류/분석용)
        reasoning_llm: Agent용 Gemini LLM (thinking_level=high, 도구 호출 + 답변 생성)
        reranker_settings: ClovaStudioRerankerSettings (Qdrant tool 내 reranking용)
        gemini_llm_settings: GeminiLLMSettings (Google Search tool용)
        check_pointers: LangGraph 체크포인터 (선택)
        debug: 디버그 모드 활성화 여부 (선택)
        enable_clarification: 명확화 기능 활성화 (기본 False)
        clarification_threshold: 명확화 트리거 신뢰도 임계값 (기본 0.5)
        max_tool_iterations: agent ⇄ tools 최대 루프 횟수 (기본 3)

    반환값:
        컴파일된 LangGraph 워크플로
    """
    classification_llm = llm
    agent_llm = reasoning_llm or llm

    # ── 도구 생성 ──
    tools = []

    # Qdrant 검색 도구 (항상 등록)
    qdrant_tool = create_qdrant_search_tool(retriever, reranker_settings)
    tools.append(qdrant_tool)

    # Google Search 도구 (API 키가 있을 때만 등록)
    if gemini_llm_settings and getattr(gemini_llm_settings, "api_key", None):
        google_tool = create_google_search_tool(gemini_llm_settings)
        tools.append(google_tool)
        logger.info("Google Search tool registered")

    logger.info(
        f"Building tool-based workflow graph "
        f"(tools={[t.name for t in tools]}, max_iterations={max_tool_iterations})"
    )

    # ── 그래프 구성 ──
    workflow = StateGraph(state_schema=AdaptiveRAGState)

    # 노드 등록
    workflow.add_node("classify_intent", partial(classify_intent_node, llm=classification_llm))
    workflow.add_node("analyze_query", partial(analyze_query_node, llm=classification_llm))
    workflow.add_node("agent", partial(agent_node, llm=agent_llm, tools=tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("post_process", post_process_node)
    workflow.add_node("generate_ood_response", generate_ood_response_node)
    workflow.add_node("finalize", finalize_node)

    if enable_clarification:
        workflow.add_node("clarify", clarify_node)

    workflow.set_entry_point("classify_intent")

    # ── 엣지 ──

    # Intent 분류 후: Hard OOD → OOD 응답, 그 외 → Query 분석
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "generate_ood_response": "generate_ood_response",
            "analyze_query": "analyze_query",
        },
    )

    workflow.add_edge("generate_ood_response", "finalize")

    # Clarification 분기
    if enable_clarification:
        workflow.add_conditional_edges(
            "analyze_query",
            partial(
                should_clarify,
                enable_clarification=enable_clarification,
                clarification_threshold=clarification_threshold,
            ),
            {
                "clarify": "clarify",
                "continue": "agent",
            },
        )
        workflow.add_edge("clarify", "finalize")
    else:
        workflow.add_edge("analyze_query", "agent")

    # Agent ⇄ Tools 루프
    workflow.add_conditional_edges(
        "agent",
        partial(should_continue, max_tool_iterations=max_tool_iterations),
        {
            "tools": "tools",
            "post_process": "post_process",
        },
    )
    workflow.add_edge("tools", "agent")

    # 후처리 → 마무리
    workflow.add_edge("post_process", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(
        checkpointer=check_pointers,
        debug=debug,
        name="NaverConnectBoostCampChatbot",
    )
