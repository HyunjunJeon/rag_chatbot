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
    finalize_node,
)
from naver_connect_chatbot.config import logger


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,
    *,
    reasoning_llm: Runnable | None = None,
    check_pointers: BaseCheckpointSaver | None = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """
    Adaptive RAG 워크플로 그래프를 구성합니다.

    새로운 워크플로 구조:
    1. Intent Classification - 의도 분류
    2. Multi-Query Generation - 다중 쿼리 생성 (Query Analysis 통합)
    3. Hybrid Retrieval - Dense + Sparse 검색
    4. Reranking - Clova Studio Reranker (Post-Retriever)
    5. Answer Generation - Reasoning 모드 활용
    6. Finalize - 완료

    매개변수:
        retriever: 하이브리드 검색기 (Multi-Query 기본 활성화)
        llm: CLOVA HCX-007 모델 (Reasoning 지원)
        reasoning_llm: 답변 생성용 Reasoning LLM (선택)
        check_pointers: LangGraph 체크포인터 (선택)
        debug: 디버그 모드 활성화 여부 (선택)

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

    logger.info("Building NaverConnectBoostCampChatbot workflow graph")

    workflow = StateGraph(state_schema=AdaptiveRAGState)

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

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "analyze_query")
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
