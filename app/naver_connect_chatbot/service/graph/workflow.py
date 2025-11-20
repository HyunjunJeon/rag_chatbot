"""
Adaptive RAG 워크플로 구성 모듈.

LangGraph StateGraph API를 사용해 전체 워크플로를 구축하고 설정합니다.
"""

from typing import Any
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
    evaluate_documents_node,
    generate_answer_node,
    validate_answer_node,
    correct_node,
    finalize_node,
)
from naver_connect_chatbot.service.graph.routing import (
    check_document_sufficiency,
    check_answer_quality,
    route_after_correction,
)
from naver_connect_chatbot.config import logger


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,
    fast_llm: Runnable | None = None,
    check_pointers: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """
    Adaptive RAG 워크플로 그래프를 구성합니다.
    
    다음 단계를 모두 포함하는 시스템을 생성합니다.
    - 의도 분류
    - 질의 분석 및 정제
    - 하이브리드 검색 기반 문서 검색
    - 문서 평가
    - 의도별 전략을 활용한 답변 생성
    - 환각·품질 검증
    - 교정 루프
    
    매개변수:
        retriever: 하이브리드 검색기
        llm: 복잡한 작업에 사용할 주 모델
        fast_llm: 단순 작업에 사용할 선택적 경량 모델
    
    반환값:
        컴파일된 LangGraph 워크플로
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from naver_connect_chatbot.rag import get_hybrid_retriever
        >>> 
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> fast_llm = ChatOpenAI(model="gpt-4o-mini")
        >>> retriever = get_hybrid_retriever(...)
        >>> 
        >>> graph = build_adaptive_rag_graph(retriever, llm, fast_llm)
        >>> result = await graph.ainvoke({
        ...     "question": "What is PyTorch?",
        ...     "max_retries": 2,
        ... })
    """
    # fast_llm이 있으면 단순 작업에 사용하고, 없으면 기본 llm을 재사용합니다.
    classification_llm = fast_llm or llm
    evaluation_llm = fast_llm or llm
    
    logger.info("Building Adaptive RAG workflow graph")
    
    # 워크플로 그래프를 생성합니다.
    workflow = StateGraph(AdaptiveRAGState)
    
    # 노드를 추가하며 필요한 의존성을 주입합니다.
    workflow.add_node(
        "classify_intent",
        partial(classify_intent_node, llm=classification_llm)
    )
    workflow.add_node(
        "analyze_query",
        partial(analyze_query_node, llm=llm)
    )
    workflow.add_node(
        "retrieve",
        partial(retrieve_node, retriever=retriever)
    )
    workflow.add_node(
        "evaluate_documents",
        partial(evaluate_documents_node, llm=evaluation_llm)
    )
    workflow.add_node(
        "generate_answer",
        partial(generate_answer_node, llm=llm)
    )
    workflow.add_node(
        "validate_answer",
        partial(validate_answer_node, llm=llm)
    )
    workflow.add_node(
        "correct",
        partial(correct_node, llm=llm)
    )
    workflow.add_node("finalize", finalize_node)
    
    # 워크플로 간선을 정의합니다.
    workflow.set_entry_point("classify_intent")
    
    # 의도 분류 -> 질의 분석
    workflow.add_edge("classify_intent", "analyze_query")
    
    # 질의 분석 -> 검색
    workflow.add_edge("analyze_query", "retrieve")
    
    # 검색 -> 문서 평가
    workflow.add_edge("retrieve", "evaluate_documents")
    
    # 문서 평가 -> 조건부 라우팅
    workflow.add_conditional_edges(
        "evaluate_documents",
        check_document_sufficiency,
        {
            "generate_answer": "generate_answer",
            "refine_query": "analyze_query",  # 더 나은 검색을 위해 루프백
            "generate_best_effort": "generate_answer",  # 문서가 부족해도 시도
        }
    )
    
    # 답변 생성 -> 검증
    workflow.add_edge("generate_answer", "validate_answer")
    
    # 답변 검증 -> 조건부 라우팅
    workflow.add_conditional_edges(
        "validate_answer",
        check_answer_quality,
        {
            "finalize": "finalize",
            "correct": "correct",
            "return_best_effort": "finalize",  # 현재 결과를 수용
        }
    )
    
    # 교정 -> 조건부 라우팅
    workflow.add_conditional_edges(
        "correct",
        route_after_correction,
        {
            "analyze_query": "analyze_query",  # 질의를 정제 후 재검색
            "generate_answer": "generate_answer",  # 동일 문맥으로 재생성
            "finalize": "finalize",  # 현재 답변을 그대로 반환
        }
    )
    
    # 종료 노드 -> END
    workflow.add_edge("finalize", END)
    
    logger.info("Adaptive RAG workflow graph built successfully")
    
    # 컴파일 후 반환합니다.
    return workflow.compile(checkpointer=check_pointers, debug=True, name="NaverConnectChatbot")

