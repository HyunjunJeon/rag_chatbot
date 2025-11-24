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
    segment_documents_node,
    evaluate_documents_node,
    generate_answer_node,
    finalize_node,
)
from naver_connect_chatbot.service.graph.routing import (
    check_document_sufficiency,
)
from naver_connect_chatbot.config import logger


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: Runnable,
    fast_llm: Runnable | None = None,
    check_pointers: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """
    Adaptive RAG 워크플로 그래프를 구성합니다 (단순화 버전).
    
    새로운 워크플로 구조:
    1. Intent Classification - 의도 분류
    2. Multi-Query Generation - 다중 쿼리 생성 (Query Analysis 통합)
    3. Hybrid Retrieval - Dense + Sparse 검색
    4. Reranking - Clova Studio Reranker (Post-Retriever)
    5. Document Segmentation - 긴 문서 자동 분할 (Post-Retriever)
    6. Document Evaluation - 간소화된 평가
    7. Answer Generation - Reasoning 모드 활용
    8. Finalize - 완료
    
    제거된 단계:
    - Answer Validation (Reasoning 모델이 자체 처리)
    - Correction Loop (Reasoning 모델이 자체 처리)
    - Reflection (불필요)
    
    매개변수:
        retriever: 하이브리드 검색기 (Multi-Query 기본 활성화)
        llm: CLOVA HCX-007 모델 (Reasoning 지원)
        fast_llm: 단순 작업용 경량 모델 (선택)
        check_pointers: LangGraph 체크포인터 (선택)
    
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
    # fast_llm이 있으면 단순 작업에 사용하고, 없으면 기본 llm을 재사용합니다.
    classification_llm = fast_llm or llm
    
    logger.info("Building Adaptive RAG workflow graph (simplified with Reasoning)")
    
    # 워크플로 그래프를 생성합니다.
    workflow = StateGraph(AdaptiveRAGState)
    
    # 노드를 추가하며 필요한 의존성을 주입합니다.
    workflow.add_node(
        "classify_intent",
        partial(classify_intent_node, llm=classification_llm)
    )
    workflow.add_node(
        "analyze_query",  # Multi-Query Generation 통합
        partial(analyze_query_node, llm=llm)
    )
    workflow.add_node(
        "retrieve",
        partial(retrieve_node, retriever=retriever)
    )
    workflow.add_node(
        "rerank",  # 새로 추가: Post-Retriever
        rerank_node
    )
    workflow.add_node(
        "segment_documents",  # 새로 추가: Post-Retriever
        segment_documents_node
    )
    workflow.add_node(
        "evaluate_documents",  # 간소화됨
        evaluate_documents_node
    )
    workflow.add_node(
        "generate_answer",  # Reasoning 모드 활용
        partial(generate_answer_node, llm=llm)
    )
    workflow.add_node("finalize", finalize_node)
    
    # 워크플로 간선을 정의합니다 (단순화된 선형 흐름).
    workflow.set_entry_point("classify_intent")
    
    # 1. Intent Classification -> Multi-Query Generation
    workflow.add_edge("classify_intent", "analyze_query")
    
    # 2. Multi-Query Generation -> Retrieval
    workflow.add_edge("analyze_query", "retrieve")
    
    # 3. Retrieval -> Reranking (Post-Retriever)
    workflow.add_edge("retrieve", "rerank")
    
    # 4. Reranking -> Document Segmentation (Post-Retriever)
    workflow.add_edge("rerank", "segment_documents")
    
    # 5. Document Segmentation -> Document Evaluation
    workflow.add_edge("segment_documents", "evaluate_documents")
    
    # 6. Document Evaluation -> Answer Generation (직접 연결, 조건부 제거)
    # 간소화: 문서가 부족해도 답변 생성 시도 (Reasoning 모델이 처리)
    workflow.add_edge("evaluate_documents", "generate_answer")
    
    # 7. Answer Generation -> Finalize (직접 연결, Validation/Correction 제거)
    # 간소화: Reasoning 모델이 자체 검증 수행
    workflow.add_edge("generate_answer", "finalize")
    
    # 8. Finalize -> END
    workflow.add_edge("finalize", END)
    
    logger.info("Adaptive RAG workflow graph built successfully (simplified)")
    logger.info("Workflow: Intent → Multi-Query → Retrieve → Rerank → Segment → Evaluate → Generate → Finalize")
    
    # 컴파일 후 반환합니다.
    return workflow.compile(checkpointer=check_pointers, debug=True, name="NaverConnectChatbot")

