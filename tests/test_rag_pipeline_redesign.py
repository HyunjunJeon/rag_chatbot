"""
새로운 RAG 파이프라인 구조 검증 테스트

CLOVA Studio 기반으로 재구성된 RAG 파이프라인을 검증합니다:
- Multi-Query Generation (Query Analysis 통합)
- Reranking (Post-Retriever)
- Document Segmentation (Post-Retriever)
- Simplified Document Evaluation
- Reasoning-enabled Answer Generation
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph
from naver_connect_chatbot.service.graph.state import AdaptiveRAGState


def test_workflow_graph_structure():
    """
    워크플로 그래프가 예상된 노드 구조를 가지는지 확인합니다.
    """
    # Mock retriever와 LLM 생성
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    
    # 워크플로 그래프 생성
    graph = build_adaptive_rag_graph(
        retriever=mock_retriever,
        llm=mock_llm,
    )
    
    # 그래프가 컴파일되었는지 확인
    assert graph is not None
    
    # 예상된 노드들이 존재하는지 확인
    expected_nodes = [
        "classify_intent",
        "analyze_query",
        "retrieve",
        "rerank",
        "segment_documents",
        "evaluate_documents",
        "generate_answer",
        "finalize",
    ]
    
    # 노드 이름 추출 (compiled graph에서)
    # Note: LangGraph의 내부 구조에 따라 접근 방법이 다를 수 있음
    graph_nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    
    for node in expected_nodes:
        assert node in graph_nodes, f"Node '{node}' not found in workflow graph"
    
    # 제거된 노드들이 없는지 확인
    removed_nodes = ["validate_answer", "correct"]
    for node in removed_nodes:
        assert node not in graph_nodes, f"Node '{node}' should be removed but still exists"


def test_workflow_nodes_removed():
    """
    Validation/Correction 노드가 제거되었는지 확인합니다.
    """
    from naver_connect_chatbot.service.graph import nodes
    
    # validate_answer_node와 correct_node가 더 이상 존재하지 않아야 함
    assert not hasattr(nodes, 'validate_answer_node'), \
        "validate_answer_node should be removed"
    assert not hasattr(nodes, 'correct_node'), \
        "correct_node should be removed"
    
    # 새로운 노드들이 존재하는지 확인
    assert hasattr(nodes, 'rerank_node'), \
        "rerank_node should exist"
    assert hasattr(nodes, 'segment_documents_node'), \
        "segment_documents_node should exist"


def test_rag_settings_updated():
    """
    RAG Settings가 새로운 feature flags를 포함하는지 확인합니다.
    """
    from naver_connect_chatbot.config.settings.rag_settings import AdaptiveRAGSettings
    
    settings = AdaptiveRAGSettings()
    
    # 비활성화된 설정들
    assert settings.enable_answer_validation == False, \
        "Answer validation should be disabled (Reasoning model handles it)"
    assert settings.enable_correction == False, \
        "Correction should be disabled (Reasoning model handles it)"
    
    # 활성화된 설정들
    assert settings.use_multi_query == True, \
        "Multi-query should be enabled by default"
    assert settings.use_reranking == True, \
        "Reranking should be enabled by default"
    
    # 새로 추가된 설정들
    assert hasattr(settings, 'use_segmentation'), \
        "use_segmentation setting should exist"
    assert hasattr(settings, 'segmentation_threshold'), \
        "segmentation_threshold setting should exist"
    
    assert settings.use_segmentation == True, \
        "Segmentation should be enabled by default"
    assert settings.segmentation_threshold == 1000, \
        "Segmentation threshold should be 1000"


def test_query_analysis_multi_query_integration():
    """
    Query Analysis와 Multi-Query Generation이 통합되었는지 확인합니다.
    """
    from naver_connect_chatbot.service.agents.query_analyzer import QueryAnalysis
    
    # QueryAnalysis 모델이 다중 쿼리 생성을 지원하는지 확인
    model_fields = QueryAnalysis.model_fields
    
    assert 'improved_queries' in model_fields, \
        "improved_queries field should exist in QueryAnalysis"
    
    # Docstring 확인
    assert "Multi-Query" in QueryAnalysis.__doc__, \
        "QueryAnalysis docstring should mention Multi-Query integration"


def test_types_updated():
    """
    ValidationUpdate와 CorrectionUpdate가 제거되었는지 확인합니다.
    """
    from naver_connect_chatbot.service.graph import types
    
    # 제거된 타입들이 없는지 확인
    assert not hasattr(types, 'ValidationUpdate'), \
        "ValidationUpdate should be removed"
    assert not hasattr(types, 'CorrectionUpdate'), \
        "CorrectionUpdate should be removed"
    
    # 유지된 타입들 확인
    assert hasattr(types, 'IntentUpdate'), "IntentUpdate should exist"
    assert hasattr(types, 'QueryAnalysisUpdate'), "QueryAnalysisUpdate should exist"
    assert hasattr(types, 'RetrievalUpdate'), "RetrievalUpdate should exist"
    assert hasattr(types, 'DocumentEvaluationUpdate'), "DocumentEvaluationUpdate should exist"
    assert hasattr(types, 'AnswerUpdate'), "AnswerUpdate should exist"


def test_retriever_factory_multi_query_default():
    """
    Retriever Factory에서 Multi-Query가 기본 활성화되는지 확인합니다.
    """
    from naver_connect_chatbot.rag.retriever_factory import build_advanced_hybrid_retriever
    import inspect
    
    # 함수 시그니처 확인
    sig = inspect.signature(build_advanced_hybrid_retriever)
    enable_multi_query_param = sig.parameters.get('enable_multi_query')
    
    assert enable_multi_query_param is not None, \
        "enable_multi_query parameter should exist"
    assert enable_multi_query_param.default == True, \
        "enable_multi_query should default to True"


@pytest.mark.asyncio
async def test_nodes_async_compatibility():
    """
    모든 노드가 비동기 호출을 지원하는지 확인합니다.
    """
    from naver_connect_chatbot.service.graph.nodes import (
        classify_intent_node,
        analyze_query_node,
        retrieve_node,
        rerank_node,
        segment_documents_node,
        evaluate_documents_node,
        generate_answer_node,
    )
    import inspect
    
    # 비동기 노드들 확인
    async_nodes = [
        classify_intent_node,
        analyze_query_node,
        retrieve_node,
        rerank_node,
        segment_documents_node,
        evaluate_documents_node,
        generate_answer_node,
    ]
    
    for node_func in async_nodes:
        assert inspect.iscoroutinefunction(node_func), \
            f"{node_func.__name__} should be async"


if __name__ == "__main__":
    # 개별 테스트 실행
    print("Running RAG Pipeline Redesign Tests...")
    
    print("\n1. Testing workflow graph structure...")
    test_workflow_graph_structure()
    print("✓ Workflow graph structure is correct")
    
    print("\n2. Testing removed nodes...")
    test_workflow_nodes_removed()
    print("✓ Old nodes removed, new nodes added")
    
    print("\n3. Testing RAG settings...")
    test_rag_settings_updated()
    print("✓ RAG settings updated correctly")
    
    print("\n4. Testing Query Analysis + Multi-Query integration...")
    test_query_analysis_multi_query_integration()
    print("✓ Query Analysis and Multi-Query integrated")
    
    print("\n5. Testing type definitions...")
    test_types_updated()
    print("✓ Type definitions updated")
    
    print("\n6. Testing Retriever Factory defaults...")
    test_retriever_factory_multi_query_default()
    print("✓ Retriever Factory Multi-Query default enabled")
    
    print("\n7. Testing node async compatibility...")
    import asyncio
    asyncio.run(test_nodes_async_compatibility())
    print("✓ All nodes support async calls")
    
    print("\n" + "="*60)
    print("All RAG Pipeline Redesign Tests Passed! ✓")
    print("="*60)

