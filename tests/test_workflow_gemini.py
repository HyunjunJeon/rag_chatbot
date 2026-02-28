"""
C-2: End-to-end 워크플로 통합 테스트 (Gemini API + MockRetriever).

실제 Gemini API를 호출하므로 GOOGLE_API_KEY가 .env에 설정되어 있어야 합니다.
VectorDB(Qdrant)는 필요하지 않습니다 — MockRetriever로 대체합니다.

실행:
    .venv/bin/python -m pytest tests/test_workflow_gemini.py -m integration -v --tb=short
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

import pytest  # noqa: E402


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_happy_path(mock_retriever, gemini_llm, gemini_reasoning_llm):
    """C-2-1: 인도메인 질문 → 전체 워크플로 → answer 반환."""
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=mock_retriever,
        llm=gemini_llm,
        reasoning_llm=gemini_reasoning_llm,
    )

    result = await graph.ainvoke({"question": "PyTorch의 텐서란 무엇인가요?"})

    assert result.get("answer"), "Answer should not be empty"
    assert result.get("intent") in ("SIMPLE_QA", "COMPLEX_REASONING", "EXPLORATORY")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_hard_ood(mock_retriever, gemini_llm, gemini_reasoning_llm):
    """C-2-2: Hard OOD 경로 — 인사말은 OOD 응답을 반환한다."""
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=mock_retriever,
        llm=gemini_llm,
        reasoning_llm=gemini_reasoning_llm,
    )

    result = await graph.ainvoke({"question": "안녕하세요!"})

    assert result.get("intent") == "OUT_OF_DOMAIN"
    assert result.get("is_out_of_domain") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_soft_ood_produces_answer(mock_retriever, gemini_llm, gemini_reasoning_llm):
    """C-2-3: Soft OOD 경로 — 검색을 먼저 시도하고 정상 답변을 반환한다."""
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=mock_retriever,
        llm=gemini_llm,
        reasoning_llm=gemini_reasoning_llm,
    )

    # Python 데코레이터는 도메인 인접 질문 — Soft OOD 또는 in-domain으로 처리될 수 있음
    result = await graph.ainvoke({"question": "Python 데코레이터의 작동 원리를 설명해주세요."})

    # 어떤 경로를 타든 answer 필드가 존재해야 한다
    assert result.get("answer") is not None, "Answer should be present for soft OOD path"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_post_retrieval_ood(empty_retriever, gemini_llm, gemini_reasoning_llm):
    """C-2-4: 빈 문서 + 도메인 무관 질문 → 워크플로가 종료되고 answer를 반환한다."""
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=empty_retriever,
        llm=gemini_llm,
        reasoning_llm=gemini_reasoning_llm,
    )

    # 양자컴퓨팅은 부트캠프 도메인 밖이지만 intent 분류를 통과할 수 있음
    result = await graph.ainvoke({"question": "양자 컴퓨팅에서 큐비트의 원리는?"})

    # OOD이든 정상 답변이든 answer 필드가 None이 아니어야 함
    assert result.get("answer") is not None
