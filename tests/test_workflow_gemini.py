"""
C-2: End-to-end 워크플로 통합 테스트 (Gemini API + MockRetriever).

실제 Gemini API를 호출하므로 GOOGLE_API_KEY가 .env에 설정되어 있어야 합니다.
VectorDB(Qdrant)는 필요하지 않습니다 — MockRetriever로 대체합니다.

실행:
    .venv/bin/python -m pytest tests/test_workflow_gemini.py -m integration -v --tb=short
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_multi_turn_followup_avoids_exact_repeat(
    mock_retriever,
    monkeypatch,
):
    """C-2-5: 같은 thread의 후속 질문에서 완전 동일 답변 반복을 피한다."""
    from langgraph.checkpoint.memory import MemorySaver
    from naver_connect_chatbot.service.agents.intent_classifier import IntentClassification
    from naver_connect_chatbot.service.agents.query_analyzer import QueryAnalysis
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aclassify_intent",
        AsyncMock(
            return_value=IntentClassification(
                intent="SIMPLE_QA",
                confidence=0.95,
                reasoning="release-gate multi-turn deterministic test",
                domain_relevance=0.95,
            )
        ),
    )
    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aanalyze_query",
        AsyncMock(
            return_value=QueryAnalysis(
                clarity_score=0.9,
                specificity_score=0.9,
                searchability_score=0.9,
                improved_queries=["PyTorch 텐서 개념", "PyTorch 텐서 예제 코드"],
            )
        ),
    )

    # 멀티턴 중복 방지 동작을 안정적으로 검증하기 위해
    # 응답 생성 단계는 결정론적 mock LLM을 사용합니다.
    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="PyTorch 텐서는 다차원 배열을 표현하는 핵심 데이터 구조입니다."),
            AIMessage(
                content=(
                    "반복 없이 새 정보로 답변합니다. 예제 코드:\n"
                    "```python\nimport torch\nx = torch.tensor([1, 2, 3])\n```"
                )
            ),
        ]
    )
    reasoning_llm = MagicMock()
    reasoning_llm.bind_tools = MagicMock(return_value=bound_llm)

    graph = build_adaptive_rag_graph(
        retriever=mock_retriever,
        llm=MagicMock(),
        reasoning_llm=reasoning_llm,
        check_pointers=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "c2-5-multiturn"}}

    first = await graph.ainvoke({"question": "PyTorch 텐서란 무엇인가요?"}, config=config)
    second = await graph.ainvoke(
        {"question": "좋아요. 이번에는 간단한 예제 코드도 보여주세요."},
        config=config,
    )

    first_answer = first.get("answer", "")
    second_answer = second.get("answer", "")

    assert first_answer.strip(), "First turn answer must not be empty"
    assert second_answer.strip(), "Second turn answer must not be empty"
    assert second_answer != first_answer, "Follow-up answer should not be an exact repeat"
    assert any(
        keyword in second_answer
        for keyword in ("예제", "코드", "torch", "```", "Python")
    ), "Follow-up answer should reflect the code-example intent"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_web_search_activation_with_two_fixed_retries(
    empty_retriever,
    monkeypatch,
):
    """C-2-6: 프롬프트 기반 web search 활성화를 최대 2회 재시도로 검증한다."""
    from naver_connect_chatbot.config import get_chat_model, settings
    from naver_connect_chatbot.service.agents.intent_classifier import IntentClassification
    from naver_connect_chatbot.service.agents.query_analyzer import QueryAnalysis
    from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.fail("GOOGLE_API_KEY is required for release-gate integration tests")

    # OOD 분기로 빠지지 않도록 intent/query 분석만 고정하고,
    # 도구 호출 자체는 Gemini agent가 자율적으로 결정하도록 둡니다.
    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aclassify_intent",
        AsyncMock(
            return_value=IntentClassification(
                intent="SIMPLE_QA",
                confidence=0.95,
                reasoning="release-gate web search activation test",
                domain_relevance=0.95,
            )
        ),
    )
    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aanalyze_query",
        AsyncMock(
            return_value=QueryAnalysis(
                clarity_score=0.8,
                specificity_score=0.8,
                searchability_score=0.9,
                improved_queries=[
                    "2026년 2월 AI 최신 뉴스",
                    "recent AI updates February 2026",
                ],
            )
        ),
    )

    llm = get_chat_model(thinking_level="low")
    reasoning_llm = get_chat_model()

    graph = build_adaptive_rag_graph(
        retriever=empty_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        gemini_llm_settings=settings.gemini_llm,
    )

    query = (
        "최신 정보 확인이 필요합니다. 반드시 web_search 도구를 사용해서 "
        "2026년 2월 AI 주요 소식을 2개만 요약해주세요."
    )

    web_search_activated = False
    for attempt in range(2):  # 고정 재시도 2회
        if attempt == 0:
            question = query
        else:
            question = (
                "반드시 web_search 도구를 호출한 뒤, 웹 검색 근거(URL 포함) 2개를 제시해 주세요. "
                "질문: 2026년 2월 AI 주요 소식"
            )

        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": f"c2-6-web-search-{attempt}"}},
        )

        messages = result.get("messages", [])
        has_web_search_tool_call = any(
            isinstance(msg, AIMessage)
            and any(tc.get("name") == "web_search" for tc in getattr(msg, "tool_calls", []))
            for msg in messages
        )
        has_web_search_tool_output = any(
            isinstance(msg, ToolMessage) and "웹 검색 결과" in msg.content
            for msg in messages
        )

        if has_web_search_tool_call or has_web_search_tool_output:
            web_search_activated = True
            break

    assert web_search_activated, (
        "Web search tool must be activated at least once within 2 attempts"
    )
