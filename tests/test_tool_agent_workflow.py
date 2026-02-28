"""
Tool 기반 Retrieval 아키텍처 테스트.

Agent ⇄ Tools 동적 루프의 핵심 컴포넌트를 검증합니다:
- Part 1: should_continue 라우팅 (순수 단위 테스트)
- Part 2: post_process_node 후처리 (단위 테스트)
- Part 3: agent_node LLM 호출 (Mock LLM)
- Part 4: create_qdrant_search_tool (Mock Retriever)
- Part 5: create_google_search_tool (Mock)
- Part 6: 전체 워크플로 통합 (Mock LLM)
- Part 7: 멀티턴 대화 (Mock LLM + Checkpointer)
- Part 8: Gemini 통합 테스트 (@pytest.mark.integration)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from naver_connect_chatbot.service.graph.nodes import agent_node, post_process_node
from naver_connect_chatbot.service.graph.workflow import (
    build_adaptive_rag_graph,
    should_continue,
)
from naver_connect_chatbot.service.tool.retrieval_tool import create_qdrant_search_tool


# ============================================================================
# Mock LLM 팩토리
# ============================================================================


def _make_tool_calling_llm(response: AIMessage):
    """bind_tools().ainvoke()가 단일 response를 반환하는 mock LLM."""
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(return_value=response)
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_bound)
    return mock_llm


def _make_sequential_tool_calling_llm(responses: list[AIMessage]):
    """bind_tools().ainvoke()가 responses를 순차적으로 반환하는 mock LLM."""
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(side_effect=responses)
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_bound)
    return mock_llm


def _mock_classification(intent="SIMPLE_QA", confidence=0.9, domain_relevance=0.9):
    """IntentClassification mock을 생성하는 헬퍼."""
    from naver_connect_chatbot.service.agents.intent_classifier import IntentClassification

    return IntentClassification(
        intent=intent,
        confidence=confidence,
        reasoning="test",
        domain_relevance=domain_relevance,
    )


def _mock_analysis(queries=None):
    """QueryAnalysis mock을 생성하는 헬퍼."""
    from naver_connect_chatbot.service.agents.query_analyzer import QueryAnalysis

    return QueryAnalysis(
        clarity_score=0.8,
        specificity_score=0.7,
        searchability_score=0.9,
        improved_queries=queries or ["query"],
    )


# ============================================================================
# Part 1: should_continue 라우팅 (순수 단위 테스트, LLM 불필요)
# ============================================================================


class TestShouldContinue:
    """should_continue 라우팅 함수의 순수 단위 테스트."""

    def test_empty_messages(self):
        assert should_continue({"messages": []}) == "post_process"

    def test_no_tool_calls(self):
        state = {"messages": [AIMessage(content="답변", tool_calls=[])]}
        assert should_continue(state) == "post_process"

    def test_tool_calls_within_limit(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                )
            ],
            "tool_call_count": 1,
        }
        assert should_continue(state, max_tool_iterations=3) == "tools"

    def test_tool_calls_at_limit(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                )
            ],
            "tool_call_count": 3,
        }
        assert should_continue(state, max_tool_iterations=3) == "post_process"

    def test_tool_calls_over_limit(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                )
            ],
            "tool_call_count": 5,
        }
        assert should_continue(state, max_tool_iterations=3) == "post_process"

    def test_default_max_iterations(self):
        """max_tool_iterations 미지정 시 기본값 3 사용."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                )
            ],
            "tool_call_count": 2,
        }
        assert should_continue(state) == "tools"

    def test_missing_tool_call_count(self):
        """state에 tool_call_count 없으면 기본값 0 사용."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                )
            ],
        }
        assert should_continue(state) == "tools"

    def test_non_ai_message_last(self):
        """마지막 메시지가 ToolMessage면 post_process."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                ),
                ToolMessage(content="결과", tool_call_id="1"),
            ],
        }
        assert should_continue(state) == "post_process"


# ============================================================================
# Part 2: post_process_node (단위 테스트, LLM 불필요)
# ============================================================================


class TestPostProcessNode:
    """post_process_node의 답변 추출 및 OOD 감지 테스트."""

    async def test_extracts_final_answer(self):
        """마지막 AIMessage(tool_calls 없는)에서 텍스트를 추출."""
        state = {
            "messages": [
                HumanMessage(content="PyTorch란?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "PyTorch"}}
                    ],
                ),
                ToolMessage(
                    content="[검색 결과: 1건]\nPyTorch는...", tool_call_id="1"
                ),
                AIMessage(content="PyTorch는 딥러닝 프레임워크입니다."),
            ],
            "domain_relevance": 0.9,
            "question": "PyTorch란?",
        }
        result = await post_process_node(state)
        assert result["answer"] == "PyTorch는 딥러닝 프레임워크입니다."
        assert result["generation_strategy"] == "tool_based_agent"

    async def test_skips_tool_call_messages(self):
        """중간 AIMessage에 tool_calls가 있으면 건너뛰고, 마지막 깨끗한 AIMessage를 추출."""
        state = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content="검색합니다",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                ),
                ToolMessage(content="결과", tool_call_id="1"),
                AIMessage(
                    content="추가 검색",
                    tool_calls=[
                        {"id": "2", "name": "qdrant_search", "args": {"query": "q2"}}
                    ],
                ),
                ToolMessage(content="결과2", tool_call_id="2"),
                AIMessage(content="최종 종합 답변"),
            ],
            "domain_relevance": 0.9,
            "question": "질문",
        }
        result = await post_process_node(state)
        assert result["answer"] == "최종 종합 답변"

    async def test_post_retrieval_ood(self):
        """모든 ToolMessage에 '검색 결과 없음' + 낮은 domain_relevance → soft decline."""
        state = {
            "messages": [
                HumanMessage(content="오늘 날씨 어때?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "날씨"}}
                    ],
                ),
                ToolMessage(
                    content="검색 결과 없음: '날씨'에 대한 관련 교육 자료를 찾지 못했습니다.",
                    tool_call_id="1",
                ),
                AIMessage(content="관련 정보를 찾지 못했습니다."),
            ],
            "domain_relevance": 0.3,
            "question": "오늘 날씨 어때?",
        }
        result = await post_process_node(state)
        assert result["is_out_of_domain"] is True
        assert result["generation_strategy"] == "post_retrieval_soft_decline"
        assert "교육 자료에서 관련 정보를 검색했으나" in result["answer"]

    async def test_no_ood_when_relevance_high(self):
        """'검색 결과 없음'이지만 domain_relevance >= 0.5이면 OOD 아님."""
        state = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                ),
                ToolMessage(content="검색 결과 없음: 관련 자료 없음", tool_call_id="1"),
                AIMessage(content="일반 지식으로 답변합니다."),
            ],
            "domain_relevance": 0.8,
            "question": "질문",
        }
        result = await post_process_node(state)
        assert result.get("is_out_of_domain") is not True
        assert result["generation_strategy"] == "tool_based_agent"

    async def test_no_ood_when_tools_have_results(self):
        """ToolMessage에 정상 결과 있으면 낮은 domain_relevance라도 OOD 아님."""
        state = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                ),
                ToolMessage(
                    content="[검색 결과: 3건]\nPyTorch는...", tool_call_id="1"
                ),
                AIMessage(content="답변입니다."),
            ],
            "domain_relevance": 0.3,
            "question": "질문",
        }
        result = await post_process_node(state)
        assert result.get("is_out_of_domain") is not True

    async def test_fallback_when_no_ai_message(self):
        """messages에 AIMessage가 없으면 폴백 메시지."""
        state = {
            "messages": [HumanMessage(content="질문")],
            "domain_relevance": 0.9,
            "question": "질문",
        }
        result = await post_process_node(state)
        assert "답변을 생성할 수 없었습니다" in result["answer"]

    async def test_gemini_thinking_block(self):
        """Gemini thinking block 형식의 content에서 type=text 블록만 추출."""
        state = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content=[
                        {"type": "thinking", "text": "내부 추론..."},
                        {"type": "text", "text": "최종 답변입니다."},
                    ]
                ),
            ],
            "domain_relevance": 0.9,
            "question": "질문",
        }
        result = await post_process_node(state)
        assert result["answer"] == "최종 답변입니다."

    async def test_max_iter_tool_calls_with_text(self):
        """max iterations 도달 시, tool_calls가 있지만 content 텍스트도 있는 AIMessage에서 추출."""
        state = {
            "messages": [
                HumanMessage(content="복잡한 질문"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q1"}}
                    ],
                ),
                ToolMessage(content="[검색 결과: 2건]\n문서 내용...", tool_call_id="1"),
                # max iterations 도달 — tool_calls 있지만 content에도 답변 텍스트 포함
                AIMessage(
                    content="여러 검색 결과를 종합하면, PyTorch는 딥러닝 프레임워크입니다.",
                    tool_calls=[
                        {"id": "2", "name": "qdrant_search", "args": {"query": "q2"}}
                    ],
                ),
            ],
            "domain_relevance": 0.9,
            "question": "복잡한 질문",
        }
        result = await post_process_node(state)
        assert "종합" in result["answer"]
        assert result["generation_strategy"] == "tool_based_agent"

    async def test_thinking_block_only_no_text(self):
        """thinking 블록만 있고 text 블록이 없으면 thinking 텍스트가 폴백으로 사용됨."""
        state = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content=[
                        {"type": "thinking", "text": "내부 추론 과정입니다..."},
                    ]
                ),
            ],
            "domain_relevance": 0.9,
            "question": "질문",
        }
        result = await post_process_node(state)
        # _extract_text_from_content 폴백: text 블록 없으면 첫 thinking 텍스트 사용
        assert result["answer"] == "내부 추론 과정입니다..."
        assert result["generation_strategy"] == "tool_based_agent"


# ============================================================================
# Part 3: agent_node (단위 테스트, Mock LLM)
# ============================================================================


class TestAgentNode:
    """agent_node의 LLM 호출 및 상태 업데이트 테스트."""

    async def test_returns_tool_calls(self):
        """LLM이 tool_calls를 반환하면 messages에 추가되고 count 증가."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "qdrant_search", "args": {"query": "PyTorch"}}
            ],
        )
        mock_llm = _make_tool_calling_llm(ai_msg)
        state = {
            "messages": [HumanMessage(content="PyTorch란?")],
            "question": "PyTorch란?",
            "intent": "SIMPLE_QA",
            "domain_relevance": 0.9,
            "refined_queries": ["PyTorch 텐서"],
            "tool_call_count": 0,
        }

        result = await agent_node(state, llm=mock_llm, tools=[])
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls
        assert result["tool_call_count"] == 1

    async def test_returns_final_answer(self):
        """LLM이 tool_calls 없이 답변만 반환하면 count 미증가."""
        mock_llm = _make_tool_calling_llm(AIMessage(content="최종 답변"))
        state = {
            "messages": [HumanMessage(content="질문")],
            "question": "질문",
            "intent": "SIMPLE_QA",
            "domain_relevance": 0.9,
            "refined_queries": [],
            "tool_call_count": 0,
        }

        result = await agent_node(state, llm=mock_llm, tools=[])
        assert result["tool_call_count"] == 0
        assert result["messages"][0].content == "최종 답변"

    async def test_increments_count(self):
        """기존 tool_call_count=2에서 tool_calls 반환 시 3으로 증가."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
            ],
        )
        mock_llm = _make_tool_calling_llm(ai_msg)
        state = {
            "messages": [HumanMessage(content="질문")],
            "question": "질문",
            "intent": "SIMPLE_QA",
            "domain_relevance": 0.9,
            "refined_queries": [],
            "tool_call_count": 2,
        }

        result = await agent_node(state, llm=mock_llm, tools=[])
        assert result["tool_call_count"] == 3

    async def test_multi_turn_filters_previous_tool_messages(self):
        """이전 턴의 ToolMessage는 LLM에 전달되지 않음 (컨텍스트 절약)."""
        mock_llm = _make_tool_calling_llm(AIMessage(content="답변"))
        state = {
            "messages": [
                # ── 이전 턴 ──
                HumanMessage(content="이전 질문"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "qdrant_search", "args": {"query": "q"}}
                    ],
                ),
                ToolMessage(content="이전 검색 결과", tool_call_id="1"),
                AIMessage(content="이전 답변"),
                # ── 현재 턴 ──
                HumanMessage(content="후속 질문"),
            ],
            "question": "후속 질문",
            "intent": "SIMPLE_QA",
            "domain_relevance": 0.9,
            "refined_queries": [],
            "tool_call_count": 0,
        }

        await agent_node(state, llm=mock_llm, tools=[])

        call_messages = mock_llm.bind_tools.return_value.ainvoke.call_args[0][0]

        # ToolMessage 없어야 함
        assert not any(isinstance(m, ToolMessage) for m in call_messages)
        # 이전 + 현재 HumanMessage 모두 포함
        humans = [m for m in call_messages if isinstance(m, HumanMessage)]
        assert len(humans) == 2

    async def test_system_prompt_includes_intent(self):
        """시스템 프롬프트에 intent 정보가 포함됨."""
        mock_llm = _make_tool_calling_llm(AIMessage(content="답변"))
        state = {
            "messages": [HumanMessage(content="질문")],
            "question": "질문",
            "intent": "COMPLEX_REASONING",
            "domain_relevance": 0.85,
            "refined_queries": [],
            "tool_call_count": 0,
        }

        await agent_node(state, llm=mock_llm, tools=[])

        call_messages = mock_llm.bind_tools.return_value.ainvoke.call_args[0][0]
        system_msg = call_messages[0]
        assert isinstance(system_msg, SystemMessage)
        assert "COMPLEX_REASONING" in system_msg.content

    async def test_previous_answer_truncated(self):
        """이전 턴의 AIMessage가 500자 초과면 500자+'...'로 잘려서 LLM에 전달."""
        long_answer = "A" * 600  # 600자 — 500자 한도 초과
        mock_llm = _make_tool_calling_llm(AIMessage(content="2턴 답변"))
        state = {
            "messages": [
                # ── 이전 턴 ──
                HumanMessage(content="이전 질문"),
                AIMessage(content=long_answer),  # tool_calls 없는 최종 답변 (600자)
                # ── 현재 턴 ──
                HumanMessage(content="후속 질문"),
            ],
            "question": "후속 질문",
            "intent": "SIMPLE_QA",
            "domain_relevance": 0.9,
            "refined_queries": [],
            "tool_call_count": 0,
        }

        await agent_node(state, llm=mock_llm, tools=[])

        call_messages = mock_llm.bind_tools.return_value.ainvoke.call_args[0][0]
        # 이전 턴의 AIMessage 찾기
        prev_ai_msgs = [
            m for m in call_messages
            if isinstance(m, AIMessage) and m.content != "2턴 답변"
        ]
        assert len(prev_ai_msgs) == 1
        assert len(prev_ai_msgs[0].content) == 503  # 500 + len("...")
        assert prev_ai_msgs[0].content.endswith("...")


# ============================================================================
# Part 4: create_qdrant_search_tool (단위 테스트, Mock Retriever)
# ============================================================================


class TestQdrantSearchTool:
    """create_qdrant_search_tool로 생성된 Qdrant 검색 도구 테스트."""

    async def test_returns_formatted_results(self, mock_retriever):
        """MockRetriever의 3개 문서가 포맷팅되어 반환."""
        tool = create_qdrant_search_tool(mock_retriever)
        result = await tool.coroutine("PyTorch", state={"retrieval_filters": None})

        assert "[검색 결과: 3건]" in result
        assert "PyTorch" in result

    async def test_empty_retriever_returns_no_results(self, empty_retriever):
        """빈 retriever → '검색 결과 없음' 메시지."""
        tool = create_qdrant_search_tool(empty_retriever)
        result = await tool.coroutine("nothing", state={"retrieval_filters": None})

        assert "검색 결과 없음" in result

    async def test_applies_filters_from_state(self, mock_retriever):
        """state의 retrieval_filters로 doc_type 필터링 적용."""
        tool = create_qdrant_search_tool(mock_retriever)
        result = await tool.coroutine(
            "PyTorch",
            state={"retrieval_filters": {"doc_type": ["pdf"]}},
        )

        # mock_retriever: pdf(1), lecture_transcript(1), notebook(1) → pdf만 통과
        assert "[검색 결과: 1건]" in result

    async def test_tool_schema_hides_state(self, mock_retriever):
        """InjectedState의 state 파라미터가 도구 스키마에 노출되지 않음."""
        tool = create_qdrant_search_tool(mock_retriever)
        assert "state" not in tool.args
        assert "query" in tool.args


# ============================================================================
# Part 5: create_google_search_tool (단위 테스트, Mock)
# ============================================================================


def _web_search_docs(count: int) -> list[Document]:
    """테스트용 웹 검색 Document 리스트 생성 헬퍼."""
    return [
        Document(
            page_content=f"문서 {i} 내용",
            metadata={
                "source_type": "web_search",
                "title": f"제목 {i}",
                "url": f"https://example.com/{i}",
            },
        )
        for i in range(count)
    ]


class TestGoogleSearchTool:
    """create_google_search_tool로 생성된 웹 검색 도구 테스트."""

    async def test_returns_formatted_web_results(self, monkeypatch):
        """검색 결과가 제목, URL 포함하여 포맷팅."""
        mock_docs = [
            Document(
                page_content="PyTorch는 딥러닝 프레임워크입니다.",
                metadata={
                    "source_type": "web_search",
                    "title": "PyTorch 공식 문서",
                    "url": "https://pytorch.org",
                },
            ),
            Document(
                page_content="PyTorch 튜토리얼입니다.",
                metadata={
                    "source_type": "web_search",
                    "title": "PyTorch Tutorial",
                    "url": "https://pytorch.org/tutorials",
                },
            ),
            Document(
                page_content="텐서 연산 가이드입니다.",
                metadata={
                    "source_type": "web_search",
                    "title": "텐서 연산",
                    "url": "https://example.com/tensor",
                },
            ),
        ]
        monkeypatch.setattr(
            "naver_connect_chatbot.rag.web_search.google_search_retrieve",
            AsyncMock(return_value=mock_docs),
        )

        from naver_connect_chatbot.rag.web_search import create_google_search_tool

        tool = create_google_search_tool(MagicMock())
        result = await tool.coroutine("PyTorch")

        assert "[웹 검색 결과: 3건]" in result
        assert "PyTorch 공식 문서" in result
        assert "https://pytorch.org" in result

    async def test_empty_results(self, monkeypatch):
        """빈 결과 → '웹 검색 결과 없음' 메시지."""
        monkeypatch.setattr(
            "naver_connect_chatbot.rag.web_search.google_search_retrieve",
            AsyncMock(return_value=[]),
        )

        from naver_connect_chatbot.rag.web_search import create_google_search_tool

        tool = create_google_search_tool(MagicMock())
        result = await tool.coroutine("nothing")

        assert "웹 검색 결과 없음" in result

    async def test_limits_to_5_results(self, monkeypatch):
        """10개 결과 중 상위 5개만 포함."""
        monkeypatch.setattr(
            "naver_connect_chatbot.rag.web_search.google_search_retrieve",
            AsyncMock(return_value=_web_search_docs(10)),
        )

        from naver_connect_chatbot.rag.web_search import create_google_search_tool

        tool = create_google_search_tool(MagicMock())
        result = await tool.coroutine("query")

        assert "[웹 검색 결과: 5건]" in result
        assert "제목 4" in result
        assert "제목 5" not in result


# ============================================================================
# Part 6: 전체 워크플로 통합 테스트 (Mock LLM)
# ============================================================================


def _patch_classifiers(monkeypatch, *, intent="SIMPLE_QA", domain_relevance=0.9):
    """classify_intent와 analyze_query를 monkeypatch로 모킹하는 헬퍼."""
    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aclassify_intent",
        AsyncMock(
            return_value=_mock_classification(
                intent=intent, domain_relevance=domain_relevance
            )
        ),
    )
    monkeypatch.setattr(
        "naver_connect_chatbot.service.graph.nodes.aanalyze_query",
        AsyncMock(return_value=_mock_analysis()),
    )


class TestWorkflowIntegration:
    """build_adaptive_rag_graph()로 실제 그래프를 빌드하고 Mock LLM으로 E2E 검증."""

    async def test_happy_path_with_tool_call(self, monkeypatch, mock_retriever):
        """classify → analyze → agent(tool) → tools → agent(답변) → post_process → finalize."""
        _patch_classifiers(monkeypatch)

        agent_llm = _make_sequential_tool_calling_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "qdrant_search",
                            "args": {"query": "PyTorch 텐서"},
                        }
                    ],
                ),
                AIMessage(content="PyTorch 텐서는 다차원 배열 데이터 구조입니다."),
            ]
        )

        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
            reasoning_llm=agent_llm,
        )

        result = await graph.ainvoke({"question": "PyTorch의 텐서란?"})
        assert "다차원 배열" in result["answer"]
        assert result["generation_strategy"] == "tool_based_agent"

    async def test_direct_answer_no_tools(self, monkeypatch, mock_retriever):
        """Agent가 도구 없이 바로 답변 (이전 컨텍스트 충분)."""
        _patch_classifiers(monkeypatch)

        agent_llm = _make_tool_calling_llm(
            AIMessage(content="이전 대화 기반으로 답변합니다.")
        )

        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
            reasoning_llm=agent_llm,
        )

        result = await graph.ainvoke({"question": "간단 질문"})
        assert "답변" in result["answer"]
        assert result["generation_strategy"] == "tool_based_agent"

    async def test_ood_bypass(self, mock_retriever):
        """'안녕하세요' → 패턴 매칭 Hard OOD → OOD 응답 (LLM 미사용)."""
        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
        )

        result = await graph.ainvoke({"question": "안녕하세요"})
        assert result["is_out_of_domain"] is True
        assert result["generation_strategy"] == "ood_decline"

    async def test_max_iterations_reached(self, monkeypatch, mock_retriever):
        """Agent가 max_tool_iterations(3)회 연속 tool_calls → 강제 종료 후 텍스트 추출."""
        _patch_classifiers(
            monkeypatch, intent="COMPLEX_REASONING", domain_relevance=0.9
        )

        # 3번 연속 tool_calls, 마지막에 텍스트 포함
        agent_llm = _make_sequential_tool_calling_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"c{i}",
                            "name": "qdrant_search",
                            "args": {"query": f"q{i}"},
                        }
                    ],
                )
                for i in range(2)
            ]
            + [
                AIMessage(
                    content="여러 차례 검색한 결과를 종합하면 답변입니다.",
                    tool_calls=[
                        {
                            "id": "c2",
                            "name": "qdrant_search",
                            "args": {"query": "q2"},
                        }
                    ],
                ),
            ]
        )

        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
            reasoning_llm=agent_llm,
            max_tool_iterations=3,
        )

        result = await graph.ainvoke({"question": "복잡한 질문"})
        assert result["answer"]
        assert "종합" in result["answer"]


# ============================================================================
# Part 7: 멀티턴 테스트 (Mock LLM + Checkpointer)
# ============================================================================


class TestMultiTurn:
    """MemorySaver를 사용한 멀티턴 대화 시뮬레이션."""

    async def test_multi_turn_context_preserved(self, monkeypatch, mock_retriever):
        """2턴 대화에서 1턴의 HumanMessage + AIMessage(최종)가 2턴에 전달됨."""
        from langgraph.checkpoint.memory import MemorySaver

        _patch_classifiers(monkeypatch)

        agent_llm = _make_sequential_tool_calling_llm(
            [
                # 턴 1: tool call → 답변
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "qdrant_search",
                            "args": {"query": "PyTorch"},
                        }
                    ],
                ),
                AIMessage(content="PyTorch는 딥러닝 프레임워크입니다."),
                # 턴 2: 바로 답변
                AIMessage(content="코드 예제: import torch"),
            ]
        )

        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
            reasoning_llm=agent_llm,
            check_pointers=MemorySaver(),
        )

        config = {"configurable": {"thread_id": "test-mt-1"}}

        result1 = await graph.ainvoke({"question": "PyTorch란?"}, config=config)
        assert "딥러닝 프레임워크" in result1["answer"]

        result2 = await graph.ainvoke({"question": "예제 코드는?"}, config=config)
        assert result2["answer"]

        # 2턴의 LLM 호출에서 1턴의 HumanMessage가 포함되었는지 검증
        call_messages = agent_llm.bind_tools.return_value.ainvoke.call_args[0][0]
        human_contents = [
            m.content for m in call_messages if isinstance(m, HumanMessage)
        ]
        assert "PyTorch란?" in human_contents

    async def test_multi_turn_tool_messages_filtered(
        self, monkeypatch, mock_retriever
    ):
        """2턴에서 1턴의 ToolMessage가 LLM에 전달되지 않음."""
        from langgraph.checkpoint.memory import MemorySaver

        _patch_classifiers(monkeypatch)

        agent_llm = _make_sequential_tool_calling_llm(
            [
                # 턴 1: tool call → 답변
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "qdrant_search",
                            "args": {"query": "q"},
                        }
                    ],
                ),
                AIMessage(content="1턴 답변"),
                # 턴 2: 바로 답변
                AIMessage(content="2턴 답변"),
            ]
        )

        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=MagicMock(),
            reasoning_llm=agent_llm,
            check_pointers=MemorySaver(),
        )

        config = {"configurable": {"thread_id": "test-mt-2"}}

        await graph.ainvoke({"question": "질문1"}, config=config)
        await graph.ainvoke({"question": "질문2"}, config=config)

        # 2턴의 LLM 호출에서 ToolMessage가 필터링되었는지 검증
        call_messages = agent_llm.bind_tools.return_value.ainvoke.call_args[0][0]
        tool_msgs = [m for m in call_messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0


# ============================================================================
# Part 8: 통합 테스트 (실제 Gemini API)
# ============================================================================


class TestGeminiIntegration:
    """실제 Gemini API를 사용한 E2E 통합 테스트."""

    @pytest.mark.integration
    async def test_gemini_tool_calling(
        self, mock_retriever, gemini_llm, gemini_reasoning_llm
    ):
        """실제 Gemini로 tool calling 워크플로가 정상 작동하는지 검증."""
        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=gemini_llm,
            reasoning_llm=gemini_reasoning_llm,
        )

        result = await graph.ainvoke({"question": "PyTorch의 텐서란 무엇인가요?"})
        assert result["answer"]
        assert result["generation_strategy"] == "tool_based_agent"

    @pytest.mark.integration
    async def test_gemini_ood_regression(
        self, mock_retriever, gemini_llm, gemini_reasoning_llm
    ):
        """OOD 질문('오늘 날씨 어때?')에 대한 기존 동작 회귀 테스트."""
        graph = build_adaptive_rag_graph(
            retriever=mock_retriever,
            llm=gemini_llm,
            reasoning_llm=gemini_reasoning_llm,
        )

        result = await graph.ainvoke({"question": "오늘 날씨 어때?"})
        assert result["is_out_of_domain"] is True
