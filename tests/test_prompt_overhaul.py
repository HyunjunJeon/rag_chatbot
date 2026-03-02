"""Prompt overhaul regression tests for English-only and multi-turn wiring."""

# ruff: noqa: E402

import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from naver_connect_chatbot.prompts.loader import load_prompt_config
from naver_connect_chatbot.rag.retriever.multi_query_retriever import (
    MultiQueryOutput,
    MultiQueryRetriever,
)
from naver_connect_chatbot.service.agents.answer_generator import generate_answer
from naver_connect_chatbot.service.agents.intent_classifier import (
    IntentClassification,
    aclassify_intent,
)
from naver_connect_chatbot.service.agents.query_analyzer import (
    QueryAnalysis,
    QueryExpansionResult,
    QueryQualityResult,
    QueryRetrievalFilters,
    aanalyze_query,
    aanalyze_query_split,
)

PROMPT_NAMES = [
    "intent_classification",
    "query_analysis",
    "multi_query_generation",
    "query_expansion",
    "query_quality_analysis",
    "answer_generation_simple",
    "answer_generation_complex",
    "answer_generation_exploratory",
]


class DummyRetriever(BaseRetriever):
    """Minimal retriever for MultiQueryRetriever construction in tests."""

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return []


class DummyLLM(Runnable):
    """Minimal runnable with with_structured_output support."""

    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, _schema):
        return self._structured

    def invoke(self, input, config=None, **kwargs):
        return input

    async def ainvoke(self, input, config=None, **kwargs):
        return input


def test_prompt_templates_require_conversation_history_variable():
    """All prompt templates in scope must accept conversation_history."""
    for name in PROMPT_NAMES:
        config = load_prompt_config(name)
        assert "conversation_history" in config.input_variables


def test_prompt_templates_are_english_only_in_scope_files():
    """Prompt templates should not contain Korean characters after overhaul."""
    template_dir = PROJECT_ROOT / "app" / "naver_connect_chatbot" / "prompts" / "templates"
    korean_pattern = re.compile(r"[가-힣]")

    for name in PROMPT_NAMES:
        text = (template_dir / f"{name}.yaml").read_text(encoding="utf-8")
        assert korean_pattern.search(text) is None


def test_prompt_templates_do_not_use_role_or_inputs_sections():
    """Role/Input sections should not appear in system prompts."""
    template_dir = PROJECT_ROOT / "app" / "naver_connect_chatbot" / "prompts" / "templates"
    for name in PROMPT_NAMES:
        text = (template_dir / f"{name}.yaml").read_text(encoding="utf-8")
        assert "## Role" not in text
        assert "## Inputs" not in text


def test_prompt_templates_use_core_task_structure():
    """All prompt templates should use the new task-centric section headers."""
    template_dir = PROJECT_ROOT / "app" / "naver_connect_chatbot" / "prompts" / "templates"
    for name in PROMPT_NAMES:
        text = (template_dir / f"{name}.yaml").read_text(encoding="utf-8")
        assert "## Core Task" in text
        assert "## Critical Rules" in text
        assert "## Output Contract" in text
        assert "## Final Check" in text


def test_prompt_templates_render_successfully_with_runtime_inputs():
    """All target templates should render with required runtime input variables."""
    runtime_inputs = {
        "intent_classification": {
            "question": "What is self-attention?",
            "conversation_history": "No prior conversation.",
        },
        "query_analysis": {
            "question": "Find CNN materials",
            "intent": "SIMPLE_QA",
            "data_source_context": "Available sources: pdf, slack_qa",
            "conversation_history": "No prior conversation.",
        },
        "query_quality_analysis": {
            "question": "Find CNN materials",
            "intent": "SIMPLE_QA",
            "conversation_history": "No prior conversation.",
        },
        "query_expansion": {
            "question": "Find CNN materials",
            "intent": "SIMPLE_QA",
            "clarity": 0.8,
            "specificity": 0.7,
            "data_source_context": "Available sources: pdf, slack_qa",
            "conversation_history": "No prior conversation.",
        },
        "multi_query_generation": {
            "query": "How to tune learning rate?",
            "num": 4,
            "conversation_history": "No prior conversation.",
        },
        "answer_generation_simple": {
            "question": "질문",
            "context": "[doc] content",
            "conversation_history": "No prior conversation.",
        },
        "answer_generation_complex": {
            "question": "질문",
            "context": "[doc] content",
            "conversation_history": "No prior conversation.",
        },
        "answer_generation_exploratory": {
            "question": "질문",
            "context": "[doc] content",
            "conversation_history": "No prior conversation.",
        },
    }

    for name, kwargs in runtime_inputs.items():
        from naver_connect_chatbot.prompts import get_prompt

        rendered = get_prompt(name).format_prompt(**kwargs).to_string()
        assert rendered
        assert isinstance(rendered, str)


def test_aclassify_intent_injects_conversation_history_into_prompt():
    """aclassify_intent should include conversation history when formatting prompt."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Intent classifier."),
            ("human", "History:{conversation_history}\nQuestion:{question}"),
        ]
    )

    mock_result = IntentClassification(
        intent="SIMPLE_QA",
        confidence=0.9,
        reasoning="clear technical question",
        domain_relevance=0.95,
    )
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_result)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    with patch(
        "naver_connect_chatbot.service.agents.intent_classifier.get_prompt",
        return_value=prompt_template,
    ):
        result = asyncio.run(
            aclassify_intent(
                "How does self-attention work?",
                llm=mock_llm,
                conversation_history="User asked about transformers in prior turn.",
            )
        )

    assert result.intent == "SIMPLE_QA"
    rendered_prompt = mock_structured.ainvoke.call_args.args[0]
    assert "prior turn" in rendered_prompt


def test_aanalyze_query_injects_history_and_data_source_context():
    """aanalyze_query should render both history and data source context."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sources:{data_source_context}\nHistory:{conversation_history}",
            ),
            (
                "human",
                "Intent:{intent}\nQuestion:{question}",
            ),
        ]
    )

    mock_result = QueryAnalysis(
        clarity_score=0.8,
        specificity_score=0.7,
        searchability_score=0.9,
        improved_queries=["query 1", "query 2", "query 3"],
    )
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_result)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    with patch(
        "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
        return_value=prompt_template,
    ):
        result = asyncio.run(
            aanalyze_query(
                question="Find CNN lecture notes",
                intent="SIMPLE_QA",
                llm=mock_llm,
                data_source_context="Available sources: pdf, slack_qa",
                conversation_history="Previously discussed CV basics.",
            )
        )

    assert isinstance(result, QueryAnalysis)
    rendered_prompt = mock_structured.ainvoke.call_args.args[0]
    assert "Available sources: pdf, slack_qa" in rendered_prompt
    assert "Previously discussed CV basics." in rendered_prompt


def test_aanalyze_query_split_passes_conversation_history_to_both_steps():
    """Split query path should forward conversation history to quality and expansion steps."""
    quality_result = QueryQualityResult(
        clarity_score=0.6,
        specificity_score=0.5,
        searchability_score=0.7,
        issues=[],
        recommendations=[],
    )
    expansion_result = QueryExpansionResult(
        improved_queries=["q1", "q2", "q3"],
        retrieval_filters=QueryRetrievalFilters(filter_confidence=0.8),
    )

    with patch(
        "naver_connect_chatbot.service.agents.query_analyzer._analyze_quality",
        new=AsyncMock(return_value=quality_result),
    ) as mock_quality:
        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer._expand_query",
            new=AsyncMock(return_value=expansion_result),
        ) as mock_expand:
            _ = asyncio.run(
                aanalyze_query_split(
                    question="Expand this",
                    intent="SIMPLE_QA",
                    llm=MagicMock(),
                    data_source_context="sources",
                    conversation_history="history text",
                )
            )

    assert mock_quality.await_count == 1
    assert mock_quality.await_args.args[-1] == "history text"
    assert mock_expand.await_count == 1
    assert mock_expand.await_args.kwargs["conversation_history"] == "history text"


def test_generate_answer_uses_formatted_prompt_with_history():
    """Answer generator should render full chat template with conversation history."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Always answer in Korean."),
            (
                "human",
                "History:{conversation_history}\nContext:{context}\nQuestion:{question}",
            ),
        ]
    )

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content="테스트 답변"))

    with patch(
        "naver_connect_chatbot.service.agents.answer_generator.get_prompt",
        return_value=prompt_template,
    ):
        answer = generate_answer(
            question="질문",
            context="문맥",
            llm=mock_llm,
            strategy="simple",
            conversation_history="이전 대화",
        )

    assert answer == "테스트 답변"
    rendered_prompt = mock_llm.invoke.call_args.args[0]
    assert "Always answer in Korean." in rendered_prompt
    assert "이전 대화" in rendered_prompt
    assert "문맥" in rendered_prompt


def test_multi_query_retriever_supplies_empty_history_when_template_supports_it():
    """MultiQueryRetriever should provide a default empty history value."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "History:{conversation_history}\nQuery:{query}\nNum:{num}",
            )
        ]
    )

    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value=MultiQueryOutput(queries=["qA", "qB"]))

    retriever = MultiQueryRetriever(
        base_retriever=DummyRetriever(),
        llm=DummyLLM(mock_structured),
        num_queries=2,
    )

    with patch(
        "naver_connect_chatbot.rag.retriever.multi_query_retriever.get_prompt",
        return_value=prompt_template,
    ):
        generated = retriever._generate_queries_sync("origin query")

    assert generated[0] == "origin query"
    rendered_prompt = mock_structured.invoke.call_args.args[0]
    assert "History:" in rendered_prompt
    assert "Query:origin query" in rendered_prompt
