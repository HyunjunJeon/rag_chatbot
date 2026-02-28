"""
C-1: Agent + Gemini API 연동 통합 테스트.

실제 Gemini API를 호출하므로 GOOGLE_API_KEY가 .env에 설정되어 있어야 합니다.
VectorDB(Qdrant)는 필요하지 않습니다.

실행:
    .venv/bin/python -m pytest tests/test_agents_gemini.py -m integration -v --tb=short
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

import pytest  # noqa: E402


@pytest.mark.integration
@pytest.mark.asyncio
async def test_classify_intent_technical_question(gemini_llm):
    """C-1-1: 기술 질문이 SIMPLE_QA 또는 COMPLEX_REASONING으로 분류된다."""
    from naver_connect_chatbot.service.agents.intent_classifier import (
        IntentClassification,
        aclassify_intent,
    )

    result = await aclassify_intent("PyTorch에서 텐서를 GPU로 옮기는 방법은?", llm=gemini_llm)

    assert isinstance(result, IntentClassification)
    assert result.intent in ("SIMPLE_QA", "COMPLEX_REASONING")
    assert result.domain_relevance >= 0.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_classify_intent_ood_question(gemini_llm):
    """C-1-2: 날씨 질문이 OUT_OF_DOMAIN으로 분류된다."""
    from naver_connect_chatbot.service.agents.intent_classifier import aclassify_intent

    result = await aclassify_intent("내일 서울 날씨 어때?", llm=gemini_llm)

    assert result.intent == "OUT_OF_DOMAIN"
    assert result.domain_relevance < 0.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_query_expansion(gemini_llm):
    """C-1-3: 쿼리 분석이 확장된 쿼리와 clarity_score를 반환한다."""
    from naver_connect_chatbot.service.agents.query_analyzer import (
        QueryAnalysis,
        aanalyze_query,
    )

    result = await aanalyze_query("CNN의 Conv 레이어 작동 원리", intent="SIMPLE_QA", llm=gemini_llm)

    assert isinstance(result, QueryAnalysis)
    assert len(result.improved_queries) >= 1
    assert 0 <= result.clarity_score <= 1


@pytest.mark.integration
def test_structured_output_returns_pydantic(gemini_llm):
    """C-1-4: with_structured_output()이 Pydantic 인스턴스를 반환한다."""
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        answer: str = Field(description="answer")
        score: float = Field(ge=0, le=1)

    structured = gemini_llm.with_structured_output(TestModel)
    result = structured.invoke("What is 1+1? Provide score=1.0")

    assert isinstance(result, TestModel), (
        f"Expected TestModel instance, got {type(result)}: {result}"
    )
    assert isinstance(result.answer, str)
    assert 0 <= result.score <= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_structured_output_thinking_levels(gemini_llm, gemini_reasoning_llm):
    """C-1-5: thinking_level=low와 기본값(high) 모두 structured output이 정상 동작한다."""
    from naver_connect_chatbot.service.agents.intent_classifier import (
        IntentClassification,
        aclassify_intent,
    )

    question = "Transformer의 Self-Attention 메커니즘이란?"

    # thinking_level=low (gemini_llm)
    result_low = await aclassify_intent(question, llm=gemini_llm)
    assert isinstance(result_low, IntentClassification)
    assert result_low.intent in (
        "SIMPLE_QA",
        "COMPLEX_REASONING",
        "EXPLORATORY",
        "CLARIFICATION_NEEDED",
        "OUT_OF_DOMAIN",
    )

    # thinking_level=high (gemini_reasoning_llm)
    result_high = await aclassify_intent(question, llm=gemini_reasoning_llm)
    assert isinstance(result_high, IntentClassification)
    assert result_high.intent in (
        "SIMPLE_QA",
        "COMPLEX_REASONING",
        "EXPLORATORY",
        "CLARIFICATION_NEEDED",
        "OUT_OF_DOMAIN",
    )
