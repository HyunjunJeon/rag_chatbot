"""
B-5: Agent Mock 단위 테스트.

classify_intent(), aclassify_intent(), analyze_query(), aanalyze_query()를
mock LLM으로 검증합니다. API 키 불필요.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from langchain_core.prompts import ChatPromptTemplate

from naver_connect_chatbot.service.agents.intent_classifier import (
    IntentClassification,
    classify_intent,
    aclassify_intent,
)
from naver_connect_chatbot.service.agents.query_analyzer import (
    QueryAnalysis,
    QueryRetrievalFilters,
    analyze_query,
    aanalyze_query,
)


def _make_fake_prompt_template() -> ChatPromptTemplate:
    """테스트용 최소 ChatPromptTemplate."""
    from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate

    return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(template="You are a helpful assistant.", input_variables=[])
            )
        ]
    )


# ============================================================================
# 헬퍼: mock LLM 팩토리
# ============================================================================


def _make_sync_llm(return_value):
    """structured_llm.invoke()가 return_value를 반환하는 mock LLM."""
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value=return_value)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


def _make_async_llm(return_value):
    """structured_llm.ainvoke()가 return_value를 반환하는 async mock LLM."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=return_value)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


def _make_error_sync_llm(exception=None):
    """invoke()가 예외를 발생시키는 mock LLM."""
    if exception is None:
        exception = Exception("Mock LLM failure")

    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(side_effect=exception)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


def _make_error_async_llm(exception=None):
    """ainvoke()가 예외를 발생시키는 async mock LLM."""
    if exception is None:
        exception = Exception("Mock async LLM failure")

    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=exception)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


# ============================================================================
# B-5-1: classify_intent() Mock LLM → IntentClassification 반환
# ============================================================================


class TestClassifyIntentSync:
    """classify_intent() 동기 버전 테스트"""

    def test_returns_intent_classification_instance(self):
        """mock LLM → IntentClassification 인스턴스 반환"""
        mock_result = IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.9,
            reasoning="Technical question about PyTorch",
            domain_relevance=0.8,
        )
        mock_llm = _make_sync_llm(mock_result)

        result = classify_intent("PyTorch란?", llm=mock_llm)

        assert isinstance(result, IntentClassification)
        assert result.intent == "SIMPLE_QA"
        assert result.confidence == 0.9
        assert result.domain_relevance == 0.8

    def test_with_structured_output_called_with_correct_model(self):
        """with_structured_output(IntentClassification)이 호출됨"""
        mock_result = IntentClassification(
            intent="COMPLEX_REASONING",
            confidence=0.85,
            reasoning="Complex analysis needed",
            domain_relevance=0.9,
        )
        mock_llm = _make_sync_llm(mock_result)

        classify_intent("Transformer의 Attention 메커니즘을 설명해줘", llm=mock_llm)

        mock_llm.with_structured_output.assert_called_once_with(IntentClassification)

    def test_out_of_domain_intent_returned(self):
        """OUT_OF_DOMAIN intent가 올바르게 전달됨"""
        mock_result = IntentClassification(
            intent="OUT_OF_DOMAIN",
            confidence=0.95,
            reasoning="Off-topic question",
            domain_relevance=0.05,
        )
        mock_llm = _make_sync_llm(mock_result)

        result = classify_intent("오늘 날씨 어때?", llm=mock_llm)

        assert result.intent == "OUT_OF_DOMAIN"
        assert result.domain_relevance == 0.05

    def test_all_intent_fields_preserved(self):
        """IntentClassification의 모든 필드가 보존됨"""
        mock_result = IntentClassification(
            intent="EXPLORATORY",
            confidence=0.7,
            reasoning="User is exploring the topic",
            domain_relevance=0.75,
        )
        mock_llm = _make_sync_llm(mock_result)

        result = classify_intent("딥러닝에 대해 전반적으로 알고 싶어", llm=mock_llm)

        assert result.intent == "EXPLORATORY"
        assert result.confidence == 0.7
        assert result.reasoning == "User is exploring the topic"
        assert result.domain_relevance == 0.75


# ============================================================================
# B-5-2: classify_intent() LLM Exception → 안전 fallback
# ============================================================================


class TestClassifyIntentSyncFallback:
    """classify_intent() 에러 시 fallback 테스트"""

    def test_llm_exception_returns_safe_fallback(self):
        """LLM 예외 → SIMPLE_QA fallback 반환 (예외 미전파)"""
        mock_llm = _make_error_sync_llm()

        result = classify_intent("질문", llm=mock_llm)

        assert isinstance(result, IntentClassification)
        assert result.intent == "SIMPLE_QA"

    def test_fallback_confidence_is_0_5(self):
        """fallback confidence=0.5"""
        mock_llm = _make_error_sync_llm()

        result = classify_intent("질문", llm=mock_llm)

        assert result.confidence == 0.5

    def test_fallback_domain_relevance_is_1_0(self):
        """fallback domain_relevance=1.0 (안전한 기본값)"""
        mock_llm = _make_error_sync_llm()

        result = classify_intent("질문", llm=mock_llm)

        assert result.domain_relevance == 1.0

    def test_fallback_reasoning_contains_error_info(self):
        """fallback reasoning에 에러 정보 포함"""
        mock_llm = _make_error_sync_llm(Exception("API timeout"))

        result = classify_intent("질문", llm=mock_llm)

        assert "error" in result.reasoning.lower() or "classification" in result.reasoning.lower()

    def test_network_error_also_returns_fallback(self):
        """네트워크 에러도 fallback 반환"""
        mock_llm = _make_error_sync_llm(ConnectionError("Network unreachable"))

        result = classify_intent("질문", llm=mock_llm)

        assert isinstance(result, IntentClassification)
        assert result.intent == "SIMPLE_QA"


# ============================================================================
# B-5-3: analyze_query() Mock LLM → QueryAnalysis 반환
# ============================================================================


class TestAnalyzeQuerySync:
    """analyze_query() 동기 버전 테스트"""

    def test_returns_query_analysis_instance(self):
        """mock LLM → QueryAnalysis 인스턴스 반환"""
        mock_result = QueryAnalysis(
            clarity_score=0.8,
            specificity_score=0.7,
            searchability_score=0.9,
            improved_queries=["PyTorch란?", "PyTorch 딥러닝 프레임워크 소개"],
        )
        mock_llm = _make_sync_llm(mock_result)

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query("PyTorch란?", intent="SIMPLE_QA", llm=mock_llm)

        assert isinstance(result, QueryAnalysis)
        assert result.clarity_score == 0.8
        assert result.specificity_score == 0.7
        assert len(result.improved_queries) == 2

    def test_improved_queries_preserved(self):
        """improved_queries 목록이 올바르게 전달됨"""
        queries = ["CNN이란?", "Convolutional Neural Network 설명", "CNN 구조와 동작 원리"]
        mock_result = QueryAnalysis(
            clarity_score=0.9,
            specificity_score=0.85,
            searchability_score=0.95,
            improved_queries=queries,
        )
        mock_llm = _make_sync_llm(mock_result)

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query("CNN이란?", intent="SIMPLE_QA", llm=mock_llm)

        assert result.improved_queries == queries

    def test_with_structured_output_called_with_query_analysis(self):
        """with_structured_output(QueryAnalysis)이 호출됨"""
        mock_result = QueryAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            searchability_score=0.5,
        )
        mock_llm = _make_sync_llm(mock_result)

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            analyze_query("질문", intent="SIMPLE_QA", llm=mock_llm)

        mock_llm.with_structured_output.assert_called_once_with(QueryAnalysis)

    def test_retrieval_filters_extracted(self):
        """retrieval_filters가 올바르게 반환됨"""
        mock_result = QueryAnalysis(
            clarity_score=0.8,
            specificity_score=0.8,
            searchability_score=0.8,
            retrieval_filters=QueryRetrievalFilters(
                doc_type=["pdf"],
                course=["CV 이론"],
                filter_confidence=0.9,
            ),
        )
        mock_llm = _make_sync_llm(mock_result)

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query(
                "CV 강의자료에서 CNN 설명 찾아줘", intent="SIMPLE_QA", llm=mock_llm
            )

        assert result.retrieval_filters is not None
        assert result.retrieval_filters.doc_type == ["pdf"]
        assert result.retrieval_filters.course == ["CV 이론"]


# ============================================================================
# B-5-4: analyze_query() LLM Exception → 안전 fallback
# ============================================================================


class TestAnalyzeQuerySyncFallback:
    """analyze_query() 에러 시 fallback 테스트"""

    def test_llm_exception_returns_default_analysis(self):
        """LLM 예외 → 기본 QueryAnalysis 반환"""
        mock_llm = _make_error_sync_llm()

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query("질문", intent="SIMPLE_QA", llm=mock_llm)

        assert isinstance(result, QueryAnalysis)

    def test_fallback_uses_original_question_as_query(self):
        """fallback 시 원본 질문을 improved_queries에 사용"""
        mock_llm = _make_error_sync_llm()
        question = "원본 질문입니다"

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query(question, intent="SIMPLE_QA", llm=mock_llm)

        assert question in result.improved_queries

    def test_fallback_scores_are_0_5(self):
        """fallback 점수는 모두 0.5"""
        mock_llm = _make_error_sync_llm()

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.get_prompt",
            return_value=_make_fake_prompt_template(),
        ):
            result = analyze_query("질문", intent="SIMPLE_QA", llm=mock_llm)

        assert result.clarity_score == 0.5
        assert result.specificity_score == 0.5
        assert result.searchability_score == 0.5


# ============================================================================
# B-5-5: aclassify_intent() async 버전
# ============================================================================


class TestClassifyIntentAsync:
    """aclassify_intent() 비동기 버전 테스트"""

    def test_async_returns_intent_classification(self):
        """async mock LLM → IntentClassification 반환"""
        mock_result = IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.88,
            reasoning="PyTorch question",
            domain_relevance=0.9,
        )
        mock_llm = _make_async_llm(mock_result)

        result = asyncio.run(aclassify_intent("PyTorch 사용법", llm=mock_llm))

        assert isinstance(result, IntentClassification)
        assert result.intent == "SIMPLE_QA"
        assert result.confidence == 0.88

    def test_async_with_structured_output_called(self):
        """ainvoke()가 호출됨"""
        mock_result = IntentClassification(
            intent="COMPLEX_REASONING",
            confidence=0.9,
            reasoning="Complex",
            domain_relevance=0.85,
        )
        mock_llm = _make_async_llm(mock_result)

        asyncio.run(aclassify_intent("복잡한 ML 질문", llm=mock_llm))

        mock_llm.with_structured_output.assert_called_once_with(IntentClassification)

    def test_async_exception_returns_fallback(self):
        """async LLM 예외 → SIMPLE_QA fallback"""
        mock_llm = _make_error_async_llm()

        result = asyncio.run(aclassify_intent("질문", llm=mock_llm))

        assert isinstance(result, IntentClassification)
        assert result.intent == "SIMPLE_QA"

    def test_async_fallback_has_safe_defaults(self):
        """async fallback의 안전한 기본값"""
        mock_llm = _make_error_async_llm(Exception("Async error"))

        result = asyncio.run(aclassify_intent("질문", llm=mock_llm))

        assert result.confidence == 0.5
        assert result.domain_relevance == 1.0

    def test_async_out_of_domain_result(self):
        """async OUT_OF_DOMAIN 결과 반환"""
        mock_result = IntentClassification(
            intent="OUT_OF_DOMAIN",
            confidence=0.99,
            reasoning="Completely off-topic",
            domain_relevance=0.02,
        )
        mock_llm = _make_async_llm(mock_result)

        result = asyncio.run(aclassify_intent("오늘 날씨?", llm=mock_llm))

        assert result.intent == "OUT_OF_DOMAIN"
        assert result.domain_relevance == 0.02


# ============================================================================
# B-5-6: aanalyze_query() async 버전
# ============================================================================


class TestAnalyzeQueryAsync:
    """aanalyze_query() 비동기 버전 테스트"""

    def test_async_returns_query_analysis(self):
        """async mock LLM → QueryAnalysis 반환"""
        mock_result = QueryAnalysis(
            clarity_score=0.85,
            specificity_score=0.75,
            searchability_score=0.9,
            improved_queries=["Transformer란?", "Transformer 구조 설명"],
        )
        mock_llm = _make_async_llm(mock_result)

        result = asyncio.run(aanalyze_query("Transformer란?", intent="SIMPLE_QA", llm=mock_llm))

        assert isinstance(result, QueryAnalysis)
        assert result.clarity_score == 0.85
        assert len(result.improved_queries) == 2

    def test_async_with_structured_output_called(self):
        """ainvoke()가 호출됨"""
        mock_result = QueryAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            searchability_score=0.5,
        )
        mock_llm = _make_async_llm(mock_result)

        asyncio.run(aanalyze_query("질문", intent="SIMPLE_QA", llm=mock_llm))

        mock_llm.with_structured_output.assert_called_once_with(QueryAnalysis)

    def test_async_exception_returns_fallback(self):
        """async LLM 예외 → 기본 QueryAnalysis fallback"""
        mock_llm = _make_error_async_llm()
        question = "비동기 에러 테스트 질문"

        result = asyncio.run(aanalyze_query(question, intent="SIMPLE_QA", llm=mock_llm))

        assert isinstance(result, QueryAnalysis)
        assert question in result.improved_queries

    def test_async_data_source_context_optional(self):
        """data_source_context=None → 정상 동작"""
        mock_result = QueryAnalysis(
            clarity_score=0.7,
            specificity_score=0.7,
            searchability_score=0.7,
        )
        mock_llm = _make_async_llm(mock_result)

        result = asyncio.run(
            aanalyze_query("질문", intent="SIMPLE_QA", llm=mock_llm, data_source_context=None)
        )

        assert isinstance(result, QueryAnalysis)

    def test_async_data_source_context_injected(self):
        """data_source_context 전달 시 ainvoke 호출됨"""
        mock_result = QueryAnalysis(
            clarity_score=0.8,
            specificity_score=0.8,
            searchability_score=0.8,
        )
        mock_llm = _make_async_llm(mock_result)

        context = "## Available Data Sources\n- PDF 강의자료\n- Slack Q&A"
        result = asyncio.run(
            aanalyze_query("질문", intent="SIMPLE_QA", llm=mock_llm, data_source_context=context)
        )

        assert isinstance(result, QueryAnalysis)
        # ainvoke가 호출되어야 함
        mock_llm.with_structured_output.return_value.ainvoke.assert_called_once()

    def test_async_filter_confidence_in_result(self):
        """filter_confidence 필드가 결과에 포함됨"""
        mock_result = QueryAnalysis(
            clarity_score=0.9,
            specificity_score=0.9,
            searchability_score=0.9,
            retrieval_filters=QueryRetrievalFilters(
                filter_confidence=0.85,
            ),
        )
        mock_llm = _make_async_llm(mock_result)

        result = asyncio.run(aanalyze_query("CV 강의 관련 질문", intent="SIMPLE_QA", llm=mock_llm))

        assert result.retrieval_filters is not None
        assert result.retrieval_filters.filter_confidence == 0.85
