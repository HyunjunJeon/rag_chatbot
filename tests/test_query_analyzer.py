"""
Query Analyzer Agent 테스트.

emit_query_analysis 도구를 사용한 구조화 출력을 검증합니다.
- 도구 기반 QueryAnalysis 반환
- ToolMessage 파싱 로직
- 폴백 메커니즘
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from naver_connect_chatbot.service.agents.query_analyzer import (
    QueryAnalysis,
    create_query_analyzer,
    emit_query_analysis_result,
    analyze_query,
)


class TestEmitQueryAnalysisTool:
    """emit_query_analysis 도구의 기본 동작을 검증합니다."""

    def test_tool_returns_query_analysis_instance(self):
        """도구가 QueryAnalysis 인스턴스를 반환하는지 확인합니다."""
        # @tool 데코레이터로 인해 .invoke()를 사용해야 합니다
        result = emit_query_analysis_result.invoke({
            "clarity_score": 0.8,
            "specificity_score": 0.7,
            "searchability_score": 0.9,
            "improved_queries": ["개선된 질문 1", "개선된 질문 2"],
            "issues": ["문제점 1"],
            "recommendations": ["권장사항 1", "권장사항 2"],
        })

        assert isinstance(result, QueryAnalysis)
        assert result.clarity_score == 0.8
        assert result.specificity_score == 0.7
        assert result.searchability_score == 0.9
        assert len(result.improved_queries) == 2
        assert len(result.issues) == 1
        assert len(result.recommendations) == 2

    def test_tool_validation_constraints(self):
        """도구가 점수 범위 검증을 수행하는지 확인합니다."""
        # 정상 범위 (0.0 ~ 1.0)
        result = emit_query_analysis_result.invoke({
            "clarity_score": 0.0,
            "specificity_score": 0.5,
            "searchability_score": 1.0,
            "improved_queries": ["테스트"],
            "issues": [],
            "recommendations": [],
        })
        assert result.clarity_score == 0.0
        assert result.specificity_score == 0.5
        assert result.searchability_score == 1.0

    def test_tool_handles_empty_lists(self):
        """도구가 빈 리스트를 허용하는지 확인합니다."""
        result = emit_query_analysis_result.invoke({
            "clarity_score": 0.5,
            "specificity_score": 0.5,
            "searchability_score": 0.5,
            "improved_queries": ["최소 1개 필요"],
            "issues": [],
            "recommendations": [],
        })
        assert isinstance(result, QueryAnalysis)
        assert len(result.issues) == 0
        assert len(result.recommendations) == 0


class TestQueryAnalysisModel:
    """QueryAnalysis Pydantic 모델의 검증 로직을 테스트합니다."""

    def test_valid_query_analysis_creation(self):
        """유효한 QueryAnalysis 인스턴스를 생성할 수 있는지 확인합니다."""
        analysis = QueryAnalysis(
            clarity_score=0.8,
            specificity_score=0.9,
            searchability_score=0.7,
            improved_queries=["질문 1", "질문 2"],
            issues=["이슈 1"],
            recommendations=["권장 1"],
        )

        assert analysis.clarity_score == 0.8
        assert analysis.specificity_score == 0.9
        assert analysis.searchability_score == 0.7

    def test_score_range_validation(self):
        """점수가 0.0 ~ 1.0 범위를 벗어나면 오류가 발생하는지 확인합니다."""
        with pytest.raises(Exception):  # Pydantic validation error
            QueryAnalysis(
                clarity_score=1.5,  # 범위 초과
                specificity_score=0.5,
                searchability_score=0.5,
                improved_queries=["test"],
                issues=[],
                recommendations=[],
            )

        with pytest.raises(Exception):
            QueryAnalysis(
                clarity_score=0.5,
                specificity_score=-0.1,  # 범위 미만
                searchability_score=0.5,
                improved_queries=["test"],
                issues=[],
                recommendations=[],
            )

    def test_default_empty_lists(self):
        """리스트 필드가 기본값으로 빈 리스트를 가지는지 확인합니다."""
        analysis = QueryAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            searchability_score=0.5,
        )

        assert analysis.improved_queries == []
        assert analysis.issues == []
        assert analysis.recommendations == []


class TestAnalyzeQueryFunction:
    """analyze_query 편의 함수의 동작을 검증합니다."""

    def test_analyze_query_with_mock_llm_returning_tool_message(self):
        """
        ToolMessage를 반환하는 Mock LLM으로 analyze_query가 정상 작동하는지 확인합니다.
        
        실제 에이전트에서는 도구가 반환한 QueryAnalysis 객체가 ToolMessage의
        content로 전달됩니다.
        """
        # Mock LLM 설정
        mock_llm = MagicMock()

        # 예상되는 QueryAnalysis 결과
        expected_analysis = QueryAnalysis(
            clarity_score=0.4,
            specificity_score=0.3,
            searchability_score=0.5,
            improved_queries=["PyTorch란 무엇인가?", "PyTorch의 개념을 설명해주세요"],
            issues=["주어가 모호함", "구체성 부족"],
            recommendations=["명확한 주제 명시", "질문 범위 구체화"],
        )

        # Mock ToolMessage - content가 QueryAnalysis 인스턴스
        mock_tool_msg = MagicMock()
        mock_tool_msg.__class__.__name__ = "ToolMessage"
        mock_tool_msg.content = expected_analysis

        # Mock agent 응답 구성
        mock_agent_response = {
            "messages": [
                HumanMessage(content="question: What is it?\nintent: CLARIFICATION_NEEDED"),
                AIMessage(content="Analyzing..."),
                mock_tool_msg,
            ]
        }

        # create_query_analyzer를 Mock으로 대체
        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.create_query_analyzer"
        ) as mock_create_analyzer:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_agent_response
            mock_create_analyzer.return_value = mock_agent

            # 실행
            result = analyze_query(
                question="What is it?", intent="CLARIFICATION_NEEDED", llm=mock_llm
            )

            # 검증
            assert isinstance(result, QueryAnalysis)
            assert result.clarity_score == 0.4
            assert result.specificity_score == 0.3
            assert result.searchability_score == 0.5
            assert len(result.improved_queries) == 2
            assert "PyTorch란 무엇인가?" in result.improved_queries

    def test_analyze_query_fallback_on_unexpected_response(self):
        """예상하지 못한 응답 형식일 때 폴백이 작동하는지 확인합니다."""
        mock_llm = MagicMock()

        # 예상 외 응답 (ToolMessage 없음)
        mock_agent_response = {
            "messages": [
                HumanMessage(content="question: Test\nintent: SIMPLE_QA"),
                AIMessage(content="Unexpected response"),
            ]
        }

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.create_query_analyzer"
        ) as mock_create_analyzer:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_agent_response
            mock_create_analyzer.return_value = mock_agent

            # 실행
            result = analyze_query(question="Test", intent="SIMPLE_QA", llm=mock_llm)

            # 폴백 검증
            assert isinstance(result, QueryAnalysis)
            assert result.clarity_score == 0.5
            assert result.specificity_score == 0.5
            assert result.searchability_score == 0.5
            assert result.improved_queries == ["Test"]
            assert "Unable to analyze query" in result.issues

    def test_analyze_query_handles_dict_content_in_tool_message(self):
        """
        ToolMessage의 content가 dict인 경우 QueryAnalysis로 변환되는지 확인합니다.
        
        일부 시나리오에서 content가 dict 형태로 전달될 수 있습니다.
        """
        mock_llm = MagicMock()

        # Mock ToolMessage - content가 dict
        mock_tool_msg = MagicMock()
        mock_tool_msg.__class__.__name__ = "ToolMessage"
        mock_tool_msg.content = {
            "clarity_score": 0.9,
            "specificity_score": 0.8,
            "searchability_score": 0.85,
            "improved_queries": ["개선된 질문"],
            "issues": [],
            "recommendations": ["권장사항"],
        }

        # content가 dict인 ToolMessage
        mock_agent_response = {
            "messages": [
                HumanMessage(content="question: Test\nintent: SIMPLE_QA"),
                mock_tool_msg,
            ]
        }

        with patch(
            "naver_connect_chatbot.service.agents.query_analyzer.create_query_analyzer"
        ) as mock_create_analyzer:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_agent_response
            mock_create_analyzer.return_value = mock_agent

            result = analyze_query(question="Test", intent="SIMPLE_QA", llm=mock_llm)

            assert isinstance(result, QueryAnalysis)
            assert result.clarity_score == 0.9
            assert result.specificity_score == 0.8
            assert result.searchability_score == 0.85


class TestCreateQueryAnalyzer:
    """create_query_analyzer 팩토리 함수를 검증합니다."""

    def test_create_query_analyzer_returns_agent(self):
        """에이전트가 정상적으로 생성되는지 확인합니다."""
        mock_llm = MagicMock()

        # LangChain의 create_agent를 Mock
        with patch("naver_connect_chatbot.service.agents.query_analyzer.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_query_analyzer(mock_llm)

            # create_agent가 호출되었는지 확인
            assert mock_create.called
            call_kwargs = mock_create.call_args.kwargs

            # tools에 emit_query_analysis가 포함되어 있는지 확인
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) > 0

            # system_prompt에 도구 사용 가이드가 포함되어 있는지 확인
            assert "system_prompt" in call_kwargs
            assert "emit_query_analysis" in call_kwargs["system_prompt"]

            assert agent == mock_agent

    def test_create_query_analyzer_logs_on_error(self):
        """에이전트 생성 중 오류 발생 시 로깅되는지 확인합니다."""
        mock_llm = MagicMock()

        with patch("naver_connect_chatbot.service.agents.query_analyzer.create_agent") as mock_create:
            mock_create.side_effect = Exception("Test error")

            with pytest.raises(Exception, match="Test error"):
                create_query_analyzer(mock_llm)


def _has_llm_available():
    """LLM이 사용 가능한지 확인합니다."""
    try:
        from naver_connect_chatbot.config.settings.main import settings
        return (
            (settings.openai.enabled and settings.openai.api_key) or
            (settings.openrouter.enabled and settings.openrouter.api_key)
        )
    except Exception:
        return False


class TestIntegrationWithRealLLM:
    """실제 LLM과의 통합 테스트 (선택적)."""

    @pytest.mark.skipif(
        not _has_llm_available(),
        reason="LLM이 설정되지 않았습니다 (OpenAI 또는 OpenRouter API 키 필요)"
    )
    def test_analyze_query_with_real_llm(self):
        """
        실제 LLM을 사용한 통합 테스트 - analyze_query 함수.
        
        OPENAI_API_KEY 환경 변수가 설정되어 있을 때만 실행됩니다.
        """
        from naver_connect_chatbot.config.llm import get_chat_model, LLMProvider

        llm = get_chat_model(LLMProvider.OPENAI)

        result = analyze_query(
            question="그게 뭐야?", intent="CLARIFICATION_NEEDED", llm=llm
        )

        # 기본 검증
        assert isinstance(result, QueryAnalysis)
        assert 0.0 <= result.clarity_score <= 1.0
        assert 0.0 <= result.specificity_score <= 1.0
        assert 0.0 <= result.searchability_score <= 1.0
        assert len(result.improved_queries) > 0
        
        # 폴백이 아닌 실제 분석 결과인지 확인
        assert result.improved_queries != ["그게 뭐야?"]
        assert result.issues != ["Unable to analyze query"]
        
        # 결과 출력 (디버깅용)
        print("\n=== Query Analysis Result ===")
        print(f"Clarity: {result.clarity_score}")
        print(f"Specificity: {result.specificity_score}")
        print(f"Searchability: {result.searchability_score}")
        print(f"Improved Queries: {result.improved_queries}")
        print(f"Issues: {result.issues}")
        print(f"Recommendations: {result.recommendations}")

    @pytest.mark.skipif(
        not _has_llm_available(),
        reason="LLM이 설정되지 않았습니다 (OpenAI 또는 OpenRouter API 키 필요)"
    )
    def test_agent_tool_calling_e2e(self):
        """
        실제 에이전트 실행 E2E 테스트 - 도구 호출 검증.
        
        에이전트가 emit_query_analysis_result 도구를 실제로 호출하고,
        ToolMessage를 통해 결과를 반환하는지 검증합니다.
        """
        from naver_connect_chatbot.config.llm import get_chat_model, LLMProvider

        # 실제 LLM으로 에이전트 생성
        llm = get_chat_model(LLMProvider.OPENAI)
        analyzer = create_query_analyzer(llm)

        # 에이전트 직접 실행
        response = analyzer.invoke({
            "messages": [{
                "role": "user",
                "content": "question: 그게 뭐야?\nintent: CLARIFICATION_NEEDED"
            }]
        })

        # 응답 구조 검증
        assert isinstance(response, dict)
        assert "messages" in response
        
        messages = response["messages"]
        assert len(messages) > 0
        
        # ToolMessage 존재 여부 확인
        tool_messages = [
            msg for msg in messages
            if msg.__class__.__name__ == "ToolMessage"
        ]
        
        print("\n=== Agent Response Analysis ===")
        print(f"Total messages: {len(messages)}")
        print(f"Tool messages found: {len(tool_messages)}")
        
        for i, msg in enumerate(messages):
            print(f"Message {i}: {msg.__class__.__name__}")
            if hasattr(msg, "content"):
                content_preview = str(msg.content)[:100]
                print(f"  Content preview: {content_preview}")
        
        # ToolMessage가 최소 1개는 있어야 함
        assert len(tool_messages) > 0, "에이전트가 도구를 호출하지 않았습니다"
        
        # 마지막 ToolMessage 확인
        last_tool_msg = tool_messages[-1]
        tool_content = last_tool_msg.content
        
        print("\n=== Tool Call Result ===")
        print(f"Tool content type: {type(tool_content)}")
        print(f"Tool content: {tool_content}")
        
        # content가 QueryAnalysis, dict, 또는 string (직렬화된 형식)이어야 함
        assert isinstance(tool_content, (QueryAnalysis, dict, str)), \
            f"도구 반환값이 예상과 다릅니다: {type(tool_content)}"
        
        # 문자열인 경우 QueryAnalysis의 repr 형식인지 확인
        if isinstance(tool_content, str):
            assert "clarity_score=" in tool_content, \
                "도구 반환값이 QueryAnalysis 형식이 아닙니다"
        
        # analyze_query로 최종 파싱 테스트
        result = analyze_query(
            question="그게 뭐야?", 
            intent="CLARIFICATION_NEEDED", 
            llm=llm
        )
        
        # 최종 결과 검증
        assert isinstance(result, QueryAnalysis)
        assert 0.0 <= result.clarity_score <= 1.0
        assert 0.0 <= result.specificity_score <= 1.0
        assert 0.0 <= result.searchability_score <= 1.0
        assert len(result.improved_queries) > 0
        
        # 폴백이 아닌 실제 결과인지 확인
        assert result.improved_queries != ["그게 뭐야?"]
        
        print("\n=== Final Parsed Result ===")
        print(f"Clarity: {result.clarity_score}")
        print(f"Specificity: {result.specificity_score}")
        print(f"Searchability: {result.searchability_score}")
        print(f"Improved Queries: {result.improved_queries}")
        print(f"Issues: {result.issues}")
        print(f"Recommendations: {result.recommendations}")
        
        print("\n✅ E2E 테스트 성공: 에이전트가 도구를 호출하고 결과를 올바르게 파싱했습니다")

