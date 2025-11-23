"""
워크플로 통합 테스트.

LangGraph 워크플로 전체의 동작을 검증합니다:
- Happy path (모든 노드 성공)
- Error path (노드 실패 시 fallback)
- Routing logic (조건부 분기)
- Retry behavior (재시도 로직)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph
from naver_connect_chatbot.service.agents import (
    IntentClassification,
    QueryAnalysis,
    DocumentEvaluation,
    AnswerValidation,
    CorrectionStrategy,
)
from naver_connect_chatbot.service.agents.answer_generator import AnswerOutput


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    return AsyncMock()


@pytest.fixture
def mock_retriever():
    """Mock retriever with default document return"""
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="Test document content 1", metadata={"source": "test1"}),
        Document(page_content="Test document content 2", metadata={"source": "test2"}),
    ]
    return retriever


class TestWorkflowHappyPath:
    """워크플로 Happy Path 테스트 - 모든 노드가 성공하는 경우"""

    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, mock_llm, mock_retriever):
        """전체 워크플로가 성공적으로 완료되는 경우"""
        # Mock agent responses for each node
        with patch("naver_connect_chatbot.service.graph.nodes.create_intent_classifier") as mock_intent, \
             patch("naver_connect_chatbot.service.graph.nodes.create_query_analyzer") as mock_query, \
             patch("naver_connect_chatbot.service.graph.nodes.create_document_evaluator") as mock_doc_eval, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_generator") as mock_answer_gen, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_validator") as mock_validator:

            # Intent classifier mock
            intent_agent = AsyncMock()
            intent_agent.ainvoke.return_value = {
                "messages": [],
                "tool_calls": [],
                "content": IntentClassification(
                    intent="SIMPLE_QA",
                    confidence=0.9,
                    reasoning="Clear simple question"
                )
            }
            mock_intent.return_value = intent_agent

            # Query analyzer mock
            query_agent = AsyncMock()
            query_agent.ainvoke.return_value = {
                "messages": [],
                "content": QueryAnalysis(
                    clarity_score=0.8,
                    specificity_score=0.9,
                    searchability_score=0.85,
                    improved_queries=["test query 1", "test query 2"],
                    issues=[],
                    recommendations=[]
                )
            }
            mock_query.return_value = query_agent

            # Document evaluator mock
            doc_eval_agent = AsyncMock()
            doc_eval_agent.ainvoke.return_value = {
                "messages": [],
                "content": DocumentEvaluation(
                    sufficient=True,
                    relevant_count=2,
                    irrelevant_count=0,
                    confidence=0.9
                )
            }
            mock_doc_eval.return_value = doc_eval_agent

            # Answer generator mock
            answer_gen_agent = AsyncMock()
            answer_gen_agent.ainvoke.return_value = {
                "messages": [],
                "content": AnswerOutput(answer="This is a test answer generated from the documents.")
            }
            mock_answer_gen.return_value = answer_gen_agent

            # Answer validator mock
            validator_agent = AsyncMock()
            validator_agent.ainvoke.return_value = {
                "messages": [],
                "content": AnswerValidation(
                    has_hallucination=False,
                    is_grounded=True,
                    is_complete=True,
                    quality_score=0.95,
                    issues=[]
                )
            }
            mock_validator.return_value = validator_agent

            # Build and run workflow
            graph = build_adaptive_rag_graph(mock_retriever, mock_llm)
            result = await graph.ainvoke({
                "question": "Test question",
                "max_retries": 2
            })

            # Verify workflow completed successfully
            assert result["answer"] == "This is a test answer generated from the documents."
            assert result["intent"] == "SIMPLE_QA"
            assert result["sufficient_context"] is True
            assert result.get("is_grounded") is True
            assert result.get("quality_score") == 0.95


class TestWorkflowErrorHandling:
    """워크플로 에러 핸들링 테스트"""

    @pytest.mark.asyncio
    async def test_retrieval_failure_fallback(self, mock_llm, mock_retriever):
        """Retrieval 실패 시 빈 문서 리스트로 계속 진행"""
        # Make retriever raise an exception
        mock_retriever.invoke.side_effect = Exception("Retrieval failed")

        with patch("naver_connect_chatbot.service.graph.nodes.create_intent_classifier") as mock_intent, \
             patch("naver_connect_chatbot.service.graph.nodes.create_query_analyzer") as mock_query, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_generator") as mock_answer_gen:

            # Setup minimal mocks
            intent_agent = AsyncMock()
            intent_agent.ainvoke.return_value = {
                "messages": [],
                "content": IntentClassification(intent="SIMPLE_QA", confidence=0.9, reasoning="test")
            }
            mock_intent.return_value = intent_agent

            query_agent = AsyncMock()
            query_agent.ainvoke.return_value = {
                "messages": [],
                "content": QueryAnalysis(
                    clarity_score=0.8,
                    specificity_score=0.8,
                    searchability_score=0.8,
                    improved_queries=["test"],
                    issues=[],
                    recommendations=[]
                )
            }
            mock_query.return_value = query_agent

            answer_gen_agent = AsyncMock()
            answer_gen_agent.ainvoke.return_value = {
                "messages": [],
                "content": AnswerOutput(answer="Fallback answer without documents")
            }
            mock_answer_gen.return_value = answer_gen_agent

            # Build and run workflow
            graph = build_adaptive_rag_graph(mock_retriever, mock_llm)
            result = await graph.ainvoke({
                "question": "Test question",
                "max_retries": 2
            })

            # Verify workflow completed despite retrieval failure
            assert result["answer"]
            assert len(result.get("documents", [])) == 0  # No documents retrieved


class TestWorkflowRetryLogic:
    """워크플로 재시도 로직 테스트"""

    @pytest.mark.asyncio
    async def test_insufficient_documents_retry(self, mock_llm, mock_retriever):
        """문서 부족 시 재시도 후 최종 답변 생성"""
        with patch("naver_connect_chatbot.service.graph.nodes.create_intent_classifier") as mock_intent, \
             patch("naver_connect_chatbot.service.graph.nodes.create_query_analyzer") as mock_query, \
             patch("naver_connect_chatbot.service.graph.nodes.create_document_evaluator") as mock_doc_eval, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_generator") as mock_answer_gen:

            # Intent classifier mock
            intent_agent = AsyncMock()
            intent_agent.ainvoke.return_value = {
                "messages": [],
                "content": IntentClassification(intent="SIMPLE_QA", confidence=0.9, reasoning="test")
            }
            mock_intent.return_value = intent_agent

            # Query analyzer mock
            query_agent = AsyncMock()
            query_agent.ainvoke.return_value = {
                "messages": [],
                "content": QueryAnalysis(
                    clarity_score=0.7,
                    specificity_score=0.7,
                    searchability_score=0.7,
                    improved_queries=["refined query"],
                    issues=[],
                    recommendations=[]
                )
            }
            mock_query.return_value = query_agent

            # Document evaluator mock - first insufficient, then sufficient
            doc_eval_agent = AsyncMock()
            doc_eval_agent.ainvoke.side_effect = [
                {  # First call: insufficient
                    "messages": [],
                    "content": DocumentEvaluation(
                        sufficient=False,
                        relevant_count=0,
                        irrelevant_count=2,
                        confidence=0.4
                    )
                },
                {  # Second call: sufficient
                    "messages": [],
                    "content": DocumentEvaluation(
                        sufficient=True,
                        relevant_count=2,
                        irrelevant_count=0,
                        confidence=0.9
                    )
                }
            ]
            mock_doc_eval.return_value = doc_eval_agent

            # Answer generator mock
            answer_gen_agent = AsyncMock()
            answer_gen_agent.ainvoke.return_value = {
                "messages": [],
                "content": AnswerOutput(answer="Final answer after retry")
            }
            mock_answer_gen.return_value = answer_gen_agent

            # Build and run workflow
            graph = build_adaptive_rag_graph(mock_retriever, mock_llm)
            result = await graph.ainvoke({
                "question": "Test question",
                "max_retries": 2
            })

            # Verify retry logic worked
            assert result["answer"] == "Final answer after retry"
            assert result.get("retry_count", 0) >= 1  # At least one retry happened


class TestWorkflowRouting:
    """워크플로 라우팅 로직 테스트"""

    @pytest.mark.asyncio
    async def test_routing_to_correction(self, mock_llm, mock_retriever):
        """답변 품질 낮을 시 교정 단계로 라우팅"""
        with patch("naver_connect_chatbot.service.graph.nodes.create_intent_classifier") as mock_intent, \
             patch("naver_connect_chatbot.service.graph.nodes.create_query_analyzer") as mock_query, \
             patch("naver_connect_chatbot.service.graph.nodes.create_document_evaluator") as mock_doc_eval, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_generator") as mock_answer_gen, \
             patch("naver_connect_chatbot.service.graph.nodes.create_answer_validator") as mock_validator, \
             patch("naver_connect_chatbot.service.graph.nodes.create_corrector") as mock_corrector:

            # Setup mocks (abbreviated for clarity)
            intent_agent = AsyncMock()
            intent_agent.ainvoke.return_value = {
                "messages": [],
                "content": IntentClassification(intent="SIMPLE_QA", confidence=0.9, reasoning="test")
            }
            mock_intent.return_value = intent_agent

            query_agent = AsyncMock()
            query_agent.ainvoke.return_value = {
                "messages": [],
                "content": QueryAnalysis(
                    clarity_score=0.8,
                    specificity_score=0.8,
                    searchability_score=0.8,
                    improved_queries=["test"],
                    issues=[],
                    recommendations=[]
                )
            }
            mock_query.return_value = query_agent

            doc_eval_agent = AsyncMock()
            doc_eval_agent.ainvoke.return_value = {
                "messages": [],
                "content": DocumentEvaluation(
                    sufficient=True,
                    relevant_count=2,
                    irrelevant_count=0,
                    confidence=0.9
                )
            }
            mock_doc_eval.return_value = doc_eval_agent

            answer_gen_agent = AsyncMock()
            answer_gen_call_count = [0]  # Use list for mutable counter

            def answer_gen_side_effect(*args, **kwargs):
                answer_gen_call_count[0] += 1
                if answer_gen_call_count[0] == 1:
                    # First call: low quality answer
                    return {
                        "messages": [],
                        "content": AnswerOutput(answer="Poor quality answer")
                    }
                else:
                    # After correction: better answer
                    return {
                        "messages": [],
                        "content": AnswerOutput(answer="Improved answer after correction")
                    }

            answer_gen_agent.ainvoke.side_effect = answer_gen_side_effect
            mock_answer_gen.return_value = answer_gen_agent

            # Validator mock - first low quality, then high quality
            validator_agent = AsyncMock()
            validator_call_count = [0]

            def validator_side_effect(*args, **kwargs):
                validator_call_count[0] += 1
                if validator_call_count[0] == 1:
                    # First validation: low quality
                    return {
                        "messages": [],
                        "content": AnswerValidation(
                            has_hallucination=True,
                            is_grounded=False,
                            is_complete=False,
                            quality_score=0.3,
                            issues=["Contains hallucination"]
                        )
                    }
                else:
                    # Second validation: high quality
                    return {
                        "messages": [],
                        "content": AnswerValidation(
                            has_hallucination=False,
                            is_grounded=True,
                            is_complete=True,
                            quality_score=0.9,
                            issues=[]
                        )
                    }

            validator_agent.ainvoke.side_effect = validator_side_effect
            mock_validator.return_value = validator_agent

            # Corrector mock
            corrector_agent = AsyncMock()
            corrector_agent.ainvoke.return_value = {
                "messages": [],
                "content": CorrectionStrategy(
                    action="REGENERATE",
                    feedback="Remove hallucinated content",
                    priority="HIGH"
                )
            }
            mock_corrector.return_value = corrector_agent

            # Build and run workflow
            graph = build_adaptive_rag_graph(mock_retriever, mock_llm)
            result = await graph.ainvoke({
                "question": "Test question",
                "max_retries": 2
            })

            # Verify correction happened
            assert result["answer"] == "Improved answer after correction"
            assert result.get("correction_count", 0) >= 1
            assert result.get("quality_score", 0) >= 0.8  # Final quality should be high
