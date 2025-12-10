"""
Adaptive RAG 워크플로에서 사용하는 노드 함수 집합.

각 노드는 RAG 프로세스의 개별 단계를 나타냅니다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage

from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.graph.types import (
    IntentUpdate,
    QueryAnalysisUpdate,
    RetrievalUpdate,
    AnswerUpdate,
    OODResponseUpdate,
)
from naver_connect_chatbot.service.agents.intent_classifier import (
    aclassify_intent,
    IntentClassification,
)
from naver_connect_chatbot.service.agents.query_analyzer import (
    aanalyze_query,
    QueryAnalysis,
)
from naver_connect_chatbot.service.agents.answer_generator import (
    get_generation_strategy,
)
from naver_connect_chatbot.service.tool.retrieval_tool import retrieve_documents_async, RetrievalResult
from naver_connect_chatbot.rag import ClovaStudioReranker
from naver_connect_chatbot.config import logger, settings


def _extract_text_response(response: Any) -> str:
    """
    LangChain 에이전트 응답에서 텍스트를 안전하게 추출합니다.
    """
    if isinstance(response, AIMessage):
        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)

    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, str):
            return output
        content_val = response.get("content")
        if isinstance(content_val, str):
            return content_val

    if isinstance(response, str):
        return response

    return str(response)


async def classify_intent_node(state: AdaptiveRAGState, llm: Runnable) -> IntentUpdate:
    """
    사용자 의도를 분류합니다 (OUT_OF_DOMAIN 감지 포함).

    Note:
        tools/function calling 대신 llm.invoke(prompt) 형태로 직접 호출하여
        CLOVA HCX-007의 reasoning 모드와 호환됩니다.

    매개변수:
        state: 현재 워크플로 상태
        llm: 분류에 사용할 언어 모델

    반환값:
        의도 분류 결과가 포함된 상태 업데이트 (domain_relevance 포함)
    """
    logger.info("---CLASSIFY INTENT---")
    question = state["question"]

    # aclassify_intent 직접 호출 (내부에서 에러 처리)
    response = await aclassify_intent(question, llm)

    # domain_relevance가 낮으면 OUT_OF_DOMAIN으로 보정
    intent = response.intent
    domain_relevance = response.domain_relevance

    if domain_relevance < 0.3 and intent != "OUT_OF_DOMAIN":
        logger.info(
            f"Low domain_relevance ({domain_relevance:.2f}), "
            f"overriding intent from {intent} to OUT_OF_DOMAIN"
        )
        intent = "OUT_OF_DOMAIN"

    if intent == "OUT_OF_DOMAIN":
        logger.info(
            f"OUT_OF_DOMAIN detected: domain_relevance={domain_relevance:.2f}, "
            f"question='{question[:50]}...'"
        )

    return {
        "intent": intent,
        "intent_confidence": response.confidence,
        "intent_reasoning": response.reasoning,
        "domain_relevance": domain_relevance,
    }


async def analyze_query_node(state: AdaptiveRAGState, llm: Runnable) -> QueryAnalysisUpdate:
    """
    질의 품질을 분석하고 다중 검색 쿼리 및 검색 필터를 생성합니다.

    이 노드는 Query Analysis, Multi-Query Generation, Filter Extraction을 통합하여:
    1. 질의의 명확성, 구체성, 검색 가능성을 평가
    2. 다양한 관점의 검색 쿼리 3-5개 생성 (Multi-Query)
    3. 질문에서 메타데이터 기반 검색 필터 추출 (doc_type, course, etc.)

    Pre-Retriever 데이터 소스 선택:
    - VectorDB 스키마 정보를 프롬프트에 주입하여 LLM이 실제 데이터 소스를 알고 필터를 추출할 수 있게 함
    - 서버 시작 시 로드된 SchemaRegistry에서 데이터 소스 컨텍스트를 가져옴

    Note:
        tools/function calling 대신 llm.invoke(prompt) 형태로 직접 호출하여
        CLOVA HCX-007의 reasoning 모드와 호환됩니다.

    매개변수:
        state: 현재 워크플로 상태
        llm: 분석 및 쿼리 생성에 사용할 언어 모델

    반환값:
        질의 분석, 다중 검색 쿼리, 검색 필터를 포함한 상태 업데이트
    """
    logger.info("---ANALYZE QUERY & GENERATE MULTI-QUERY & EXTRACT FILTERS---")
    question = state["question"]
    intent = state.get("intent", "SIMPLE_QA")

    try:
        # VectorDB 스키마 정보를 가져와 프롬프트에 주입
        data_source_context = None
        try:
            from naver_connect_chatbot.rag.schema_registry import (
                get_data_source_context,
                get_schema_registry,
            )

            data_source_context = get_data_source_context(max_courses=10)

            # 별칭 컨텍스트 추가 (VectorDB 기반 동적 생성)
            registry = get_schema_registry()
            if registry.is_loaded():
                alias_context = registry.get_alias_context_for_prompt()
                if alias_context:
                    data_source_context = f"{data_source_context}\n\n{alias_context}"

            logger.debug("Data source context with aliases loaded for query analysis")
        except Exception as e:
            logger.warning(f"Failed to load data source context: {e}")

        # aanalyze_query 직접 호출
        response = await aanalyze_query(question, intent, llm, data_source_context)

        # retrieval_filters를 RetrievalFilters TypedDict로 변환
        filters = {}
        if response.retrieval_filters:
            rf = response.retrieval_filters
            if rf.doc_type:
                filters["doc_type"] = rf.doc_type
            if rf.course:
                # Fuzzy + Alias 후처리로 과정명 확장
                try:
                    from naver_connect_chatbot.rag.schema_registry import get_schema_registry

                    registry = get_schema_registry()
                    if not registry.is_loaded():
                        logger.warning(
                            "SchemaRegistry not loaded, using original course names"
                        )
                        filters["course"] = rf.course
                    else:
                        resolved_courses: list[str] = []
                        for course in rf.course:
                            try:
                                resolved = registry.resolve_course_with_fuzzy(course)
                                resolved_courses.extend(resolved)
                            except Exception as course_error:
                                logger.error(
                                    f"Failed to resolve course '{course}': {course_error}",
                                    exc_info=True,
                                )
                                resolved_courses.append(course)  # Fallback to original
                        # 중복 제거 (순서 유지)
                        filters["course"] = list(dict.fromkeys(resolved_courses))
                        logger.info(
                            f"Course names resolved: {rf.course} → {filters['course']}"
                        )
                except Exception as e:
                    logger.error(
                        f"Critical error in course fuzzy resolution: {e}",
                        exc_info=True,
                    )
                    filters["course"] = rf.course
            if rf.course_topic:
                filters["course_topic"] = rf.course_topic
            if rf.generation:
                filters["generation"] = rf.generation

        if filters:
            logger.info(f"Extracted retrieval filters: {filters}")

        # filter_confidence 추출
        filter_confidence = 1.0
        if response.retrieval_filters:
            filter_confidence = response.retrieval_filters.filter_confidence
            if filter_confidence < 0.5:
                logger.info(
                    f"Low filter confidence ({filter_confidence:.2f}), "
                    "clarification may be needed"
                )

        # 분석 결과를 추출합니다.
        return {
            "query_analysis": {
                "clarity_score": response.clarity_score,
                "specificity_score": response.specificity_score,
                "searchability_score": response.searchability_score,
            },
            "refined_queries": response.improved_queries
            if response.improved_queries
            else [question],
            "original_query": question,
            "retrieval_filters": filters if filters else None,
            "filter_confidence": filter_confidence,
        }

    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        return {
            "query_analysis": {"error": str(e)},
            "refined_queries": [question],
            "original_query": question,
            "retrieval_filters": None,
            "filter_confidence": 1.0,  # 에러 시 기본값
        }


async def retrieve_node(state: AdaptiveRAGState, retriever: BaseRetriever) -> RetrievalUpdate:
    """
    문서를 검색하고 메타데이터 기반 필터를 적용합니다.

    매개변수:
        state: 현재 워크플로 상태
        retriever: 문서 검색기

    반환값:
        검색된 문서와 필터링 메타데이터를 포함한 상태 업데이트
    """
    logger.info("---RETRIEVE---")

    # 가능하면 정제된 질의를 사용하여 검색합니다.
    queries = state.get("refined_queries", [state["question"]])
    primary_query = queries[0] if queries else state["question"]

    # 상태에서 필터를 가져옵니다.
    filters = state.get("retrieval_filters")
    if filters:
        logger.info(f"Applying retrieval filters: {filters}")

    try:
        # 문서를 검색하고 필터를 적용합니다.
        result: RetrievalResult = await retrieve_documents_async(
            retriever,
            primary_query,
            filters=filters,
            fallback_on_empty=True,
            min_results=1,
        )

        logger.info(
            f"Retrieved {result.original_count} docs, "
            f"filtered to {result.filtered_count}, "
            f"filters_applied={result.filters_applied}, "
            f"fallback_used={result.fallback_used}"
        )

        return {
            "documents": result.documents,
            "context": result.documents,  # 하위 호환성 유지를 위한 필드
            "retrieval_strategy": "hybrid",
            "retrieval_filters_applied": result.filters_applied,
            "retrieval_fallback_used": result.fallback_used,
            "retrieval_metadata": {
                "original_count": result.original_count,
                "filtered_count": result.filtered_count,
                "filters": filters,
            },
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {
            "documents": [],
            "context": [],
            "retrieval_strategy": "hybrid",
            "retrieval_filters_applied": False,
            "retrieval_fallback_used": False,
        }


async def rerank_node(state: AdaptiveRAGState) -> dict[str, Any]:
    """
    검색된 문서를 Clova Studio Reranker로 재정렬합니다.

    Post-Retriever 단계로, 검색된 문서의 관련도를 재평가하여
    가장 관련성 높은 문서를 상위로 정렬합니다.

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        재정렬된 문서를 포함한 상태 업데이트
    """
    logger.info("---RERANK DOCUMENTS---")

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents to rerank")
        return {
            "documents": [],
            "context": [],
        }

    # Reranking 설정 확인
    use_reranking = (
        settings.adaptive_rag.use_reranking if hasattr(settings, "adaptive_rag") else True
    )

    if not use_reranking:
        logger.info("Reranking disabled, skipping")
        return {
            "documents": documents,
            "context": documents,
        }

    try:
        # Clova Studio Reranker 초기화
        reranker = ClovaStudioReranker(
            api_key=settings.clova_llm.api_key.get_secret_value()
            if settings.clova_llm.api_key
            else "",
            max_tokens=1024,
        )

        # Reranking 수행
        logger.info(f"Reranking {len(documents)} documents")
        reranked_docs = await reranker.arerank(
            query=question,
            documents=documents,
            top_k=min(len(documents), 10),  # 최대 10개까지 유지
        )

        logger.info(f"Reranked to {len(reranked_docs)} documents")

        return {
            "documents": reranked_docs,
            "context": reranked_docs,
        }

    except Exception as e:
        logger.error(f"Reranking error: {e}, using original documents")
        return {
            "documents": documents,
            "context": documents,
        }


async def generate_answer_node(state: AdaptiveRAGState, llm: Runnable) -> AnswerUpdate:
    """
    문맥을 기반으로 답변을 생성합니다 (Reasoning 모드 활용).

    CLOVA HCX-007 모델의 Reasoning 능력을 활용하여:
    1. 단계별 추론을 통해 답변 품질 향상
    2. 자체 검증을 통해 환각 방지
    3. 복잡한 질문에 대한 논리적 답변 생성

    매개변수:
        state: 현재 워크플로 상태
        llm: 생성에 사용할 언어 모델 (Reasoning 지원)

    반환값:
        생성된 답변을 포함한 상태 업데이트
    """
    logger.info("---GENERATE ANSWER (with Reasoning)---")
    question = state["question"]
    documents = state.get("documents", [])
    intent = state.get("intent", "SIMPLE_QA")

    try:
        # 사용할 생성 전략을 결정합니다.
        strategy = get_generation_strategy(intent)

        # Reasoning effort 설정 (intent 기반)
        # COMPLEX_REASONING: high, EXPLORATORY: medium, SIMPLE_QA: low
        thinking_effort = "medium"  # 기본값
        if intent == "COMPLEX_REASONING":
            thinking_effort = "high"
        elif intent == "SIMPLE_QA":
            thinking_effort = "low"
        elif intent == "EXPLORATORY":
            thinking_effort = "medium"

        logger.info(f"Using thinking_effort: {thinking_effort} for intent: {intent}")

        # 생성에 사용할 문맥을 포맷합니다.
        context_text = "\n\n".join(
            [f"[문서 {i + 1}]\n{doc.page_content}" for i, doc in enumerate(documents)]
        )

        if not context_text:
            context_text = "참고할 수 있는 문서가 없습니다."

        # 답변을 생성합니다. Reasoning이 활성화된 LLM을 직접 호출하며, tools는 사용하지 않습니다.
        prompt = (
            "당신은 Naver Boost Camp 학생들에게 AI/ML을 가르치는 조교입니다. "
            "주어진 문맥만을 근거로, 단계별로 사고한 뒤 한국어로 답변하세요.\n\n"
            f"question: {question}\n\ncontext:\n{context_text}"
        )

        response_raw = await llm.ainvoke(prompt)
        answer = _extract_text_response(response_raw)
        logger.info(f"Generated answer with reasoning: {answer[:100]}...")

        return {
            "answer": answer,
            "generation_metadata": {
                "strategy": strategy,
                "context_length": len(context_text),
                "thinking_effort": thinking_effort,
                "reasoning_enabled": True,
            },
            "generation_strategy": strategy,
        }

    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return {
            "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
            "generation_metadata": {"error": str(e)},
            "generation_strategy": "error",
        }


async def generate_ood_response_node(state: AdaptiveRAGState) -> OODResponseUpdate:
    """
    Out-of-Domain 질문에 대한 정중한 거절 응답을 생성합니다.

    AI/ML 교육과 무관한 질문 (날씨, 음식, 여행 등)에 대해
    정중히 거절하고 도움 가능한 영역을 안내합니다.

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        OOD 거절 응답을 포함한 상태 업데이트
    """
    logger.info("---GENERATE OOD RESPONSE---")

    question = state.get("question", "")
    domain_relevance = state.get("domain_relevance", 0.0)

    # 질문 내용을 간략히 요약 (너무 길면 자르기)
    question_preview = question[:50] + "..." if len(question) > 50 else question

    response = (
        f"죄송합니다. '{question_preview}'에 대해서는 답변드리기 어렵습니다.\n\n"
        "저는 네이버 부스트캠프 AI 교육 과정과 관련된 질문에 답변드릴 수 있습니다:\n"
        "• **AI/ML 개념 설명** - Transformer, CNN, RNN, 추천 시스템 등\n"
        "• **딥러닝 프레임워크** - PyTorch, TensorFlow 사용법\n"
        "• **코드 구현 방법** - 모델 학습, 데이터 전처리 등\n"
        "• **강의 내용 관련 질문** - CV, NLP, RecSys 강의\n"
        "• **실습/과제 관련 질문**\n\n"
        "위와 관련된 질문이 있으시면 언제든 도움드리겠습니다! 🤖"
    )

    logger.info(
        f"OOD response generated for question: '{question_preview}' "
        f"(domain_relevance: {domain_relevance:.2f})"
    )

    return {
        "answer": response,
        "generation_strategy": "ood_decline",
        "workflow_stage": "completed",
        "is_out_of_domain": True,
    }


async def clarify_node(state: AdaptiveRAGState) -> dict[str, Any]:
    """
    사용자에게 명확화를 요청하는 응답을 생성합니다.

    필터 추출 신뢰도가 낮을 때 (filter_confidence < 0.5) 호출되어
    사용자에게 검색 범위를 좁힐 수 있는 선택지를 제시합니다.

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        명확화 요청 응답을 포함한 상태 업데이트
    """
    logger.info("---CLARIFY FILTER---")

    question = state["question"]
    filters = state.get("retrieval_filters", {})
    courses = filters.get("course", []) if filters else []

    # 명확화 메시지 생성
    clarification_parts = [
        "질문을 더 정확하게 이해하기 위해 확인이 필요합니다.\n"
    ]

    if courses:
        clarification_parts.append(f"'{question}'에서 언급하신 과정이 다음 중 어느 것인가요?\n")
        for i, course in enumerate(courses[:5], 1):
            clarification_parts.append(f"{i}. {course}")
        clarification_parts.append("\n원하시는 과정 번호를 알려주시거나, 더 구체적으로 질문해 주세요.")
    else:
        clarification_parts.append(
            "어떤 자료에서 찾아볼까요?\n"
            "- **강의자료** (PDF 슬라이드)\n"
            "- **녹취록** (강의 내용)\n"
            "- **슬랙 Q&A** (질의응답)\n"
            "- **실습 노트북** (코드)\n"
            "- **미션** (과제)\n"
        )

    clarification_message = "\n".join(clarification_parts)

    return {
        "answer": clarification_message,
        "workflow_stage": "awaiting_clarification",
    }


def finalize_node(state: AdaptiveRAGState) -> dict[str, Any]:
    """
    워크플로를 마무리합니다.

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        종료 상태 업데이트
    """
    logger.info("---FINALIZE---")

    # 워크플로를 완료 상태로 표시합니다.
    return {
        "workflow_stage": "completed",
    }
