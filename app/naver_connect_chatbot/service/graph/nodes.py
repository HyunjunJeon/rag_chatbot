"""
Adaptive RAG 워크플로에서 사용하는 노드 함수 집합.

각 노드는 RAG 프로세스의 개별 단계를 나타냅니다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage, ToolMessage

from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.service.graph.types import (
    IntentUpdate,
    QueryAnalysisUpdate,
    AgentUpdate,
    PostProcessUpdate,
    OODResponseUpdate,
)
from naver_connect_chatbot.service.agents.intent_classifier import aclassify_intent
from naver_connect_chatbot.service.agents.query_analyzer import aanalyze_query
from naver_connect_chatbot.config import logger


# =============================================================================
# Multi-turn 대화 지원 유틸리티
# =============================================================================

# 대화 히스토리 최대 턴 수 (메모리 및 컨텍스트 길이 관리)
MAX_HISTORY_TURNS = 5


def _format_chat_history(messages: list[BaseMessage], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """
    대화 히스토리를 프롬프트용 텍스트로 포맷팅합니다.

    매개변수:
        messages: BaseMessage 리스트 (HumanMessage, AIMessage)
        max_turns: 포함할 최대 턴 수 (기본값: 5)

    반환값:
        포맷팅된 대화 히스토리 문자열 (턴 번호 포함, 중복 방지 지시 포함)
    """
    if not messages:
        return ""

    # 최근 N턴만 사용 (1턴 = 사용자 + AI 쌍)
    recent_messages = messages[-(max_turns * 2) :]

    if not recent_messages:
        return ""

    history_lines = ["[이전 대화]"]
    turn_num = 0
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            turn_num += 1
            history_lines.append(f"[턴 {turn_num}] 사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            # AI 응답은 너무 길면 요약 (500자로 확장)
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"[턴 {turn_num}] 어시스턴트: {content}")

    history_lines.append(
        "[주의] 위 대화에서 이미 답변한 내용을 반복하지 마세요. 새로운 정보에 집중하세요."
    )

    return "\n".join(history_lines) + "\n"


# =============================================================================
# 응답 추출 유틸리티
# =============================================================================


def _extract_text_from_content(content: Any) -> str:
    """
    content 필드에서 텍스트를 추출합니다.

    Gemini thinking_level 사용 시 content가 리스트 형태로 반환될 수 있습니다:
    [{"type": "thinking", "text": "..."}, {"type": "text", "text": "actual answer"}]
    이 경우 type="text" 블록만 추출합니다.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                # Gemini thinking block 형식: {"type": "text", "text": "..."}
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block and "type" not in block:
                    # type 필드 없는 단순 텍스트 블록
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return "\n".join(text_parts)
        # thinking 블록만 있는 경우 (text 블록 없음) → 첫 번째 블록의 text 사용
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return block["text"]

    return str(content)


def _extract_text_response(response: Any) -> str:
    """
    LangChain 에이전트 응답에서 텍스트를 안전하게 추출합니다.

    Gemini thinking blocks (리스트 형태 content)도 올바르게 처리합니다.
    """
    if isinstance(response, AIMessage):
        return _extract_text_from_content(response.content)

    if hasattr(response, "content"):
        return _extract_text_from_content(response.content)

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

    Gemini with_structured_output()을 사용하여 IntentClassification을 직접 반환받습니다.

    매개변수:
        state: 현재 워크플로 상태
        llm: 분류에 사용할 언어 모델

    반환값:
        의도 분류 결과가 포함된 상태 업데이트 (domain_relevance 포함)
    """
    logger.info("---CLASSIFY INTENT---")
    question = state["question"]
    question_lower = question.lower().strip()

    # 1. 패턴 매칭으로 확실한 OUT_OF_DOMAIN 먼저 처리 (LLM 호출 없이 빠르게)
    ood_patterns = {
        "greeting": [
            "안녕",
            "반가",
            "하이",
            "헬로",
            "hello",
            "hi ",
            "hey",
            "잘 지내",
            "좋은 아침",
            "좋은 저녁",
        ],
        "self_intro": [
            "이름이 뭐",
            "넌 누구",
            "너 누구",
            "뭘 할 수 있",
            "뭘 도와줄 수 있",
            "어떤 봇",
            "무슨 봇",
            "뭐하는 봇",
            "소개해",
            "자기소개",
            "who are you",
            "what can you do",
            "what's your name",
        ],
        "chitchat": [
            "뭐해",
            "심심",
            "배고파",
            "졸려",
            "피곤",
        ],
        "off_topic": [
            "날씨",
            "점심 메뉴",
            "저녁 메뉴",
            "아침 메뉴",
            "메뉴 추천",
            "맛집",
            "여행",
            "주식",
            "투자",
            "연예",
            "스포츠",
            "축구 경기",
            "야구 경기",
        ],
    }

    for pattern_type, patterns in ood_patterns.items():
        if any(pattern in question_lower for pattern in patterns):
            logger.info(f"Pattern-matched OUT_OF_DOMAIN ({pattern_type}): '{question[:50]}'")
            return {
                "intent": "OUT_OF_DOMAIN",
                "intent_confidence": 0.95,
                "intent_reasoning": f"Pattern matched: {pattern_type}",
                "domain_relevance": 0.0,
                # Multi-turn: 사용자 질문을 HumanMessage로 저장
                "messages": [HumanMessage(content=question)],
            }

    # 2. 패턴 매칭에 걸리지 않으면 LLM으로 분류
    response = await aclassify_intent(question, llm)

    # domain_relevance가 매우 낮으면 OUT_OF_DOMAIN으로 보정 (Hard OOD 임계값)
    intent = response.intent
    domain_relevance = response.domain_relevance

    if domain_relevance < 0.2 and intent != "OUT_OF_DOMAIN":
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

    # Multi-turn: 사용자 질문을 HumanMessage로 저장
    return {
        "intent": intent,
        "intent_confidence": response.confidence,
        "intent_reasoning": response.reasoning,
        "domain_relevance": domain_relevance,
        "messages": [HumanMessage(content=question)],
    }


async def analyze_query_node(state: AdaptiveRAGState, llm: Runnable) -> QueryAnalysisUpdate:
    """
    질의 품질을 분석하고 다중 검색 쿼리 및 검색 필터를 생성합니다.

    이 노드는 Query Analysis, Multi-Query Generation, Filter Extraction을 통합하여:
    1. 질의의 명확성, 구체성, 검색 가능성을 평가
    2. 다양한 관점의 검색 쿼리 3-5개 생성 (Multi-Query)
    3. 질문에서 메타데이터 기반 검색 필터 추출 (doc_type, course, etc.)

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
                        logger.warning("SchemaRegistry not loaded, using original course names")
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
                        logger.info(f"Course names resolved: {rf.course} → {filters['course']}")
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
                    f"Low filter confidence ({filter_confidence:.2f}), clarification may be needed"
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


# =============================================================================
# Agent Node (Tool-based Retrieval)
# =============================================================================


def _build_agent_system_prompt(
    intent: str,
    domain_relevance: float,
    refined_queries: list[str],
) -> str:
    """
    Agent LLM에 전달할 시스템 프롬프트를 구성합니다.

    매개변수:
        intent: 분류된 질문 의도
        domain_relevance: 도메인 관련성 점수 (0.0~1.0)
        refined_queries: 분석을 통해 생성된 개선 검색 쿼리 목록

    반환값:
        시스템 프롬프트 문자열
    """
    queries_hint = ""
    if refined_queries:
        queries_hint = (
            "\n\n## 추천 검색 쿼리 (참고용)\n"
            + "\n".join(f"- {q}" for q in refined_queries)
        )

    return (
        "<role>당신은 Naver Boost Camp AI Tech 학생들을 위한 학습 도우미입니다.</role>\n\n"
        "## 도구 사용 지침\n"
        "- `qdrant_search`: 부스트캠프 교육 자료(강의, 노트북, Slack Q&A, 미션) 검색\n"
        "- `web_search`: 웹에서 최신 정보 검색 (교육 자료에 없는 일반 개념/최신 정보용)\n\n"
        "## 규칙\n"
        "1. 교육 자료 관련 질문은 먼저 `qdrant_search`를 사용하세요.\n"
        "2. 교육 자료에 정보가 부족하면 `web_search`로 보충하세요.\n"
        "3. 이전 대화에서 이미 충분한 정보가 있으면 도구 없이 바로 답변하세요.\n"
        "4. 답변은 한국어로 작성하세요.\n"
        "5. 문서를 인용할 때는 대괄호 라벨을 사용하세요 (예: [강의자료: CV 이론/3강]).\n"
        "6. '문서 1', '문서 2' 같은 순번 참조는 사용하지 마세요.\n"
        "7. 이전 대화에서 이미 답변한 내용을 반복하지 마세요.\n\n"
        f"## 현재 질문 분석\n"
        f"- 의도: {intent}\n"
        f"- 도메인 관련성: {domain_relevance:.2f}\n"
        f"{queries_hint}"
    )


async def agent_node(
    state: AdaptiveRAGState,
    llm: Runnable,
    tools: list,
) -> AgentUpdate:
    """
    LLM에 도구를 바인딩하고 호출하는 에이전트 노드입니다.

    LLM이 필요에 따라 qdrant_search, web_search 도구를 선택적으로 호출합니다.
    tool_calls가 있으면 tools 노드로, 없으면 post_process 노드로 라우팅됩니다.

    Multi-turn 지원:
    - 이전 턴의 HumanMessage + AIMessage(최종 답변만) 포함
    - 이전 턴의 ToolMessage는 필터링하여 컨텍스트 절약

    매개변수:
        state: 현재 워크플로 상태
        llm: 도구 호출이 가능한 LLM (Gemini)
        tools: 바인딩할 도구 리스트

    반환값:
        messages에 AIMessage가 append된 상태 업데이트
    """
    logger.info("---AGENT NODE---")
    messages = state.get("messages", [])
    intent = state.get("intent", "SIMPLE_QA")
    domain_relevance = state.get("domain_relevance", 1.0)
    refined_queries = state.get("refined_queries", [])

    # 1. 시스템 프롬프트 구성
    system_content = _build_agent_system_prompt(intent, domain_relevance, refined_queries)

    # 2. LLM용 메시지 리스트 구성
    llm_messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    # 현재 턴 시작점 찾기 (마지막 HumanMessage)
    current_turn_idx = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            current_turn_idx = i
            break

    # 이전 턴: HumanMessage + AIMessage(tool_calls 없는 최종 답변만)
    for msg in messages[:current_turn_idx]:
        if isinstance(msg, HumanMessage):
            llm_messages.append(msg)
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = _extract_text_from_content(msg.content)
            if len(content) > 500:
                content = content[:500] + "..."
            llm_messages.append(AIMessage(content=content))

    # 현재 턴: HumanMessage + 도구 호출/응답 전부 포함
    for msg in messages[current_turn_idx:]:
        llm_messages.append(msg)

    # 3. LLM 호출 (tools bind)
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(llm_messages)

    # 4. tool_call_count 업데이트
    tool_call_count = state.get("tool_call_count", 0)
    if response.tool_calls:
        tool_call_count += 1
        logger.info(
            f"Agent requested {len(response.tool_calls)} tool call(s), "
            f"iteration {tool_call_count}"
        )
    else:
        answer_preview = _extract_text_from_content(response.content)[:100]
        logger.info(f"Agent produced final answer: {answer_preview}...")

    return {
        "messages": [response],
        "tool_call_count": tool_call_count,
    }


async def post_process_node(state: AdaptiveRAGState) -> PostProcessUpdate:
    """
    Agent 루프 완료 후 최종 답변을 추출하고 후처리합니다.

    처리 내용:
    1. messages에서 마지막 AIMessage(tool_calls 없는 것)의 텍스트 추출
    2. Post-retrieval OOD 감지: 모든 도구가 "검색 결과 없음" + domain_relevance < 0.5
    3. 답변과 생성 메타데이터 반환

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        최종 답변과 메타데이터를 포함한 상태 업데이트
    """
    logger.info("---POST PROCESS---")
    messages = state.get("messages", [])
    domain_relevance = state.get("domain_relevance", 1.0)
    question = state.get("question", "")

    # 마지막 AIMessage에서 최종 답변 추출
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not msg.tool_calls:
                answer = _extract_text_from_content(msg.content)
                break
            else:
                # max iterations 도달 — tool_calls는 있지만 텍스트도 있을 수 있음
                text = _extract_text_from_content(msg.content)
                if text.strip():
                    answer = text
                    break

    if not answer:
        answer = "죄송합니다. 답변을 생성할 수 없었습니다. 질문을 다시 시도해주세요."

    # Post-retrieval OOD 감지
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    all_tools_empty = (
        tool_msgs
        and all("검색 결과 없음" in m.content for m in tool_msgs)
        and domain_relevance < 0.5
    )

    if all_tools_empty:
        logger.info(
            f"Post-retrieval soft decline: all tools empty + "
            f"low relevance ({domain_relevance:.2f})"
        )
        return {
            "answer": (
                f"'{question}'에 대해 교육 자료에서 관련 정보를 검색했으나, "
                "직접적으로 관련된 문서를 찾지 못했습니다.\n\n"
                "다음을 시도해보세요:\n"
                "- 질문에 구체적인 기술 용어를 포함해주세요\n"
                "- 부스트캠프 강의나 과제와 관련된 맥락을 추가해주세요"
            ),
            "generation_metadata": {
                "strategy": "post_retrieval_soft_decline",
                "domain_relevance": domain_relevance,
            },
            "generation_strategy": "post_retrieval_soft_decline",
            "is_out_of_domain": True,
        }

    return {
        "answer": answer,
        "generation_metadata": {
            "strategy": "tool_based_agent",
            "tool_calls_count": state.get("tool_call_count", 0),
        },
        "generation_strategy": "tool_based_agent",
    }


async def generate_ood_response_node(state: AdaptiveRAGState) -> OODResponseUpdate:
    """
    Out-of-Domain 질문에 대한 응답을 생성합니다.

    인사/잡담에는 친근하게 응답하고, 그 외 AI/ML 교육과 무관한 질문에 대해서는
    정중히 거절하고 도움 가능한 영역을 안내합니다.

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        OOD 응답을 포함한 상태 업데이트
    """
    logger.info("---GENERATE OOD RESPONSE---")

    question = state.get("question", "")
    domain_relevance = state.get("domain_relevance", 0.0)

    # 패턴 감지
    question_lower = question.lower().strip()

    # 챗봇 자기소개 패턴
    self_intro_patterns = [
        "이름이 뭐",
        "넌 누구",
        "너 누구",
        "뭘 할 수 있",
        "뭘 도와줄 수 있",
        "어떤 봇",
        "무슨 봇",
        "뭐하는 봇",
        "소개해",
        "자기소개",
        "who are you",
        "what can you do",
        "what's your name",
    ]
    is_self_intro = any(pattern in question_lower for pattern in self_intro_patterns)

    # 인사/잡담 패턴
    greeting_patterns = [
        "안녕",
        "반가",
        "하이",
        "헬로",
        "hello",
        "hi ",
        "hey",
        "잘 지내",
        "뭐해",
        "심심",
        "좋은 아침",
        "좋은 저녁",
    ]
    is_greeting = any(pattern in question_lower for pattern in greeting_patterns)

    if is_self_intro:
        response = (
            "안녕하세요! 저는 **네이버 부스트캠프 AI Tech 학습 도우미**입니다. 🤖\n\n"
            "부스트캠프 교육 과정에서 학습하시면서 궁금한 점이 있을 때 도움을 드리기 위해 만들어졌어요.\n\n"
            "**제가 도와드릴 수 있는 영역:**\n"
            "• AI/ML 개념 설명 (Transformer, CNN, 추천 시스템 등)\n"
            "• PyTorch, 딥러닝 코드 구현 방법\n"
            "• 강의 내용 관련 질문 (CV, NLP, RecSys)\n"
            "• 실습 및 과제 관련 질문\n\n"
            "편하게 질문해주세요! 😊"
        )
        logger.info(f"Self-intro response generated for: '{question}'")
    elif is_greeting:
        response = (
            "안녕하세요! 😊 네이버 부스트캠프 AI Tech 학습 도우미입니다.\n\n"
            "무엇을 도와드릴까요? 다음과 같은 질문에 답변드릴 수 있어요:\n"
            "• AI/ML 개념 (Transformer, CNN, 추천 시스템 등)\n"
            "• PyTorch, 딥러닝 코드 구현\n"
            "• 강의 내용 및 실습/과제 관련 질문\n\n"
            "편하게 질문해주세요! 🤖"
        )
        logger.info(f"Greeting response generated for: '{question}'")
    else:
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

    # Multi-turn: OOD 응답도 AIMessage로 저장
    return {
        "answer": response,
        "generation_strategy": "ood_decline",
        "workflow_stage": "completed",
        "is_out_of_domain": True,
        "messages": [AIMessage(content=response)],
    }


async def clarify_node(state: AdaptiveRAGState) -> dict[str, Any]:
    """
    사용자에게 명확화를 요청하는 응답을 생성합니다.

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
    clarification_parts = ["질문을 더 정확하게 이해하기 위해 확인이 필요합니다.\n"]

    if courses:
        clarification_parts.append(f"'{question}'에서 언급하신 과정이 다음 중 어느 것인가요?\n")
        for i, course in enumerate(courses[:5], 1):
            clarification_parts.append(f"{i}. {course}")
        clarification_parts.append(
            "\n원하시는 과정 번호를 알려주시거나, 더 구체적으로 질문해 주세요."
        )
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
