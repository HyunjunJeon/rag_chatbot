"""
Slack Handler 모듈
Slack App 을 초기화하고, 이벤트를 처리하는 핸들러를 제공합니다.
참고 문서:
    - https://github.com/slackapi/bolt-python
    - https://github.com/slackapi/bolt-python/tree/main/examples/fastapi
"""

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from slack_bolt.async_app import AsyncApp

from naver_connect_chatbot.config.settings.main import settings
from naver_connect_chatbot.config.log import get_logger
from naver_connect_chatbot.config.llm import get_chat_model
from naver_connect_chatbot.config.embedding import get_embeddings
from naver_connect_chatbot.config.monitoring import get_langfuse_callback
from naver_connect_chatbot.rag.retriever_factory import (
    build_dense_sparse_hybrid_from_saved,
    get_hybrid_retriever,
)
from naver_connect_chatbot.service.graph.workflow import build_adaptive_rag_graph
from naver_connect_chatbot.config.settings.base import PROJECT_ROOT

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

# Logging setup
logger = get_logger()

# Initialize Slack App
app = AsyncApp(
    token=settings.slack.bot_token.get_secret_value(),
    signing_secret=settings.slack.signing_secret.get_secret_value(),
)

# Initialize RAG Components (Global for now, or could be dependency injected)
# We need to make sure these are initialized when the app starts or on first request
# For simplicity, we'll initialize them lazily or globally if env vars are present.


def get_agent_app(checkpointer: "BaseCheckpointSaver | None" = None):
    """
    LangGraph 애플리케이션을 초기화하고 반환합니다.

    매개변수:
        checkpointer: LangGraph 체크포인터 (대화 상태 저장용)
                      None이면 대화 상태가 저장되지 않음

    반환값:
        Compiled LangGraph application

    예외:
        ValueError: 필수 설정이 누락된 경우
        Exception: 초기화 중 오류 발생 시
    """
    # 1. Embeddings - 팩토리 함수 사용 (langchain_naver.ClovaXEmbeddings)
    embeddings = get_embeddings()

    # 2. LLM - 팩토리 함수 사용 (langchain_google_genai.ChatGoogleGenerativeAI)
    #    - 기본 llm: thinking_level="low" (분류/분석 등 가벼운 작업용)
    #    - reasoning_llm: generate_answer_node 전용 (기본 thinking_level="high")
    llm = get_chat_model(thinking_level="low")
    reasoning_llm = get_chat_model()

    # 3. Retriever - 저장된 BM25 인덱스 로드 (없으면 Qdrant만 사용)
    bm25_index_path = PROJECT_ROOT / settings.retriever.bm25_index_path
    qdrant_api_key = (
        settings.qdrant_vector_store.api_key.get_secret_value()
        if settings.qdrant_vector_store.api_key
        else None
    )

    if bm25_index_path.exists():
        logger.info(f"BM25 인덱스 로드: {bm25_index_path}")
        retriever = build_dense_sparse_hybrid_from_saved(
            bm25_index_path=bm25_index_path,
            embedding_model=embeddings,
            qdrant_url=settings.qdrant_vector_store.url,
            collection_name=settings.qdrant_vector_store.collection_name,
            qdrant_api_key=qdrant_api_key,
            k=settings.retriever.default_k,
        )
    else:
        logger.warning(
            f"BM25 인덱스를 찾을 수 없습니다: {bm25_index_path}. "
            "Qdrant Dense 검색만 사용합니다. "
            "Sparse 검색을 활성화하려면 document_processing/rebuild_unified_bm25.py를 실행하세요."
        )
        retriever = get_hybrid_retriever(
            documents=[],  # 빈 BM25 (Qdrant만 사용)
            embedding_model=embeddings,
            qdrant_url=settings.qdrant_vector_store.url,
            collection_name=settings.qdrant_vector_store.collection_name,
            qdrant_api_key=qdrant_api_key,
            k=settings.retriever.default_k,
        )

    # 4. Build Graph with Checkpointer (tool-based agent)
    workflow_app = build_adaptive_rag_graph(
        retriever=retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        reranker_settings=settings.reranker,
        gemini_llm_settings=settings.gemini_llm,
        check_pointers=checkpointer,
    )

    if checkpointer:
        logger.info("✓ Checkpointer 활성화됨 - 대화 상태가 저장됩니다")
    else:
        logger.warning("⚠ Checkpointer 없음 - 대화 상태가 저장되지 않습니다")

    return workflow_app


# Global agent instance with thread-safe initialization
_agent_app = None
_agent_lock = asyncio.Lock()
_agent_init_failed = False

# Global checkpointer instance (initialized in server.py lifespan)
_checkpointer: "BaseCheckpointSaver | None" = None


def set_checkpointer(checkpointer: "BaseCheckpointSaver | None") -> None:
    """
    전역 체크포인터를 설정합니다.

    이 함수는 server.py의 lifespan에서 호출되어야 합니다.

    매개변수:
        checkpointer: AsyncSqliteSaver 등의 체크포인터 인스턴스
    """
    global _checkpointer
    _checkpointer = checkpointer
    if checkpointer:
        logger.info("전역 checkpointer 설정 완료")
    else:
        logger.warning("전역 checkpointer가 None으로 설정됨")


# =============================================================================
# 메시지 전처리 유틸리티
# =============================================================================

# Slack 멘션 패턴: <@U0A091KUCDV> 형태
SLACK_MENTION_PATTERN = re.compile(r"<@[A-Z0-9]+>")

# 간단한 인사/감정 표현 패턴 (RAG 파이프라인 불필요)
GREETING_PATTERNS = [
    r"^(안녕|하이|헬로|hi|hello|hey)[\s!?.]*$",
    r"^(야호|우와|와|오|헉|ㅎㅎ|ㅋㅋ|ㄱㄱ|ㄴㄴ)[\s!?.]*$",
    r"^(감사|고마워|땡큐|thanks|thx)[\s!?.]*$",
    r"^(ㅇㅇ|ㅇㅋ|ㄹㅇ|ㄳ|ㅎㅇ)[\s!?.]*$",
    r"^[\s!?.~ㅋㅎ]+$",  # 이모티콘/감탄사만 있는 경우
]
GREETING_REGEX = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)

# 간단한 인사에 대한 응답 목록
GREETING_RESPONSES = [
    "안녕하세요! 👋 네이버 부스트캠프 관련 질문이 있으시면 말씀해주세요.",
    "안녕하세요! 무엇을 도와드릴까요? 🤖",
    "반갑습니다! AI/ML 학습 관련 궁금한 점이 있으시면 질문해주세요.",
]


def preprocess_slack_message(text: str) -> str:
    """
    Slack 메시지를 전처리합니다.

    처리 내용:
    1. Slack 멘션 제거 (<@U0A091KUCDV> → "")
    2. Slack 특수 포맷 정리 (<#C12345|channel> → #channel 등)
    3. 앞뒤 공백 정리

    매개변수:
        text: 원본 Slack 메시지 텍스트

    반환값:
        전처리된 텍스트

    예시:
        >>> preprocess_slack_message("<@U0A091KUCDV> 야호!")
        "야호!"
        >>> preprocess_slack_message("<@U123> PyTorch가 뭐야?")
        "PyTorch가 뭐야?"
    """
    if not text:
        return ""

    # 1. Slack 멘션 제거
    cleaned = SLACK_MENTION_PATTERN.sub("", text)

    # 2. Slack 채널/링크 포맷 정리 (<#C12345|channel> → #channel)
    cleaned = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", cleaned)

    # 3. Slack URL 포맷 정리 (<http://...|label> → label 또는 URL)
    cleaned = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2", cleaned)
    cleaned = re.sub(r"<(https?://[^>]+)>", r"\1", cleaned)

    # 4. 앞뒤 공백 및 연속 공백 정리
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def is_simple_greeting(text: str) -> bool:
    """
    메시지가 간단한 인사/감정 표현인지 확인합니다.

    RAG 파이프라인을 거치지 않고 바로 응답할 수 있는 메시지인지 판단합니다.

    매개변수:
        text: 전처리된 메시지 텍스트

    반환값:
        True이면 간단한 인사, False이면 실제 질문

    예시:
        >>> is_simple_greeting("야호!")
        True
        >>> is_simple_greeting("PyTorch가 뭐야?")
        False
    """
    if not text:
        return True  # 빈 메시지는 인사로 처리

    # 너무 짧은 메시지 (3자 이하)
    if len(text) <= 3:
        return True

    # 패턴 매칭
    return bool(GREETING_REGEX.match(text))


def get_greeting_response() -> str:
    """무작위 인사 응답을 반환합니다."""
    import random

    return random.choice(GREETING_RESPONSES)


# =============================================================================
# Rate Limiting
# =============================================================================

# Rate limiting configuration
RATE_LIMIT_MAX_REQUESTS = 5  # 분당 최대 요청 수
RATE_LIMIT_WINDOW_SECONDS = 60  # 제한 윈도우 (초)
_rate_limit_cache: dict[str, list[datetime]] = defaultdict(list)

# Request timeout configuration
REQUEST_TIMEOUT_SECONDS = 120.0  # 2분 타임아웃


def _check_rate_limit(user_id: str) -> tuple[bool, int]:
    """
    사용자별 요청 속도 제한을 확인합니다.

    매개변수:
        user_id: Slack 사용자 ID

    반환값:
        (허용 여부, 남은 요청 수) 튜플
    """
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    # 윈도우 내 요청만 유지
    _rate_limit_cache[user_id] = [ts for ts in _rate_limit_cache[user_id] if ts > window_start]

    current_count = len(_rate_limit_cache[user_id])

    if current_count >= RATE_LIMIT_MAX_REQUESTS:
        return False, 0

    # 요청 기록
    _rate_limit_cache[user_id].append(now)
    return True, RATE_LIMIT_MAX_REQUESTS - current_count - 1


async def get_or_create_agent():
    """
    Thread-safe하게 Agent 인스턴스를 가져오거나 생성합니다.

    동시 요청 시 race condition을 방지하기 위해 asyncio.Lock을 사용합니다.
    전역 checkpointer가 설정되어 있으면 대화 상태가 저장됩니다.

    반환값:
        Compiled LangGraph application

    예외:
        RuntimeError: Agent 초기화가 이전에 실패한 경우
        Exception: 초기화 중 오류 발생 시
    """
    global _agent_app, _agent_init_failed

    # Fast path: 이미 초기화된 경우
    if _agent_app is not None:
        return _agent_app

    # 이전 초기화 실패 체크
    if _agent_init_failed:
        raise RuntimeError("Agent 초기화가 이전에 실패했습니다. 서버를 재시작해주세요.")

    async with _agent_lock:
        # Double-check pattern: 락 획득 후 다시 확인
        if _agent_app is not None:
            return _agent_app

        try:
            logger.info("Agent 초기화 시작...")
            # 전역 checkpointer를 사용하여 Agent 생성
            agent = get_agent_app(checkpointer=_checkpointer)
            _agent_app = agent
            logger.info("Agent 초기화 완료")
            return _agent_app
        except Exception as e:
            _agent_init_failed = True
            logger.error("Agent 초기화 실패", error=str(e), exc_info=True)
            raise


@app.event("app_mention")
async def handle_app_mention(event, say):
    """
    Handle app_mention events with LangFuse tracing.
    사용자가 봇을 멘션하면 질문에 대한 답변을 생성합니다.

    매개변수:
        event: Slack event payload
        say: Slack response function
    """
    # Extract Slack context (먼저 추출하여 rate limiting에 사용)
    user_id = event.get("user")
    channel_id = event.get("channel")
    raw_input = event.get("text", "")
    thread_ts = event.get("ts")  # Use message ts as thread_ts for the reply

    # If it's already in a thread, use that thread_ts
    if "thread_ts" in event:
        thread_ts = event["thread_ts"]

    # =========================================================================
    # 메시지 전처리: Slack 멘션 제거 및 정리
    # =========================================================================
    user_input = preprocess_slack_message(raw_input)
    logger.info(f"멘션 수신 (원본): {raw_input}")
    logger.info(f"멘션 수신 (전처리): {user_input} (thread: {thread_ts})")

    # =========================================================================
    # 간단한 인사/감정 표현 처리 (RAG 파이프라인 우회)
    # =========================================================================
    if is_simple_greeting(user_input):
        logger.info(f"간단한 인사로 감지됨: '{user_input}' → 빠른 응답")
        await say(text=get_greeting_response(), thread_ts=thread_ts)
        return

    # Rate limiting 체크
    allowed, remaining = _check_rate_limit(user_id)
    if not allowed:
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await say(
            text=f"⏳ 요청이 너무 많습니다. {RATE_LIMIT_WINDOW_SECONDS}초 후에 다시 시도해주세요.",
            thread_ts=thread_ts,
        )
        return

    logger.info(f"RAG 파이프라인 시작: '{user_input[:50]}...' (remaining: {remaining})")

    try:
        agent_app = await get_or_create_agent()
    except Exception as e:
        error_msg = "챗봇을 초기화하는 중 오류가 발생했습니다. 관리자에게 문의해주세요."
        logger.error("Agent 초기화 실패", error=str(e))
        await say(text=error_msg, thread_ts=thread_ts)
        return

    # Create LangFuse callback with Slack metadata
    langfuse_handler = get_langfuse_callback(
        user_id=user_id, channel_id=channel_id, thread_ts=thread_ts, event_type="slack_mention"
    )

    # Prepare callbacks list (empty if LangFuse disabled)
    callbacks = [langfuse_handler] if langfuse_handler else []

    # Create runnable config with callbacks, metadata, and thread_id for checkpointing
    # thread_ts를 thread_id로 사용하여 같은 Slack 스레드의 대화 맥락을 유지
    config = {
        "callbacks": callbacks,
        "configurable": {
            "thread_id": thread_ts,  # Slack thread_ts를 LangGraph thread_id로 사용
        },
        "metadata": {
            "source": "slack",
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
        },
    }

    inputs = {"question": user_input}

    try:
        # Run the graph with callback (auto-propagates to all nodes)
        # 타임아웃 적용으로 무한 대기 방지
        logger.info("Agent 실행 시작...")
        try:
            result = await asyncio.wait_for(
                agent_app.ainvoke(inputs, config=config),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Agent 실행 타임아웃 ({REQUEST_TIMEOUT_SECONDS}초)",
                user_id=user_id,
                channel_id=channel_id,
            )
            await say(
                text="⏱️ 요청 처리 시간이 초과되었습니다. 질문을 더 간단하게 해주시거나 잠시 후 다시 시도해주세요.",
                thread_ts=thread_ts,
            )
            return

        answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")

        logger.info(f"답변 생성 완료: {answer[:100]}...")
        await say(text=answer, thread_ts=thread_ts)

        # Ensure trace is flushed before function returns
        # (Critical for LangChain 0.3+ async callbacks)
        if langfuse_handler:
            await langfuse_handler.aflush()

    except Exception as e:
        logger.error("요청 처리 중 오류 발생", error=str(e), exc_info=True)
        await say(
            text="죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            thread_ts=thread_ts,
        )


@app.message("")
async def handle_message(message, say, client):
    """
    Handle direct messages and thread replies in conversations where the bot participated.

    봇이 이미 참여한 Thread 내에서는 @멘션 없이도 자동 응답합니다.
    이를 통해 자연스러운 대화 흐름을 유지할 수 있습니다.

    매개변수:
        message: Slack message payload
        say: Slack response function
        client: Slack WebClient for API calls
    """
    # Ignore bot's own messages and message subtypes (edits, deletes, etc.)
    if message.get("subtype") is not None or message.get("bot_id") is not None:
        return

    channel_type = message.get("channel_type")
    thread_ts = message.get("thread_ts")

    # Case 1: DM (Direct Message) - 항상 응답
    if channel_type == "im":
        await handle_app_mention(message, say)
        return

    # Case 2: Thread 내 메시지 - 봇이 참여한 Thread인지 확인
    if thread_ts:
        try:
            # Thread 내 메시지 목록 조회
            result = await client.conversations_replies(
                channel=message.get("channel"),
                ts=thread_ts,
                limit=50,  # 최근 50개 메시지만 확인
            )

            # 봇이 이 Thread에 참여했는지 확인
            bot_user_id = (await client.auth_test())["user_id"]
            bot_participated = any(
                msg.get("user") == bot_user_id or msg.get("bot_id") is not None
                for msg in result.get("messages", [])
            )

            if bot_participated:
                logger.info(
                    f"Thread reply detected (bot participated): {message.get('text', '')[:50]}"
                )
                await handle_app_mention(message, say)
                return

        except Exception as e:
            logger.warning(f"Failed to check thread participation: {e}")
            # 실패 시 무시 (멘션이 있을 때만 응답)
