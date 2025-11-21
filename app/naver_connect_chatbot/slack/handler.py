"""
Slack Handler 모듈
Slack App 을 초기화하고, 이벤트를 처리하는 핸들러를 제공합니다.
참고 문서:
    - https://github.com/slackapi/bolt-python
    - https://github.com/slackapi/bolt-python/tree/main/examples/fastapi
"""

import asyncio
from slack_bolt.async_app import AsyncApp

from naver_connect_chatbot.config.settings.main import settings
from naver_connect_chatbot.config.log import get_logger
from naver_connect_chatbot.config.llm import get_chat_model
from naver_connect_chatbot.config.monitoring import get_langfuse_callback
from naver_connect_chatbot.rag.embeddings import NaverCloudEmbeddings
from naver_connect_chatbot.rag.retriever_factory import get_hybrid_retriever
from naver_connect_chatbot.agent.graph.workflow import build_graph

# Logging setup
logger = get_logger()

# Initialize Slack App
app = AsyncApp(
    token=settings.slack.bot_token.get_secret_value(),
    signing_secret=settings.slack.signing_secret.get_secret_value()
)

# Initialize RAG Components (Global for now, or could be dependency injected)
# We need to make sure these are initialized when the app starts or on first request
# For simplicity, we'll initialize them lazily or globally if env vars are present.


def get_agent_app():
    """
    LangGraph 애플리케이션을 초기화하고 반환합니다.
    
    반환값:
        Compiled LangGraph application
        
    예외:
        ValueError: 필수 설정이 누락된 경우
        Exception: 초기화 중 오류 발생 시
    """
    # 1. Embeddings - Settings 사용
    embeddings = NaverCloudEmbeddings(
        model_url=settings.naver_cloud_embeddings.model_url,
        api_key=settings.naver_cloud_embeddings.api_key.get_secret_value() if settings.naver_cloud_embeddings.api_key else None,
    )

    # 2. LLM - 팩토리 함수 사용으로 일관성 유지
    llm = get_chat_model()

    # 3. Retriever - BM25 인덱스 로드 또는 빈 리스트
    # 빈 리스트로 시작 (Qdrant만 사용)
    # 향후 BM25 인덱스를 로드하려면 settings.retriever.bm25_index_path 사용
    initial_docs = []

    retriever = get_hybrid_retriever(
        documents=initial_docs,
        embedding_model=embeddings,
        qdrant_url=settings.qdrant_vector_store.url,
        collection_name=settings.qdrant_vector_store.collection_name,
        qdrant_api_key=settings.qdrant_vector_store.api_key.get_secret_value() if settings.qdrant_vector_store.api_key else None,
        k=settings.retriever.default_k,
    )

    # 4. Build Graph
    workflow_app = build_graph(retriever=retriever, llm=llm)
    return workflow_app


# Global agent instance with thread-safe initialization
_agent_app = None
_agent_lock = asyncio.Lock()
_agent_init_failed = False


async def get_or_create_agent():
    """
    Thread-safe하게 Agent 인스턴스를 가져오거나 생성합니다.

    동시 요청 시 race condition을 방지하기 위해 asyncio.Lock을 사용합니다.

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
            agent = get_agent_app()
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
    try:
        agent_app = await get_or_create_agent()
    except Exception as e:
        error_msg = "챗봇을 초기화하는 중 오류가 발생했습니다. 관리자에게 문의해주세요."
        logger.error("Agent 초기화 실패", error=str(e))
        await say(error_msg)
        return

    # Extract Slack context
    user_id = event.get("user")
    channel_id = event.get("channel")
    user_input = event.get("text")
    thread_ts = event.get("ts")  # Use message ts as thread_ts for the reply

    # If it's already in a thread, use that thread_ts
    if "thread_ts" in event:
        thread_ts = event["thread_ts"]

    logger.info(f"멘션 수신: {user_input} (thread: {thread_ts})")

    # Create LangFuse callback with Slack metadata
    langfuse_handler = get_langfuse_callback(
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
        event_type="slack_mention"
    )

    # Prepare callbacks list (empty if LangFuse disabled)
    callbacks = [langfuse_handler] if langfuse_handler else []

    # Create runnable config with callbacks and metadata
    config = {
        "callbacks": callbacks,
        "metadata": {
            "source": "slack",
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
        }
    }

    inputs = {"question": user_input}

    try:
        # Run the graph with callback (auto-propagates to all nodes)
        logger.info("Agent 실행 시작...")
        result = await agent_app.ainvoke(inputs, config=config)
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
async def handle_message(message, say):
    """
    Handle direct messages or messages where the bot is mentioned (if configured).
    Usually app_mention is preferred for bots in channels.
    This handler might catch all messages if not careful.
    
    매개변수:
        message: Slack message payload
        say: Slack response function
    """
    # Ignore bot's own messages
    if message.get("subtype") is None and message.get("bot_id") is None:
        # For now, let's only respond to mentions to avoid noise,
        # or if it's a DM.
        channel_type = message.get("channel_type")
        if channel_type == "im":
            await handle_app_mention(message, say)
