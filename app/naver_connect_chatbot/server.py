"""
FastAPI 웹 서버 모듈

Slack Bot을 위한 FastAPI 웹 서버를 제공합니다.
Socket Mode와 HTTP Mode를 모두 지원합니다.

Socket Mode (권장):
    - SLACK_APP_TOKEN 환경변수 설정 시 자동 활성화
    - URL 설정 불필요, WebSocket 기반
    - IP 주소로도 사용 가능

HTTP Mode (기존):
    - SLACK_APP_TOKEN 미설정 시 사용
    - Slack Events API URL 설정 필요
    - HTTPS 도메인 필요

참고 문서:
    - https://github.com/slackapi/bolt-python/tree/main/examples/fastapi
    - https://api.slack.com/apis/connections/socket-mode
"""

import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from naver_connect_chatbot.config.log import get_logger
from naver_connect_chatbot.config.settings.main import settings
from naver_connect_chatbot.config.settings.base import PROJECT_ROOT
from naver_connect_chatbot.slack import app as slack_app
from naver_connect_chatbot.slack.handler import set_checkpointer

# Logging setup
logger = get_logger()

# Socket Mode 사용 여부
USE_SOCKET_MODE = settings.slack.app_token is not None


# Checkpointer database path
CHECKPOINTER_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션 라이프사이클 관리

    서버 시작 시 초기화 작업을 수행하고,
    서버 종료 시 정리 작업을 수행합니다.

    포함된 초기화 작업:
    - AsyncSqliteSaver checkpointer 초기화 (대화 상태 저장)
    - BM25 인덱스 자동 복구
    - VectorDB 스키마 로드
    """
    # ========================================================================
    # Startup
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Naver Connect Chatbot 서버 시작")
    logger.info(f"포트: {settings.slack.port}")
    logger.info(f"로그 레벨: {settings.logging.level}")
    logger.info(f"Slack 모드: {'Socket Mode ✅' if USE_SOCKET_MODE else 'HTTP Mode (Events API)'}")
    logger.info("=" * 80)

    # Checkpointer 데이터베이스 디렉토리 생성
    CHECKPOINTER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # AsyncSqliteSaver 초기화
    logger.info(f"Checkpointer 초기화 중... (DB: {CHECKPOINTER_DB_PATH})")
    sqlite_conn = await aiosqlite.connect(str(CHECKPOINTER_DB_PATH))
    checkpointer = AsyncSqliteSaver(sqlite_conn)
    await checkpointer.setup()  # 테이블 생성
    set_checkpointer(checkpointer)
    logger.info("✓ Checkpointer 초기화 완료 - 대화 상태가 SQLite에 저장됩니다")

    # BM25 인덱스 자동 복구
    bm25_index_path = PROJECT_ROOT / settings.retriever.bm25_index_path
    if not bm25_index_path.exists():
        logger.warning(f"BM25 인덱스가 없습니다: {bm25_index_path}")
        logger.info("BM25 통합 인덱스 재생성을 시도합니다...")
        try:
            subprocess.run(
                [sys.executable, "document_processing/rebuild_unified_bm25.py"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=False,
            )
            logger.info("BM25 통합 인덱스 재생성 완료")
        except subprocess.CalledProcessError as e:
            logger.error(f"BM25 인덱스 재생성 실패: {e}")
            logger.warning("Qdrant Dense 검색만 사용합니다.")
        except FileNotFoundError:
            logger.error("rebuild_unified_bm25.py 스크립트를 찾을 수 없습니다.")
            logger.warning("Qdrant Dense 검색만 사용합니다.")
    else:
        logger.info(f"BM25 인덱스 확인: {bm25_index_path}")

    # VectorDB 스키마 로드 (Pre-Retriever 데이터 소스 선택용)
    logger.info("VectorDB 스키마 로딩 중...")
    try:
        from qdrant_client import QdrantClient

        from naver_connect_chatbot.rag.schema_registry import SchemaRegistry

        qdrant_url = settings.qdrant_vector_store.url
        qdrant_api_key = (
            settings.qdrant_vector_store.api_key.get_secret_value() if settings.qdrant_vector_store.api_key else None
        )
        collection_name = settings.qdrant_vector_store.collection_name

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        registry = SchemaRegistry.get_instance()
        schema = registry.load_from_qdrant(client, collection_name)

        logger.info(
            f"VectorDB 스키마 로드 완료: {schema.total_documents:,}개 문서, {len(schema.data_sources)}개 데이터 소스"
        )
    except Exception as e:
        logger.warning(f"VectorDB 스키마 로드 실패: {e}")
        logger.warning("기본 필터링만 사용합니다.")

    # Slack Bot Token 검증 (auth.test)
    logger.info("Slack Bot Token 검증 중...")
    try:
        auth_info = await slack_app.client.auth_test()
        logger.info(
            f"✅ Slack Bot 연결 성공: {auth_info['team']} (Team ID: {auth_info['team_id']}, Bot ID: {auth_info['user_id']})"
        )
    except Exception as e:
        logger.error(f"❌ Slack Bot 연결 실패: {e}")
        logger.error("SLACK_BOT_TOKEN이 올바른지, 해당 워크스페이스 토큰인지 확인해주세요.")

    # Socket Mode 시작 (SLACK_APP_TOKEN 설정 시)
    socket_mode_handler = None
    socket_mode_task = None
    if USE_SOCKET_MODE:
        logger.info("Socket Mode 연결 시작...")
        try:
            socket_mode_handler = AsyncSocketModeHandler(
                slack_app,
                settings.slack.app_token.get_secret_value(),
            )
            # 백그라운드 태스크로 실행 (블로킹 방지)
            socket_mode_task = asyncio.create_task(socket_mode_handler.start_async())
            logger.info("✓ Socket Mode 연결 완료 - Slack 이벤트 수신 준비됨")
        except Exception as e:
            logger.error(f"Socket Mode 연결 실패: {e}")
            logger.warning("HTTP Mode로 fallback합니다. Event Subscriptions URL 설정이 필요합니다.")

    yield

    # ========================================================================
    # Shutdown
    # ========================================================================
    logger.info("=" * 80)
    logger.info("서버 종료 중...")
    logger.info("=" * 80)

    # Socket Mode 정리
    if socket_mode_handler is not None:
        try:
            logger.info("Socket Mode 연결 종료 중...")
            await socket_mode_handler.close_async()
            if socket_mode_task is not None:
                socket_mode_task.cancel()
                try:
                    await socket_mode_task
                except asyncio.CancelledError:
                    pass
            logger.info("✓ Socket Mode 연결 종료 완료")
        except Exception as e:
            logger.warning(f"Socket Mode 종료 중 오류 발생: {e}")

    # Slack handler의 global agent app cleanup
    try:
        from naver_connect_chatbot.slack.handler import _agent_app

        if _agent_app is not None:
            logger.info("Agent app 리소스 정리 중...")
            # LangGraph app은 특별한 cleanup이 필요 없지만,
            # 향후 확장을 위해 로깅만 수행
            logger.info("✓ Agent app 정리 완료")
    except Exception as e:
        logger.warning(f"Agent app 정리 중 오류 발생: {e}")

    # Checkpointer 정리 (SQLite connection 닫기)
    try:
        logger.info("Checkpointer SQLite connection 정리 중...")
        set_checkpointer(None)  # 전역 checkpointer 해제
        await sqlite_conn.close()
        logger.info("✓ Checkpointer SQLite connection 정리 완료")
    except Exception as e:
        logger.warning(f"Checkpointer 정리 중 오류 발생: {e}")

    logger.info("=" * 80)
    logger.info("서버 종료 완료")
    logger.info("=" * 80)


# FastAPI 앱 생성
api = FastAPI(
    title="Naver Connect Chatbot",
    description="Slack Bot for Naver Connect documentation Q&A",
    version="0.0.1",
    lifespan=lifespan,
)

# Slack Request Handler
slack_handler = AsyncSlackRequestHandler(slack_app)


@api.get("/")
async def root():
    """
    루트 엔드포인트 - 서버 상태 확인

    반환값:
        dict: 서버 상태 정보
    """
    return {
        "service": "Naver Connect Chatbot",
        "status": "running",
        "version": "0.0.1",
    }


@api.get("/health")
async def health():
    """
    헬스체크 엔드포인트

    실제 컴포넌트 상태를 확인하여 반환합니다:
    - agent: RAG 에이전트 초기화 상태
    - qdrant: 벡터 DB 연결 상태 (선택적)

    반환값:
        dict: 헬스체크 결과 (status, checks, details)
    """
    from naver_connect_chatbot.slack.handler import _agent_app, _agent_init_failed

    checks = {}
    details = {}
    overall_healthy = True

    # 1. Agent 상태 확인
    if _agent_init_failed:
        checks["agent"] = "failed"
        details["agent"] = "Agent 초기화 실패. 서버 재시작 필요."
        overall_healthy = False
    elif _agent_app is not None:
        checks["agent"] = "ready"
    else:
        checks["agent"] = "not_initialized"
        details["agent"] = "아직 초기화되지 않음 (첫 요청 시 초기화)"

    # 2. Qdrant 연결 확인 (선택적 - 비용 절감을 위해 비활성화 가능)
    try:
        from qdrant_client import QdrantClient

        qdrant_url = settings.qdrant_vector_store.url
        qdrant_api_key = (
            settings.qdrant_vector_store.api_key.get_secret_value() if settings.qdrant_vector_store.api_key else None
        )
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5.0)
        collections = client.get_collections()
        checks["qdrant"] = "connected"
        details["qdrant"] = f"{len(collections.collections)}개 컬렉션"
    except Exception as e:
        checks["qdrant"] = "unreachable"
        details["qdrant"] = str(e)
        # Qdrant 연결 실패는 warning으로 처리 (agent가 실패하지 않았다면)
        if overall_healthy:
            overall_healthy = False

    status = "healthy" if overall_healthy else "unhealthy"

    return {
        "status": status,
        "checks": checks,
        "details": details if details else None,
    }


@api.post("/slack/events")
async def slack_events(req: Request):
    """
    Slack Events API 엔드포인트 (HTTP Mode용)

    Slack으로부터 이벤트를 수신하고 처리합니다.
    URL Verification과 Event Callback을 처리합니다.

    Note:
        Socket Mode 사용 시 이 엔드포인트는 사용되지 않습니다.
        Socket Mode는 WebSocket으로 직접 이벤트를 수신합니다.

    매개변수:
        req: FastAPI Request 객체

    반환값:
        Slack API 응답

    예외:
        HTTPException: Slack 요청 검증 실패 시
    """
    if USE_SOCKET_MODE:
        logger.warning("Socket Mode 활성화 상태에서 HTTP 이벤트 수신됨. Slack App 설정 확인 필요.")
    logger.debug("Slack 이벤트 수신 (HTTP Mode)")
    return await slack_handler.handle(req)


# 서버 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "naver_connect_chatbot.server:api",
        host="0.0.0.0",
        port=settings.slack.port,
        log_level=settings.logging.level.lower(),
        reload=False,  # Production에서는 False
    )
