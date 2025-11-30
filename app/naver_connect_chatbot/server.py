"""
FastAPI 웹 서버 모듈

Slack Bot을 위한 FastAPI 웹 서버를 제공합니다.
Slack Events API 엔드포인트와 헬스체크 엔드포인트를 포함합니다.

참고 문서:
    - https://github.com/slackapi/bolt-python/tree/main/examples/fastapi
    - https://api.slack.com/apis/connections/events-api
"""

import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from naver_connect_chatbot.config.log import get_logger
from naver_connect_chatbot.config.settings.main import settings
from naver_connect_chatbot.config.settings.base import PROJECT_ROOT
from naver_connect_chatbot.slack import app as slack_app

# Logging setup
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션 라이프사이클 관리

    서버 시작 시 초기화 작업을 수행하고,
    서버 종료 시 정리 작업을 수행합니다.
    """
    # ========================================================================
    # Startup
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Naver Connect Chatbot 서버 시작")
    logger.info(f"포트: {settings.slack.port}")
    logger.info(f"로그 레벨: {settings.logging.level}")
    logger.info("=" * 80)

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

    yield

    # ========================================================================
    # Shutdown
    # ========================================================================
    logger.info("=" * 80)
    logger.info("서버 종료 중...")
    logger.info("=" * 80)

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

    # 기타 전역 리소스 cleanup (필요 시 추가)
    # 예: HTTP clients, database connections, cache 등

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
            settings.qdrant_vector_store.api_key.get_secret_value()
            if settings.qdrant_vector_store.api_key
            else None
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
    Slack Events API 엔드포인트

    Slack으로부터 이벤트를 수신하고 처리합니다.
    URL Verification과 Event Callback을 처리합니다.

    매개변수:
        req: FastAPI Request 객체

    반환값:
        Slack API 응답

    예외:
        HTTPException: Slack 요청 검증 실패 시
    """
    logger.debug("Slack 이벤트 수신")
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
