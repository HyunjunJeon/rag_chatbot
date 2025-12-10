"""테스트 전역 설정."""

# pytest-asyncio 플러그인을 명시적으로 로드하여 asyncio 테스트를 지원합니다.
pytest_plugins = ("pytest_asyncio",)

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from loguru import logger

# ============================================================================
# Test Logging Configuration
# ============================================================================

# 테스트 로그 디렉토리
TEST_LOG_DIR = Path(__file__).parent.parent / "logs" / "tests"

# 세션 레벨에서 공유할 로그 파일 경로 (전역 변수)
_test_log_file: Path | None = None


def get_test_log_file() -> Path | None:
    """현재 테스트 세션의 로그 파일 경로를 반환합니다."""
    return _test_log_file


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """
    세션 스코프 fixture로 테스트 로깅을 설정합니다.

    - evaluation 컨텍스트가 바인딩된 로그만 파일에 저장
    - 앱 로거와 분리되어 테스트 로그만 별도 파일에 기록
    """
    global _test_log_file

    # 로그 디렉토리 생성
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 타임스탬프 기반 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _test_log_file = TEST_LOG_DIR / f"evaluation_{timestamp}.log"

    # 테스트 전용 파일 핸들러 추가
    handler_id = logger.add(
        str(_test_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {message}",
        level="DEBUG",
        filter=lambda record: "evaluation" in record["extra"].get("context", ""),
        encoding="utf-8",
        enqueue=True,
    )

    # 테스트 로거 초기화 로그
    test_logger = logger.bind(context="evaluation")
    test_logger.info(f"Test logging initialized: {_test_log_file}")

    yield

    # 세션 종료 시 정리
    test_logger.info("Test session completed")
    logger.remove(handler_id)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """async def 테스트를 asyncio.run으로 강제 실행합니다."""
    testfunction = pyfuncitem.obj
    import inspect
    if inspect.iscoroutinefunction(testfunction):
        sig = inspect.signature(testfunction)
        kwargs = {name: pyfuncitem.funcargs[name] for name in sig.parameters}
        asyncio.run(testfunction(**kwargs))
        return True
    return None
