"""
Loguru 로거 테스트 모듈

로거의 기본 동작, JSON 포맷팅, 예외 처리 등을 테스트합니다.
"""

import json
from pathlib import Path

import pytest
from loguru import logger

from naver_connect_chatbot.config.log import setup_logger
from naver_connect_chatbot.config.settings import settings


class TestLoggerSetup:
    """로거 설정 테스트"""

    def test_logger_is_configured(self) -> None:
        """로거가 제대로 설정되었는지 확인"""
        # 로거에 핸들러가 추가되었는지 확인
        assert len(logger._core.handlers) > 0, "Logger should have at least one handler"

    def test_log_directory_created(self) -> None:
        """로그 디렉토리가 생성되었는지 확인"""
        if settings.logging.enable_file:
            log_dir = Path(settings.logging.log_dir)
            assert log_dir.exists(), f"Log directory {log_dir} should exist"
            assert log_dir.is_dir(), f"Log directory {log_dir} should be a directory"


class TestLoggerBasicFunctionality:
    """로거 기본 기능 테스트"""

    def test_info_log(self) -> None:
        """INFO 레벨 로그 출력 테스트"""
        logger.info("This is an info log")
        logger.info("Info log with context", user_id="12345", action="login")

    def test_debug_log(self) -> None:
        """DEBUG 레벨 로그 출력 테스트"""
        logger.debug("This is a debug log")
        logger.debug("Debug log with data", data={"key": "value", "count": 42})

    def test_warning_log(self) -> None:
        """WARNING 레벨 로그 출력 테스트"""
        logger.warning("This is a warning log")
        logger.warning("Warning with reason", reason="rate limit exceeded")

    def test_error_log(self) -> None:
        """ERROR 레벨 로그 출력 테스트"""
        logger.error("This is an error log")
        logger.error("Error with details", error_code=500, message="Internal server error")

    def test_success_log(self) -> None:
        """SUCCESS 레벨 로그 출력 테스트"""
        logger.success("Operation completed successfully")
        logger.success("Successfully processed request", request_id="abc-123", duration_ms=150)

    def test_critical_log(self) -> None:
        """CRITICAL 레벨 로그 출력 테스트"""
        logger.critical("This is a critical log")
        logger.critical("Critical system failure", component="database", action="restart")


class TestLoggerExceptionHandling:
    """로거 예외 처리 테스트"""

    def test_exception_logging(self) -> None:
        """예외를 포함한 로그 출력 테스트"""
        try:
            # 의도적으로 예외 발생
            result = 1 / 0  # noqa: F841
        except ZeroDivisionError:
            logger.exception("Division by zero error occurred")

    def test_error_with_exception_object(self) -> None:
        """예외 객체를 포함한 에러 로그 테스트"""
        try:
            data = {"key": "value"}
            _ = data["non_existent_key"]
        except KeyError as e:
            logger.error("Key error occurred", key="non_existent_key", error=str(e))


class TestLoggerContextData:
    """로거 컨텍스트 데이터 테스트"""

    def test_log_with_dict_context(self) -> None:
        """딕셔너리 컨텍스트를 포함한 로그 테스트"""
        context = {
            "request_id": "req-12345",
            "user_id": "user-67890",
            "ip_address": "192.168.1.1",
            "method": "POST",
            "path": "/api/v1/users",
        }
        logger.info("API request received", **context)

    def test_log_with_nested_data(self) -> None:
        """중첩된 데이터 구조를 포함한 로그 테스트"""
        nested_data = {
            "user": {
                "id": "12345",
                "name": "홍길동",
                "email": "hong@example.com",
            },
            "metadata": {
                "timestamp": "2025-11-20T10:30:00Z",
                "version": "1.0.0",
            },
        }
        logger.info("Processing user data", data=nested_data)

    def test_log_with_list_data(self) -> None:
        """리스트 데이터를 포함한 로그 테스트"""
        items = ["item1", "item2", "item3"]
        logger.debug("Processing items", items=items, count=len(items))


class TestLoggerFileOutput:
    """로거 파일 출력 테스트"""

    def test_log_file_created(self) -> None:
        """로그 파일이 생성되었는지 확인"""
        if not settings.logging.enable_file:
            pytest.skip("File logging is disabled")

        log_dir = Path(settings.logging.log_dir)
        log_files = list(log_dir.glob("*.log"))
        
        assert len(log_files) > 0, "At least one log file should exist"

    def test_log_file_contains_json(self) -> None:
        """로그 파일에 JSON 형식의 로그가 작성되었는지 확인"""
        if not settings.logging.enable_file:
            pytest.skip("File logging is disabled")

        log_dir = Path(settings.logging.log_dir)
        log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        
        if not log_files:
            pytest.skip("No log files found")
        
        # 가장 최근 로그 파일 읽기
        latest_log = log_files[0]
        
        with latest_log.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "Log file should contain at least one line"
        
        # 첫 번째 줄이 유효한 JSON인지 확인
        try:
            first_log = json.loads(lines[0])
            assert "timestamp" in first_log, "Log should contain timestamp"
            assert "level" in first_log, "Log should contain level"
            assert "message" in first_log, "Log should contain message"
        except json.JSONDecodeError:
            pytest.fail("Log line is not valid JSON")


class TestLoggerPerformance:
    """로거 성능 테스트"""

    def test_bulk_logging_performance(self) -> None:
        """대량 로그 출력 성능 테스트"""
        import time

        start_time = time.time()
        
        # 1000개의 로그 출력
        for i in range(1000):
            logger.debug(f"Performance test log {i}", index=i, batch="test")
        
        elapsed_time = time.time() - start_time
        
        # 1000개 로그가 10초 이내에 처리되어야 함
        assert elapsed_time < 10.0, f"Bulk logging took too long: {elapsed_time:.2f}s"
        
        logger.info(
            "Bulk logging performance test completed",
            total_logs=1000,
            elapsed_time=f"{elapsed_time:.3f}s",
            logs_per_second=f"{1000/elapsed_time:.1f}",
        )


def test_logger_comprehensive_scenario() -> None:
    """종합 시나리오 테스트"""
    logger.info("=" * 50)
    logger.info("Starting comprehensive logger test scenario")
    logger.info("=" * 50)
    
    # 1. 기본 로그 레벨 테스트
    logger.debug("Debug: Application started")
    logger.info("Info: Loading configuration")
    logger.success("Success: Configuration loaded")
    logger.warning("Warning: Deprecated feature used")
    
    # 2. 컨텍스트 데이터와 함께 로그
    logger.info(
        "Processing user request",
        user_id="usr_12345",
        request_type="GET",
        endpoint="/api/users",
    )
    
    # 3. 중첩된 데이터 로그
    response_data = {
        "status": "success",
        "data": {
            "users": [
                {"id": 1, "name": "홍길동"},
                {"id": 2, "name": "김철수"},
            ],
            "total": 2,
        },
    }
    logger.debug("API response prepared", response=response_data)
    
    # 4. 예외 처리 시뮬레이션
    try:
        # 의도적 예외
        raise ValueError("Invalid input value")
    except ValueError:
        logger.exception("Validation error occurred")
    
    # 5. 성공 로그
    logger.success(
        "Request completed successfully",
        duration_ms=123,
        status_code=200,
    )
    
    logger.info("=" * 50)
    logger.info("Comprehensive logger test scenario completed")
    logger.info("=" * 50)


if __name__ == "__main__":
    # 직접 실행 시 종합 시나리오 테스트
    test_logger_comprehensive_scenario()

