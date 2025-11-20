"""
로깅 설정 모듈

Loguru를 활용하여 애플리케이션 전역에서 사용 가능한 로거를 설정합니다.
JSON 형식의 구조화된 로그를 Console 및 File에 출력할 수 있습니다.

사용 예:
    from naver_connect_chatbot.config.log import logger
    
    logger.info("일반 정보 로그")
    logger.debug("디버그 로그", extra_data={"key": "value"})
    logger.error("에러 로그", error=str(e))
    logger.success("성공 로그")
"""

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from naver_connect_chatbot.config.settings import settings

# 기존 핸들러 제거 (Loguru의 기본 stderr 핸들러)
logger.remove()


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    로그 레코드를 JSON 직렬화 가능한 딕셔너리로 변환합니다.
    
    매개변수:
        record: Loguru의 로그 레코드
        
    반환값:
        직렬화 가능한 딕셔너리
    """
    # 기본 필드 추출
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    
    # 추가 필드(extra) 병합
    if record["extra"]:
        subset.update(record["extra"])
    
    # 예외 정보가 있는 경우 포함
    if record["exception"]:
        subset["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
        }
    
    return subset


def _console_json_sink(message: Any) -> None:
    """
    Console용 JSON 싱크 함수
    
    매개변수:
        message: Loguru 메시지 객체
    """
    record = message.record
    serialized = _serialize_record(record)
    sys.stderr.write(json.dumps(serialized, ensure_ascii=False, default=str) + "\n")
    sys.stderr.flush()


def _file_json_sink(file_path: Path) -> callable:
    """
    File용 JSON 싱크 함수 팩토리
    
    매개변수:
        file_path: 로그 파일 경로
        
    반환값:
        싱크 함수
    """
    def sink(message: Any) -> None:
        record = message.record
        serialized = _serialize_record(record)
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serialized, ensure_ascii=False, default=str) + "\n")
    
    return sink


def setup_logger() -> None:
    """
    애플리케이션 로거를 설정합니다.
    
    Settings의 logging 설정을 기반으로 Console 및 File 핸들러를 구성합니다.
    
    - Console: 설정에 따라 JSON 또는 컬러 텍스트 형식
    - File: 항상 JSON 형식으로 저장
    - 로그 파일은 자동으로 로테이션 및 압축됩니다
    
    예외:
        ValueError: 로그 레벨이 올바르지 않은 경우
    """
    log_config = settings.logging
    
    # 로그 레벨 검증
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_config.level.upper() not in valid_levels:
        msg = f"Invalid log level: {log_config.level}. Must be one of {valid_levels}"
        raise ValueError(msg)
    
    # Console 핸들러 추가
    if log_config.enable_console:
        if log_config.json_format:
            # JSON 형식 출력
            logger.add(
                _console_json_sink,
                level=log_config.level.upper(),
                backtrace=True,
                diagnose=True,
                enqueue=True,
            )
        else:
            # 사람이 읽기 쉬운 형식 출력
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=log_config.level.upper(),
                colorize=True,
                backtrace=True,
                diagnose=True,
                enqueue=True,
            )
    
    # File 핸들러 추가
    if log_config.enable_file:
        # 로그 디렉토리 생성
        log_dir = Path(log_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로 설정
        log_file_path = log_dir / "app_{time:YYYY-MM-DD}.log"
        
        if log_config.serialize:
            # Loguru의 내장 직렬화 사용
            logger.add(
                str(log_file_path),
                level=log_config.level.upper(),
                rotation=log_config.rotation,
                retention=log_config.retention,
                compression=log_config.compression,
                serialize=True,
                backtrace=True,
                diagnose=True,
                enqueue=True,
            )
        else:
            # 커스텀 JSON 싱크 함수 사용
            def file_sink(message: Any) -> None:
                record = message.record
                serialized = _serialize_record(record)
                # 파일명에서 날짜 추출 및 생성
                current_date = record["time"].strftime("%Y-%m-%d")
                actual_file_path = log_dir / f"app_{current_date}.log"
                with actual_file_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(serialized, ensure_ascii=False, default=str) + "\n")
            
            logger.add(
                file_sink,
                level=log_config.level.upper(),
                backtrace=True,
                diagnose=True,
                enqueue=True,
            )
    
    logger.info(
        "Logger initialized",
        level=log_config.level,
        console_enabled=log_config.enable_console,
        file_enabled=log_config.enable_file,
        json_format=log_config.json_format,
    )


# 모듈 임포트 시 자동으로 로거 설정
setup_logger()

def get_logger() -> "logger":
    """
    전역 로거 인스턴스를 반환합니다.

    이 함수는 모듈에서 logger를 import하는 대안으로 사용할 수 있습니다.
    주로 테스트나 동적 로거 접근이 필요한 경우에 유용합니다.

    반환값:
        Loguru logger 인스턴스

    예시:
        from naver_connect_chatbot.config.log import get_logger
        logger = get_logger()
        logger.info("메시지")
    """
    return logger


# 전역 로거 인스턴스 노출
__all__ = ["logger", "setup_logger", "get_logger"]
