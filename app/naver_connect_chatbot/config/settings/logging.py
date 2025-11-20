"""
로깅 설정 모듈

이 모듈은 애플리케이션의 로깅 동작을 제어하는 설정을 관리합니다.
Loguru 라이브러리를 활용한 구조화된 로깅을 지원합니다.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base import PROJECT_ROOT


class LoggingSettings(BaseSettings):
    """
    로깅 설정
    
    환경변수 prefix: LOG_
    예: LOG_LEVEL=INFO
    """
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    level: str = Field(
        default="INFO",
        description="로그 레벨 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)"
    )
    enable_console: bool = Field(
        default=True,
        description="콘솔 로그 출력 활성화"
    )
    enable_file: bool = Field(
        default=True,
        description="파일 로그 출력 활성화"
    )
    log_dir: str = Field(
        default="logs",
        description="로그 파일 저장 디렉토리"
    )
    rotation: str = Field(
        default="100 MB",
        description="로그 파일 로테이션 크기 (예: 100 MB, 1 GB)"
    )
    retention: str = Field(
        default="30 days",
        description="로그 파일 보관 기간 (예: 30 days, 1 week)"
    )
    compression: str = Field(
        default="zip",
        description="로그 파일 압축 형식 (gz, zip, tar, tar.gz 등)"
    )
    json_format: bool = Field(
        default=True,
        description="JSON 형태로 로그 출력"
    )
    serialize: bool = Field(
        default=False,
        description="로그를 완전히 직렬화하여 파싱 가능한 JSON으로 출력"
    )


__all__ = [
    "LoggingSettings",
]

