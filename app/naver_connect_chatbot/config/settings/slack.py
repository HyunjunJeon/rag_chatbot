"""
Slack App 설정 모듈

Slack Bot 연동에 필요한 토큰 및 서버 설정을 관리합니다.

Environment Variables:
    SLACK_BOT_TOKEN: Slack Bot User OAuth Token (xoxb-로 시작)
    SLACK_SIGNING_SECRET: Slack App Signing Secret
    SLACK_APP_TOKEN: Slack App-Level Token (Socket Mode용, xapp-로 시작) - 선택사항
    SLACK_PORT: Slack App 서버 포트 (기본값: 3000)
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlackSettings(BaseSettings):
    """
    Slack App 설정
    
    Slack Bolt를 사용한 대화형 챗봇 구현에 필요한 설정을 제공합니다.
    
    속성:
        bot_token: Slack Bot User OAuth Token (xoxb-로 시작)
        signing_secret: Slack App Signing Secret (요청 검증용)
        app_token: Slack App-Level Token (Socket Mode용, xapp-로 시작) - 선택사항
        port: Slack App 서버 포트
    """
    
    model_config = SettingsConfigDict(
        env_prefix="SLACK_",
        case_sensitive=False,
    )
    
    bot_token: SecretStr = Field(
        ...,
        description="Slack Bot User OAuth Token (xoxb-로 시작)",
    )
    signing_secret: SecretStr = Field(
        ...,
        description="Slack App Signing Secret",
    )
    app_token: SecretStr | None = Field(
        default=None,
        description="Slack App-Level Token (Socket Mode용, xapp-로 시작)",
    )
    port: int = Field(
        default=3000,
        description="Slack App 서버 포트",
        ge=1,
        le=65535,
    )


__all__ = ["SlackSettings"]

