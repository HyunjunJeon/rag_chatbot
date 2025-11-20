"""
Slack Bot 패키지

이 패키지는 Slack Bot의 이벤트 핸들링과 관련된 기능을 제공합니다.

주요 구성 요소:
- app: Slack Bolt AsyncApp 인스턴스 (이벤트 핸들러 포함)
- get_agent_app: LangGraph 에이전트 애플리케이션 초기화 함수

사용 예시:
    >>> from naver_connect_chatbot.slack import app
    >>> # FastAPI와 함께 사용
    >>> from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
    >>> handler = AsyncSlackRequestHandler(app)
"""

from naver_connect_chatbot.slack.handler import app, get_agent_app

__all__ = [
    "app",
    "get_agent_app",
]

