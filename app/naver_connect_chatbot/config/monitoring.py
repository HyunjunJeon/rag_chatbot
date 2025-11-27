"""LangFuse monitoring configuration and callback factory."""

from typing import Optional
import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langfuse.langchain import CallbackHandler
from loguru import logger


class LangfuseSettings(BaseSettings):
    """LangFuse monitoring configuration.

    Environment Variables:
        LANGFUSE_HOST: LangFuse server URL (default: http://langfuse-web:3000)
        LANGFUSE_PUBLIC_KEY: API public key from LangFuse UI
        LANGFUSE_SECRET_KEY: API secret key from LangFuse UI
        LANGFUSE_ENABLED: Enable/disable monitoring (default: true)
        LANGFUSE_HEALTH_CHECK_TIMEOUT: Health check timeout in seconds (default: 2.0)

    Configuration files loaded in order:
        1. .env (main application config)
        2. .env.langfuse (LangFuse-specific secrets)
    """

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=(".env", ".env.langfuse"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="http://langfuse-web:3000", description="LangFuse server URL")
    public_key: str = Field(default="", description="LangFuse API public key")
    secret_key: str = Field(default="", description="LangFuse API secret key")
    enabled: bool = Field(default=True, description="Enable/disable LangFuse monitoring")
    health_check_timeout: float = Field(default=2.0, description="Health check timeout in seconds")


async def check_langfuse_health(settings: LangfuseSettings) -> bool:
    """Check LangFuse server health.

    Args:
        settings: LangFuse configuration settings

    Returns:
        True if LangFuse is healthy and enabled, False otherwise

    Note:
        Returns False (not exception) on any error for graceful degradation.
    """
    if not settings.enabled:
        logger.debug("LangFuse monitoring is disabled")
        return False

    try:
        async with httpx.AsyncClient(timeout=settings.health_check_timeout) as client:
            response = await client.get(f"{settings.host}/api/public/health")
            is_healthy = response.status_code == 200

            if is_healthy:
                logger.debug(f"LangFuse health check passed: {settings.host}")
            else:
                logger.warning(
                    f"LangFuse health check failed: {settings.host} (status={response.status_code})"
                )

            return is_healthy

    except Exception as e:
        logger.warning(f"LangFuse health check failed: {e}")
        return False


def get_langfuse_callback(
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    **metadata,
) -> Optional[CallbackHandler]:
    """Create LangFuse callback handler with Slack metadata.

    This factory function creates a CallbackHandler for tracing LangChain/LangGraph
    executions. The callback automatically propagates through graph.ainvoke() to
    all node LLM calls.

    Args:
        user_id: Slack user ID for trace tagging
        channel_id: Slack channel ID for trace metadata
        **metadata: Additional metadata to attach to traces

    Returns:
        CallbackHandler instance if LangFuse is enabled and credentials valid,
        None otherwise (allows graceful degradation)

    Example:
        >>> callback = get_langfuse_callback(
        ...     user_id="U12345",
        ...     channel_id="C67890",
        ...     event_type="slack_mention"
        ... )
        >>> if callback:
        ...     result = await graph.ainvoke(
        ...         {"question": "..."},
        ...         config={"callbacks": [callback]}
        ...     )

    Note:
        - Returns None if LANGFUSE_ENABLED=false
        - Returns None if CallbackHandler creation fails
        - Logs errors but never raises exceptions (graceful degradation)
    """
    # Import here to avoid circular dependency
    from naver_connect_chatbot.config import settings

    langfuse_settings = settings.langfuse

    if not langfuse_settings.enabled:
        logger.debug("LangFuse callback not created: monitoring disabled")
        return None

    try:
        # Set environment variables for CallbackHandler
        # LangFuse SDK v3 requires environment variables to be set for proper
        # initialization of the CallbackHandler. While public_key can be passed
        # explicitly, the SDK internally reads SECRET_KEY and HOST from env vars
        # for authentication and connection configuration.
        import os

        os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_settings.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = langfuse_settings.secret_key
        os.environ["LANGFUSE_HOST"] = langfuse_settings.host

        # Create handler (reads from env vars)
        handler = CallbackHandler(
            public_key=langfuse_settings.public_key,  # Can also be passed explicitly
        )

        # Set Slack user ID for trace filtering
        if user_id:
            handler.set_trace_params(user_id=user_id)

        # Set additional metadata (channel, custom fields)
        if channel_id or metadata:
            trace_metadata = {}
            if channel_id:
                trace_metadata["channel_id"] = channel_id
            trace_metadata.update(metadata)
            handler.set_trace_params(metadata=trace_metadata)

        logger.debug(
            f"LangFuse callback created successfully (user_id={user_id}, channel_id={channel_id})"
        )
        return handler

    except Exception as e:
        logger.error(f"Failed to create LangFuse callback: {e}")
        return None
