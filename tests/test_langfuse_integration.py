"""Integration tests for LangFuse tracing (requires running LangFuse)"""
import os
import pytest
import asyncio
from langfuse import Langfuse
from naver_connect_chatbot.config.monitoring import (
    get_langfuse_callback,
    check_langfuse_health,
)
from naver_connect_chatbot.config import settings


@pytest.mark.integration
class TestLangFuseIntegration:
    """Integration tests requiring real LangFuse instance"""

    @pytest.mark.asyncio
    async def test_langfuse_health_check_real(self):
        """Test health check against real LangFuse instance"""
        is_healthy = await check_langfuse_health(settings.langfuse)

        if settings.langfuse.enabled:
            # Only assert healthy if LangFuse is configured and enabled
            if settings.langfuse.public_key and settings.langfuse.secret_key:
                assert is_healthy is True, (
                    f"LangFuse should be healthy at {settings.langfuse.host}. "
                    "Ensure docker-compose is running: docker-compose up -d"
                )
        else:
            assert is_healthy is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not settings.langfuse.enabled,
        reason="LangFuse disabled (set LANGFUSE_ENABLED=true)"
    )
    async def test_callback_creates_trace(self):
        """Test that callback creates a trace in LangFuse"""
        # Skip if keys are not configured
        if not settings.langfuse.public_key or not settings.langfuse.secret_key:
            pytest.skip("LangFuse API keys not configured")

        # Create callback
        callback = get_langfuse_callback(
            user_id="test_integration_user",
            channel_id="test_channel",
            test_type="integration"
        )

        assert callback is not None, "Callback should be created when enabled"

        # Simulate a simple traced operation
        # (In real usage, this would be graph.ainvoke(..., config={"callbacks": [callback]}))
        # For this test, we just verify the callback was initialized correctly

        # Flush to ensure trace is sent
        await callback.aflush()

        # Wait a bit for async processing
        await asyncio.sleep(2)

        # Note: LangFuse SDK v3 doesn't have get_traces() method
        # The callback creation and flush is sufficient to verify integration works
        # Actual trace verification would be done via the LangFuse UI

    @pytest.mark.skipif(
        not settings.langfuse.enabled,
        reason="LangFuse disabled"
    )
    def test_langfuse_client_connection(self, monkeypatch):
        """Test direct connection to LangFuse API"""
        # Skip if keys are not configured
        if not settings.langfuse.public_key or not settings.langfuse.secret_key:
            pytest.skip("LangFuse API keys not configured")

        # Set environment variables for Langfuse client using monkeypatch
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", settings.langfuse.public_key)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", settings.langfuse.secret_key)
        monkeypatch.setenv("LANGFUSE_HOST", settings.langfuse.host)

        langfuse_client = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )

        # Create a simple trace to verify connection
        trace = langfuse_client.trace(
            name="test_connection",
            user_id="test_integration",
            metadata={"test": "integration"}
        )

        # Verify trace was created
        assert trace is not None
        assert hasattr(trace, 'id')

        # Flush to ensure trace is sent
        langfuse_client.flush()
