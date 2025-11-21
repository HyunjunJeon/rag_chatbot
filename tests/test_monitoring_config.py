"""Tests for LangFuse monitoring configuration"""
import pytest
from unittest.mock import AsyncMock, patch, Mock
from naver_connect_chatbot.config.monitoring import (
    LangfuseSettings,
    check_langfuse_health,
    get_langfuse_callback,
)


class TestLangfuseSettings:
    """Test LangfuseSettings Pydantic model"""

    def test_default_values(self):
        """Test default configuration values"""
        settings = LangfuseSettings()
        assert settings.host == "http://langfuse-web:3000"
        assert settings.enabled is True
        assert settings.health_check_timeout == 2.0

    def test_disabled_via_env(self, monkeypatch):
        """Test disabling LangFuse via environment variable"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        settings = LangfuseSettings()
        assert settings.enabled is False

    def test_custom_host_via_env(self, monkeypatch):
        """Test custom host via environment variable"""
        monkeypatch.setenv("LANGFUSE_HOST", "http://custom:3000")
        settings = LangfuseSettings()
        assert settings.host == "http://custom:3000"


@pytest.mark.asyncio
class TestHealthCheck:
    """Test LangFuse health check functionality"""

    async def test_health_check_success(self):
        """Test successful health check"""
        settings = LangfuseSettings(
            host="http://test:3000",
            enabled=True
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await check_langfuse_health(settings)
            assert result is True

    async def test_health_check_disabled(self):
        """Test health check returns False when disabled"""
        settings = LangfuseSettings(enabled=False)
        result = await check_langfuse_health(settings)
        assert result is False

    async def test_health_check_timeout(self):
        """Test health check handles timeout gracefully"""
        settings = LangfuseSettings(
            host="http://nonexistent:3000",
            health_check_timeout=0.1
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = TimeoutError()

            result = await check_langfuse_health(settings)
            assert result is False

    async def test_health_check_connection_error(self):
        """Test health check handles connection errors gracefully"""
        settings = LangfuseSettings(host="http://nonexistent:3000")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection refused")

            result = await check_langfuse_health(settings)
            assert result is False


class TestCallbackFactory:
    """Test get_langfuse_callback factory function"""

    def test_callback_creation_disabled(self, monkeypatch):
        """Test callback returns None when disabled"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")

        # Mock settings import within the function
        with patch("naver_connect_chatbot.config.settings") as mock_settings:
            mock_settings.langfuse = LangfuseSettings(enabled=False)
            callback = get_langfuse_callback()
            assert callback is None

    def test_callback_creation_with_metadata(self, monkeypatch):
        """Test callback creation with Slack metadata"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        with patch("naver_connect_chatbot.config.settings") as mock_settings:
            mock_settings.langfuse = LangfuseSettings(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test"
            )

            with patch("naver_connect_chatbot.config.monitoring.CallbackHandler") as mock_handler:
                mock_instance = Mock()
                mock_handler.return_value = mock_instance

                callback = get_langfuse_callback(
                    user_id="U12345",
                    channel_id="C67890",
                    custom_field="value"
                )

                # Verify CallbackHandler was instantiated
                mock_handler.assert_called_once()
                assert callback == mock_instance

                # Verify metadata was set
                assert mock_instance.set_trace_params.called

    def test_callback_creation_handles_exception(self, monkeypatch):
        """Test callback returns None when creation fails"""
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")

        with patch("naver_connect_chatbot.config.settings") as mock_settings:
            mock_settings.langfuse = LangfuseSettings(enabled=True)

            with patch("naver_connect_chatbot.config.monitoring.CallbackHandler") as mock_handler:
                mock_handler.side_effect = Exception("API error")

                callback = get_langfuse_callback()
                assert callback is None
