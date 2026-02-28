"""
B-1: Config & Settings 단위 테스트.

GeminiLLMSettings 기본값, 환경변수 로드, get_chat_model() 팩토리를 검증합니다.
API 키 없이 ChatGoogleGenerativeAI 생성자를 mock하여 실행됩니다.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))


# ============================================================================
# B-1-1: GeminiLLMSettings 기본값 검증
# ============================================================================


class TestGeminiLLMSettingsDefaults:
    """GeminiLLMSettings 기본값 테스트"""

    def test_default_model(self):
        """기본 모델명 검증"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None)
        assert settings.model == "gemini-3.1-pro-preview"

    def test_default_temperature(self):
        """기본 temperature 검증 (Gemini 3 권장값 1.0)"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None)
        assert settings.temperature == 1.0

    def test_default_max_output_tokens_is_none(self):
        """기본 max_output_tokens는 None (모델 기본값 사용)"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None)
        assert settings.max_output_tokens is None

    def test_temperature_bounds(self):
        """temperature 범위 검증 (0.0 ~ 2.0)"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        # 유효한 범위
        settings_low = GeminiLLMSettings(google_api_key=None, temperature=0.0)
        assert settings_low.temperature == 0.0

        settings_high = GeminiLLMSettings(google_api_key=None, temperature=2.0)
        assert settings_high.temperature == 2.0


# ============================================================================
# B-1-2: GeminiLLMSettings 환경변수 로드
# ============================================================================


class TestGeminiLLMSettingsEnvVars:
    """GeminiLLMSettings 환경변수 로드 테스트"""

    def test_api_key_loaded_from_env(self, monkeypatch):
        """GOOGLE_API_KEY 환경변수에서 api_key 로드"""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-api-key-12345")
        # model_config에서 env_file을 참조하지 않도록 직접 환경변수 주입

        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key="test-google-api-key-12345")
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "test-google-api-key-12345"

    def test_model_override_via_constructor(self):
        """model 필드 직접 설정"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None, model="gemini-2.5-pro")
        assert settings.model == "gemini-2.5-pro"

    def test_api_key_none_by_default(self):
        """API 키가 없으면 None"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None)
        assert settings.api_key is None

    def test_max_output_tokens_set(self):
        """max_output_tokens 설정 가능"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        settings = GeminiLLMSettings(google_api_key=None, max_output_tokens=4096)
        assert settings.max_output_tokens == 4096


# ============================================================================
# B-1-3: get_chat_model() → ChatGoogleGenerativeAI 인스턴스 반환
# ============================================================================


class TestGetChatModelFactory:
    """get_chat_model() 팩토리 함수 테스트"""

    def _make_mock_settings(self, api_key: str = "fake-key", **gemini_kwargs):
        """테스트용 mock settings 생성 헬퍼"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        mock_settings = MagicMock()
        mock_settings.gemini_llm = GeminiLLMSettings(google_api_key=api_key, **gemini_kwargs)
        return mock_settings

    def test_returns_chat_google_generative_ai_instance(self):
        """get_chat_model()이 ChatGoogleGenerativeAI 인스턴스를 반환"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            MockLLM.return_value = mock_instance

            from naver_connect_chatbot.config.llm import get_chat_model

            result = get_chat_model(settings_obj=mock_settings)

            assert result is mock_instance
            assert MockLLM.called

    def test_model_passed_to_constructor(self):
        """model 파라미터가 생성자에 전달됨"""
        mock_settings = self._make_mock_settings(model="gemini-3.1-pro-preview")

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings)

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs["model"] == "gemini-3.1-pro-preview"

    def test_api_key_passed_as_google_api_key(self):
        """api_key가 google_api_key로 생성자에 전달됨"""
        mock_settings = self._make_mock_settings(api_key="my-secret-key")

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings)

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs["google_api_key"] == "my-secret-key"

    def test_temperature_passed_to_constructor(self):
        """temperature가 생성자에 전달됨"""
        mock_settings = self._make_mock_settings(temperature=1.0)

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings)

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs["temperature"] == 1.0

    def test_raises_value_error_without_api_key(self):
        """API 키 없으면 ValueError 발생"""
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        mock_settings = MagicMock()
        mock_settings.gemini_llm = GeminiLLMSettings(google_api_key=None)

        from naver_connect_chatbot.config.llm import get_chat_model

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            get_chat_model(settings_obj=mock_settings)


# ============================================================================
# B-1-4: get_chat_model(thinking_level="low") vs 기본값 설정 차이
# ============================================================================


class TestGetChatModelThinkingLevel:
    """thinking_level 파라미터 테스트"""

    def _make_mock_settings(self, api_key: str = "fake-key"):
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        mock_settings = MagicMock()
        mock_settings.gemini_llm = GeminiLLMSettings(google_api_key=api_key)
        return mock_settings

    def test_thinking_level_low_passed_to_constructor(self):
        """thinking_level='low'이 생성자에 전달됨"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, thinking_level="low")

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "low"

    def test_no_thinking_level_by_default(self):
        """기본 호출 시 thinking_level 파라미터 없음 (Gemini 기본 high 사용)"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings)

            call_kwargs = MockLLM.call_args[1]
            # thinking_level이 없어야 함 (Gemini 기본 high 적용)
            assert "thinking_level" not in call_kwargs

    def test_thinking_level_minimal(self):
        """thinking_level='minimal' 전달"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, thinking_level="minimal")

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "minimal"

    def test_thinking_level_high(self):
        """thinking_level='high' 명시적 전달"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, thinking_level="high")

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "high"


# ============================================================================
# B-1-5: reasoning_effort → thinking_level 하위 호환성 매핑
# ============================================================================


class TestGetChatModelBackwardCompat:
    """하위 호환성 kwargs 테스트"""

    def _make_mock_settings(self, api_key: str = "fake-key"):
        from naver_connect_chatbot.config.settings.gemini import GeminiLLMSettings

        mock_settings = MagicMock()
        mock_settings.gemini_llm = GeminiLLMSettings(google_api_key=api_key)
        return mock_settings

    def test_reasoning_effort_maps_to_thinking_level(self):
        """reasoning_effort kwarg가 thinking_level로 매핑됨"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, reasoning_effort="medium")

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "medium"

    def test_reasoning_effort_low_maps_to_thinking_level_low(self):
        """reasoning_effort='low' → thinking_level='low'"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, reasoning_effort="low")

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "low"

    def test_thinking_level_takes_priority_over_reasoning_effort(self):
        """thinking_level이 reasoning_effort보다 우선 적용됨"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            # thinking_level이 있으면 reasoning_effort는 무시됨
            get_chat_model(
                settings_obj=mock_settings,
                thinking_level="high",
                reasoning_effort="low",
            )

            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("thinking_level") == "high"

    def test_use_reasoning_true_does_not_set_thinking_level(self):
        """use_reasoning=True 단독 사용 시 thinking_level 미설정 (Gemini 기본값 사용)"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, use_reasoning=True)

            call_kwargs = MockLLM.call_args[1]
            # use_reasoning만으로는 thinking_level이 설정되지 않음
            assert "thinking_level" not in call_kwargs

    def test_reasoning_effort_not_passed_to_constructor(self):
        """reasoning_effort 자체는 생성자에 전달되지 않음 (excluded_keys)"""
        mock_settings = self._make_mock_settings()

        with patch("naver_connect_chatbot.config.llm.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = MagicMock()

            from naver_connect_chatbot.config.llm import get_chat_model

            get_chat_model(settings_obj=mock_settings, reasoning_effort="medium")

            call_kwargs = MockLLM.call_args[1]
            assert "reasoning_effort" not in call_kwargs
            assert "use_reasoning" not in call_kwargs
