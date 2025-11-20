"""
LLM 설정 및 팩토리 함수 테스트 모듈

이 모듈은 OpenAI, OpenRouter, Naver Cloud 세 가지 LLM 제공자의
설정 클래스와 ChatOpenAI 인스턴스 생성 테스트를 포함합니다.

Integration Tests:
    실제 LLM API를 호출하는 통합 테스트는 @pytest.mark.integration으로 표시되어 있습니다.
    이러한 테스트는 기본적으로 실행되지 않으며, 다음 명령으로 실행할 수 있습니다:
    
    pytest tests/test_llm.py -m integration
"""

import os
from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from naver_connect_chatbot.config import settings as global_settings
from naver_connect_chatbot.config.llm import (
    LLMProvider,
    OpenAISettings,
    OpenRouterSettings,
    get_chat_model,
)
from naver_connect_chatbot.config.settings import Settings
from naver_connect_chatbot.config.settings.naver_cloud import (
    NaverCloudOpenAICompatibleSettings,
)


class TestOpenAISettings:
    """OpenAI 설정 클래스 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch):
        """기본값 테스트"""
        # 환경변수 초기화
        for key in list(os.environ.keys()):
            if key.startswith("OPENAI_"):
                monkeypatch.delenv(key, raising=False)

        # env_file을 비활성화하여 .env 파일 로드 방지
        settings = OpenAISettings(_env_file=None)

        assert settings.api_key is None
        assert settings.model_name == "gpt-4o-mini"
        assert settings.temperature == 0.7
        assert settings.max_tokens is None
        assert settings.enabled is False

    def test_env_var_loading(self):
        """환경변수 로드 테스트"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL_NAME": "gpt-4o",
            "OPENAI_TEMPERATURE": "0.9",
            "OPENAI_MAX_TOKENS": "2000",
            "OPENAI_ENABLED": "true",
        }):
            settings = OpenAISettings(_env_file=None)

            assert settings.api_key.get_secret_value() == "test-key-123"
            assert settings.model_name == "gpt-4o"
            assert settings.temperature == 0.9
            assert settings.max_tokens == 2000
            assert settings.enabled is True

    def test_temperature_validation(self):
        """온도 값 검증 테스트"""
        # 정상 범위
        settings = OpenAISettings(temperature=0.0)
        assert settings.temperature == 0.0
        
        settings = OpenAISettings(temperature=2.0)
        assert settings.temperature == 2.0
        
        # 범위 초과시 Pydantic이 ValidationError 발생
        with pytest.raises(Exception):  # ValidationError
            OpenAISettings(temperature=-0.1)
        
        with pytest.raises(Exception):  # ValidationError
            OpenAISettings(temperature=2.1)


class TestOpenRouterSettings:
    """OpenRouter 설정 클래스 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch):
        """기본값 테스트"""
        # 환경변수 초기화
        for key in list(os.environ.keys()):
            if key.startswith("OPENROUTER_"):
                monkeypatch.delenv(key, raising=False)

        # env_file을 비활성화하여 .env 파일 로드 방지
        settings = OpenRouterSettings(_env_file=None)

        assert settings.api_key is None
        assert settings.base_url == "https://openrouter.ai/api/v1"
        assert settings.model_name == "anthropic/claude-3.5-sonnet"
        assert settings.temperature == 0.7
        assert settings.max_tokens is None
        assert settings.enabled is False

    def test_env_var_loading(self):
        """환경변수 로드 테스트"""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-test-123",
            "OPENROUTER_BASE_URL": "https://custom.openrouter.ai/v1",
            "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
            "OPENROUTER_TEMPERATURE": "0.5",
            "OPENROUTER_MAX_TOKENS": "4000",
            "OPENROUTER_ENABLED": "true",
        }):
            settings = OpenRouterSettings(_env_file=None)

            assert settings.api_key.get_secret_value() == "sk-or-test-123"
            assert settings.base_url == "https://custom.openrouter.ai/v1"
            assert settings.model_name == "openai/gpt-4o"
            assert settings.temperature == 0.5
            assert settings.max_tokens == 4000
            assert settings.enabled is True


class TestGetChatModel:
    """get_chat_model 팩토리 함수 테스트"""

    def test_openai_chat_model_creation(self):
        """OpenAI ChatModel 생성 테스트"""
        # Mock settings
        settings = Settings(
            openai=OpenAISettings(
                api_key="test-openai-key",
                model_name="gpt-4o-mini",
                temperature=0.7,
                enabled=True
            )
        )
        
        model = get_chat_model(LLMProvider.OPENAI, settings_obj=settings)
        
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.7
        # API 키는 실제로는 마스킹되어 확인하기 어렵지만, 
        # 예외가 발생하지 않으면 성공

    def test_openrouter_chat_model_creation(self):
        """OpenRouter ChatModel 생성 테스트"""
        settings = Settings(
            openrouter=OpenRouterSettings(
                api_key="sk-or-test-key",
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.8,
                enabled=True
            )
        )
        
        model = get_chat_model(LLMProvider.OPENROUTER, settings_obj=settings)
        
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "anthropic/claude-3.5-sonnet"
        assert model.temperature == 0.8

    def test_naver_cloud_chat_model_creation(self):
        """Naver Cloud ChatModel 생성 테스트 (커스텀 헤더 포함)"""
        settings = Settings(
            openai_compatible=NaverCloudOpenAICompatibleSettings(
                base_url="https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-003",
                api_key="test-ncp-key",
                api_gateway_key="test-gateway-key",
                default_model="HCX-007",
                enabled=True
            )
        )
        
        model = get_chat_model(LLMProvider.NAVER_CLOUD, settings_obj=settings)
        
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "HCX-007"
        # default_headers는 내부적으로 설정되므로 직접 확인하기 어려움

    def test_openai_disabled_raises_error(self):
        """비활성화된 OpenAI 제공자 오류 테스트"""
        settings = Settings(
            openai=OpenAISettings(
                api_key="test-key",
                enabled=False  # 비활성화
            )
        )
        
        with pytest.raises(ValueError, match="OpenAI is not enabled"):
            get_chat_model(LLMProvider.OPENAI, settings_obj=settings)

    def test_missing_api_key_raises_error(self):
        """API 키 누락 오류 테스트"""
        settings = Settings(
            openai=OpenAISettings(
                api_key=None,  # 키 누락
                enabled=True
            )
        )
        
        with pytest.raises(ValueError, match="API key is missing"):
            get_chat_model(LLMProvider.OPENAI, settings_obj=settings)

    def test_unsupported_provider_raises_error(self):
        """지원하지 않는 제공자 오류 테스트"""
        settings = Settings()
        
        # 잘못된 provider 값 (실제로는 Enum이므로 발생하기 어렵지만)
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_chat_model("invalid_provider", settings_obj=settings)  # type: ignore

    def test_override_model_name(self):
        """모델명 오버라이드 테스트"""
        settings = Settings(
            openai=OpenAISettings(
                api_key="test-key",
                model_name="gpt-4o-mini",
                enabled=True
            )
        )
        
        model = get_chat_model(
            LLMProvider.OPENAI,
            settings_obj=settings,
            model="gpt-4o"  # 오버라이드
        )
        
        assert model.model_name == "gpt-4o"

    def test_override_temperature(self):
        """온도 오버라이드 테스트"""
        settings = Settings(
            openai=OpenAISettings(
                api_key="test-key",
                temperature=0.7,
                enabled=True
            )
        )
        
        model = get_chat_model(
            LLMProvider.OPENAI,
            settings_obj=settings,
            temperature=0.9  # 오버라이드
        )
        
        assert model.temperature == 0.9

    def test_override_max_tokens(self):
        """최대 토큰 수 오버라이드 테스트"""
        settings = Settings(
            openai=OpenAISettings(
                api_key="test-key",
                max_tokens=1000,
                enabled=True
            )
        )
        
        model = get_chat_model(
            LLMProvider.OPENAI,
            settings_obj=settings,
            max_tokens=2000  # 오버라이드
        )
        
        assert model.max_tokens == 2000

    def test_multiple_overrides(self):
        """여러 파라미터 동시 오버라이드 테스트"""
        settings = Settings(
            openrouter=OpenRouterSettings(
                model_name="anthropic/claude-sonnet-4.5",
                temperature=0.3,
                max_tokens=2000,
                enabled=True
            )
        )
        
        model = get_chat_model(
            LLMProvider.OPENROUTER,
            settings_obj=settings,
            model="openai/gpt-4.1-mini",
            temperature=0.3,
            max_tokens=2000
        )
        
        assert model.model_name == "openai/gpt-4.1-mini"
        assert model.temperature == 0.3
        assert model.max_tokens == 2000


class TestLLMProviderEnum:
    """LLMProvider Enum 테스트"""

    def test_enum_values(self):
        """Enum 값 테스트"""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.OPENROUTER == "openrouter"
        assert LLMProvider.NAVER_CLOUD == "naver_cloud"

    def test_enum_membership(self):
        """Enum 멤버십 테스트"""
        assert LLMProvider.OPENAI in LLMProvider
        assert LLMProvider.OPENROUTER in LLMProvider
        assert LLMProvider.NAVER_CLOUD in LLMProvider

    def test_enum_iteration(self):
        """Enum 이터레이션 테스트"""
        providers = list(LLMProvider)
        assert len(providers) == 3
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.OPENROUTER in providers
        assert LLMProvider.NAVER_CLOUD in providers


# ============================================================================
# 통합 테스트 (실제 API 호출)
# ============================================================================


@pytest.mark.integration
class TestLLMIntegration:
    """
    실제 LLM API 호출 통합 테스트
    
    이 테스트들은 실제 API를 호출하므로:
    - API 키가 필요합니다
    - 네트워크 연결이 필요합니다
    - 비용이 발생할 수 있습니다
    - 실행 시간이 더 오래 걸립니다
    
    실행 방법:
        pytest tests/test_llm.py -m integration -v
    """

    @pytest.mark.asyncio
    async def test_openai_actual_call(self):
        """OpenAI 실제 호출 테스트"""
        if not global_settings.openai.enabled or not global_settings.openai.api_key:
            pytest.skip("OpenAI가 설정되지 않았거나 비활성화되어 있습니다")
        
        model = get_chat_model(LLMProvider.OPENAI)
        
        # 간단한 테스트 프롬프트
        response = await model.ainvoke("Hello! Please respond with 'Hi' only.")
        
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        print(f"\n✅ OpenAI 응답: {response.content}")

    @pytest.mark.asyncio
    async def test_openrouter_actual_call(self):
        """OpenRouter 실제 호출 테스트"""
        if not global_settings.openrouter.enabled or not global_settings.openrouter.api_key:
            pytest.skip("OpenRouter가 설정되지 않았거나 비활성화되어 있습니다")
        
        model = get_chat_model(LLMProvider.OPENROUTER)
        
        # 간단한 테스트 프롬프트
        response = await model.ainvoke("Hello! Please respond with 'Hi' only.")
        
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        print(f"\n✅ OpenRouter 응답: {response.content}")

    @pytest.mark.asyncio
    async def test_naver_cloud_actual_call(self):
        """Naver Cloud 실제 호출 테스트"""
        if (not global_settings.openai_compatible.enabled or 
            not global_settings.openai_compatible.api_key or 
            not global_settings.openai_compatible.base_url):
            pytest.skip("Naver Cloud가 설정되지 않았거나 비활성화되어 있습니다")
        
        model = get_chat_model(LLMProvider.NAVER_CLOUD)
        
        # 간단한 테스트 프롬프트 (한국어)
        response = await model.ainvoke("안녕하세요! '안녕'이라고만 답해주세요.")
        
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        print(f"\n✅ Naver Cloud 응답: {response.content}")

    def test_openai_sync_call(self):
        """OpenAI 동기 호출 테스트"""
        if not global_settings.openai.enabled or not global_settings.openai.api_key:
            pytest.skip("OpenAI가 설정되지 않았거나 비활성화되어 있습니다")
        
        model = get_chat_model(LLMProvider.OPENAI)
        
        # 간단한 테스트 프롬프트
        response = model.invoke("What is 2+2? Answer with just the number.")
        
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        assert "4" in response.content
        print(f"\n✅ OpenAI 동기 응답: {response.content}")

    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """스트리밍 응답 테스트"""
        if not global_settings.openai.enabled or not global_settings.openai.api_key:
            pytest.skip("OpenAI가 설정되지 않았거나 비활성화되어 있습니다")
        
        model = get_chat_model(LLMProvider.OPENAI)
        
        # 스트리밍 응답 수집
        chunks = []
        async for chunk in model.astream("Count from 1 to 3. Answer: "):
            chunks.append(chunk.content)
        
        # 모든 청크를 합쳐서 검증
        full_response = "".join(chunks)
        assert len(chunks) > 0
        assert len(full_response) > 0
        print(f"\n✅ 스트리밍 청크 수: {len(chunks)}, 전체 응답: {full_response}")

    @pytest.mark.asyncio
    async def test_temperature_effect(self):
        """Temperature 파라미터 효과 테스트"""
        if not global_settings.openai.enabled or not global_settings.openai.api_key:
            pytest.skip("OpenAI가 설정되지 않았거나 비활성화되어 있습니다")
        
        prompt = "Write a creative one-sentence story about a cat."
        
        # 낮은 temperature (더 결정적)
        model_low = get_chat_model(LLMProvider.OPENAI, temperature=0.1)
        response_low = await model_low.ainvoke(prompt)
        
        # 높은 temperature (더 창의적)
        model_high = get_chat_model(LLMProvider.OPENAI, temperature=0.9)
        response_high = await model_high.ainvoke(prompt)
        
        # 둘 다 응답이 있어야 함
        assert response_low.content
        assert response_high.content
        
        print(f"\n✅ Low temp (0.1): {response_low.content}")
        print(f"✅ High temp (0.9): {response_high.content}")

    @pytest.mark.asyncio
    async def test_all_providers_comparison(self):
        """모든 제공자 비교 테스트"""
        prompt = "What is AI? Answer in one short sentence."
        results = {}
        
        # OpenAI
        if global_settings.openai.enabled and global_settings.openai.api_key:
            try:
                model = get_chat_model(LLMProvider.OPENAI)
                response = await model.ainvoke(prompt)
                results["OpenAI"] = response.content
            except Exception as e:
                results["OpenAI"] = f"Error: {e}"
        
        # OpenRouter
        if global_settings.openrouter.enabled and global_settings.openrouter.api_key:
            try:
                model = get_chat_model(LLMProvider.OPENROUTER)
                response = await model.ainvoke(prompt)
                results["OpenRouter"] = response.content
            except Exception as e:
                results["OpenRouter"] = f"Error: {e}"
        
        # Naver Cloud
        if (global_settings.openai_compatible.enabled and 
            global_settings.openai_compatible.api_key and
            global_settings.openai_compatible.base_url):
            try:
                model = get_chat_model(LLMProvider.NAVER_CLOUD)
                response = await model.ainvoke(prompt)
                results["Naver Cloud"] = response.content
            except Exception as e:
                results["Naver Cloud"] = f"Error: {e}"
        
        if not results:
            pytest.skip("활성화된 LLM 제공자가 없습니다")
        
        print("\n✅ 모든 제공자 응답 비교:")
        for provider, response in results.items():
            print(f"\n{provider}:")
            print(f"  {response}")
        
        # 최소 하나는 성공해야 함
        assert any("Error" not in resp for resp in results.values())
