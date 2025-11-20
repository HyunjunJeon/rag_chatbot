"""
Settings 모듈 테스트

Pydantic Settings의 로딩, 기본값, 환경변수 우선순위 등을 검증합니다.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from naver_connect_chatbot.config import (
    AdvancedHybridSettings,
    HybridMethodType,
    LoggingSettings,
    MultiQuerySettings,
    NaverCloudChatSettings,
    NaverCloudEmbeddingsSettings,
    NaverCloudOpenAICompatibleSettings,
    NaverCloudRAGReasoningSettings,
    NaverCloudRerankerSettings,
    NaverCloudSegmentationSettings,
    NaverCloudSummarizationSettings,
    OpenAISettings,
    OpenRouterSettings,
    QdrantVectorStoreSettings,
    RetrieverSettings,
    RetrieverStrategy,
    Settings,
    SlackSettings,
    settings,
)


class TestEnumTypes:
    """Enum 타입 정의 테스트"""

    def test_retriever_strategy_values(self) -> None:
        """RetrieverStrategy Enum 값 검증"""
        assert RetrieverStrategy.SPARSE_ONLY.value == "sparse_only"
        assert RetrieverStrategy.DENSE_ONLY.value == "dense_only"
        assert RetrieverStrategy.HYBRID.value == "hybrid"
        assert RetrieverStrategy.MULTI_QUERY.value == "multi_query"
        assert RetrieverStrategy.ADVANCED.value == "advanced"

    def test_hybrid_method_type_values(self) -> None:
        """HybridMethodType Enum 값 검증"""
        assert HybridMethodType.RRF.value == "rrf"
        assert HybridMethodType.CC.value == "cc"


class TestRetrieverSettings:
    """RetrieverSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        # 환경변수 초기화
        for key in list(os.environ.keys()):
            if key.startswith("RETRIEVER_"):
                monkeypatch.delenv(key, raising=False)

        retriever_settings = RetrieverSettings()

        assert retriever_settings.default_k == 10
        assert retriever_settings.default_sparse_weight == 0.5
        assert retriever_settings.default_dense_weight == 0.5
        assert retriever_settings.default_rrf_c == 60

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("RETRIEVER_DEFAULT_K", "20")
        monkeypatch.setenv("RETRIEVER_DEFAULT_SPARSE_WEIGHT", "0.7")
        monkeypatch.setenv("RETRIEVER_DEFAULT_DENSE_WEIGHT", "0.3")
        monkeypatch.setenv("RETRIEVER_DEFAULT_RRF_C", "100")

        retriever_settings = RetrieverSettings()

        assert retriever_settings.default_k == 20
        assert retriever_settings.default_sparse_weight == 0.7
        assert retriever_settings.default_dense_weight == 0.3
        assert retriever_settings.default_rrf_c == 100

    def test_validation_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """필드 제약조건 검증"""
        # default_k는 1~100 범위
        monkeypatch.setenv("RETRIEVER_DEFAULT_K", "0")
        with pytest.raises(ValidationError):
            RetrieverSettings()

        monkeypatch.setenv("RETRIEVER_DEFAULT_K", "101")
        with pytest.raises(ValidationError):
            RetrieverSettings()

        # weights는 0.0~1.0 범위
        monkeypatch.setenv("RETRIEVER_DEFAULT_K", "10")
        monkeypatch.setenv("RETRIEVER_DEFAULT_SPARSE_WEIGHT", "1.5")
        with pytest.raises(ValidationError):
            RetrieverSettings()


class TestMultiQuerySettings:
    """MultiQuerySettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("MULTI_QUERY_"):
                monkeypatch.delenv(key, raising=False)

        multi_query_settings = MultiQuerySettings(_env_file=None)

        assert multi_query_settings.num_queries == 4
        assert multi_query_settings.default_strategy == "rrf"
        assert multi_query_settings.rrf_k == 60
        assert multi_query_settings.include_original is True

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("MULTI_QUERY_NUM_QUERIES", "6")
        monkeypatch.setenv("MULTI_QUERY_DEFAULT_STRATEGY", "max")
        monkeypatch.setenv("MULTI_QUERY_RRF_K", "80")
        monkeypatch.setenv("MULTI_QUERY_INCLUDE_ORIGINAL", "false")

        multi_query_settings = MultiQuerySettings(_env_file=None)

        assert multi_query_settings.num_queries == 6
        assert multi_query_settings.default_strategy == "max"
        assert multi_query_settings.rrf_k == 80
        assert multi_query_settings.include_original is False

    def test_strategy_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """병합 전략 값 검증"""
        monkeypatch.setenv("MULTI_QUERY_DEFAULT_STRATEGY", "invalid_strategy")
        with pytest.raises(ValidationError):
            MultiQuerySettings()

    def test_num_queries_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """쿼리 개수 범위 검증 (1~10)"""
        monkeypatch.setenv("MULTI_QUERY_NUM_QUERIES", "0")
        with pytest.raises(ValidationError):
            MultiQuerySettings()

        monkeypatch.setenv("MULTI_QUERY_NUM_QUERIES", "11")
        with pytest.raises(ValidationError):
            MultiQuerySettings()


class TestAdvancedHybridSettings:
    """AdvancedHybridSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("ADVANCED_HYBRID_"):
                monkeypatch.delenv(key, raising=False)

        advanced_settings = AdvancedHybridSettings(_env_file=None)

        assert advanced_settings.base_hybrid_weight == 0.4
        assert advanced_settings.multi_query_weight == 0.6

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("ADVANCED_HYBRID_BASE_HYBRID_WEIGHT", "0.3")
        monkeypatch.setenv("ADVANCED_HYBRID_MULTI_QUERY_WEIGHT", "0.7")

        advanced_settings = AdvancedHybridSettings(_env_file=None)

        assert advanced_settings.base_hybrid_weight == 0.3
        assert advanced_settings.multi_query_weight == 0.7

    def test_weight_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """가중치 범위 검증 (0.0~1.0)"""
        monkeypatch.setenv("ADVANCED_HYBRID_BASE_HYBRID_WEIGHT", "-0.1")
        with pytest.raises(ValidationError):
            AdvancedHybridSettings()

        monkeypatch.setenv("ADVANCED_HYBRID_BASE_HYBRID_WEIGHT", "1.1")
        with pytest.raises(ValidationError):
            AdvancedHybridSettings()


class TestQdrantVectorStoreSettings:
    """QdrantVectorStoreSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("QDRANT_"):
                monkeypatch.delenv(key, raising=False)

        qdrant_settings = QdrantVectorStoreSettings(_env_file=None)

        assert qdrant_settings.url is None
        assert qdrant_settings.api_key is None
        assert qdrant_settings.collection_name == "default"
        assert qdrant_settings.embedding_dimensions == 1024

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_key")
        monkeypatch.setenv("QDRANT_COLLECTION_NAME", "custom_collection")
        monkeypatch.setenv("QDRANT_EMBEDDING_DIMENSIONS", "768")

        qdrant_settings = QdrantVectorStoreSettings(_env_file=None)

        assert qdrant_settings.url == "http://qdrant:6333"
        assert qdrant_settings.api_key.get_secret_value() == "test_key"
        assert qdrant_settings.collection_name == "custom_collection"
        assert qdrant_settings.embedding_dimensions == 768


class TestNaverCloudEmbeddingsSettings:
    """NaverCloudEmbeddingsSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_EMBEDDINGS_"):
                monkeypatch.delenv(key, raising=False)

        embeddings_settings = NaverCloudEmbeddingsSettings(_env_file=None)

        assert embeddings_settings.model_url is None
        assert embeddings_settings.api_key is None

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_EMBEDDINGS_MODEL_URL", "https://example.com")
        monkeypatch.setenv("NAVER_CLOUD_EMBEDDINGS_API_KEY", "test_api_key")

        embeddings_settings = NaverCloudEmbeddingsSettings(_env_file=None)

        assert embeddings_settings.model_url == "https://example.com"
        assert embeddings_settings.api_key.get_secret_value() == "test_api_key"


class TestNaverCloudRerankerSettings:
    """NaverCloudRerankerSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_RERANKER_"):
                monkeypatch.delenv(key, raising=False)

        reranker_settings = NaverCloudRerankerSettings(_env_file=None)

        assert reranker_settings.endpoint is None
        assert reranker_settings.api_key is None
        assert reranker_settings.api_gateway_key is None
        assert reranker_settings.request_timeout == 30.0
        assert reranker_settings.default_top_k == 10
        assert reranker_settings.enabled is False
        assert reranker_settings.id_key is None

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_ENDPOINT", "https://api.example.com/rerank")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT", "60.0")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_DEFAULT_TOP_K", "20")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_ENABLED", "true")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_ID_KEY", "doc_id")

        reranker_settings = NaverCloudRerankerSettings(_env_file=None)

        assert reranker_settings.endpoint == "https://api.example.com/rerank"
        assert reranker_settings.api_key.get_secret_value() == "test_api_key"
        assert reranker_settings.api_gateway_key.get_secret_value() == "test_gateway_key"
        assert reranker_settings.request_timeout == 60.0
        assert reranker_settings.default_top_k == 20
        assert reranker_settings.enabled is True
        assert reranker_settings.id_key == "doc_id"

    def test_validation_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """필드 제약조건 검증"""
        # request_timeout은 1.0~300.0 범위
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT", "0.5")
        with pytest.raises(ValidationError):
            NaverCloudRerankerSettings()

        monkeypatch.setenv("NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT", "301.0")
        with pytest.raises(ValidationError):
            NaverCloudRerankerSettings()

        # default_top_k는 1~100 범위
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_REQUEST_TIMEOUT", "30.0")
        monkeypatch.setenv("NAVER_CLOUD_RERANKER_DEFAULT_TOP_K", "0")
        with pytest.raises(ValidationError):
            NaverCloudRerankerSettings()

        monkeypatch.setenv("NAVER_CLOUD_RERANKER_DEFAULT_TOP_K", "101")
        with pytest.raises(ValidationError):
            NaverCloudRerankerSettings()


class TestGlobalSettings:
    """전역 Settings 클래스 테스트"""

    def test_settings_structure(self) -> None:
        """Settings 계층 구조 검증"""
        test_settings = Settings()

        # 각 서브 설정이 올바르게 구성되었는지 확인
        assert isinstance(test_settings.naver_cloud_embeddings, NaverCloudEmbeddingsSettings)
        assert isinstance(test_settings.qdrant_vector_store, QdrantVectorStoreSettings)
        assert isinstance(test_settings.retriever, RetrieverSettings)
        assert isinstance(test_settings.multi_query, MultiQuerySettings)
        assert isinstance(test_settings.advanced_hybrid, AdvancedHybridSettings)
        assert isinstance(test_settings.reranker, NaverCloudRerankerSettings)

    def test_settings_singleton(self) -> None:
        """전역 settings 인스턴스가 올바르게 생성되었는지 확인"""
        assert settings is not None
        assert isinstance(settings, Settings)


class TestSettingsAccess:
    """Settings 접근 방식 테스트"""

    def test_settings_direct_access(self) -> None:
        """settings 객체를 통한 직접 접근 검증"""
        from naver_connect_chatbot.config import settings

        # Retriever 설정 접근
        assert isinstance(settings.retriever.default_k, int)
        assert isinstance(settings.retriever.default_sparse_weight, float)
        assert isinstance(settings.retriever.default_dense_weight, float)
        assert isinstance(settings.retriever.default_rrf_c, int)

        # MultiQuery 설정 접근
        assert isinstance(settings.multi_query.num_queries, int)
        assert isinstance(settings.multi_query.default_strategy, str)
        assert isinstance(settings.multi_query.rrf_k, int)
        assert isinstance(settings.multi_query.include_original, bool)

        # Advanced Hybrid 설정 접근
        assert isinstance(settings.advanced_hybrid.base_hybrid_weight, float)
        assert isinstance(settings.advanced_hybrid.multi_query_weight, float)

    def test_enum_types_available(self) -> None:
        """Enum 타입이 config 모듈에서 접근 가능한지 검증"""
        from naver_connect_chatbot.config import (
            HybridMethodType,
            RetrieverStrategy,
        )

        assert RetrieverStrategy.HYBRID.value == "hybrid"
        assert HybridMethodType.RRF.value == "rrf"


class TestDynamicSettingsModification:
    """런타임에 설정을 동적으로 변경하는 시나리오 테스트"""

    def test_modify_retriever_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """런타임에 retriever 설정을 변경할 수 있는지 검증"""
        from naver_connect_chatbot.config import settings

        # 원본 값 저장
        original_k = settings.retriever.default_k

        # 값 변경
        settings.retriever.default_k = 20

        assert settings.retriever.default_k == 20

        # 원복
        settings.retriever.default_k = original_k

    def test_settings_persistence_across_imports(self) -> None:
        """여러 모듈에서 settings를 import해도 동일 인스턴스인지 검증"""
        from naver_connect_chatbot.config import settings as settings1
        from naver_connect_chatbot.config.settings import settings as settings2

        # 동일한 인스턴스 확인
        assert settings1 is settings2


class TestEnvFileLoading:
    """
    .env 파일 로딩 테스트
    
    참고: 실제 .env 파일 존재 여부에 따라 동작이 달라질 수 있음
    """

    def test_env_file_path_resolution(self) -> None:
        """프로젝트 루트 경로가 올바르게 계산되는지 확인"""
        from naver_connect_chatbot.config.settings import PROJECT_ROOT

        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.name == "naver_connect_chatbot"

    def test_settings_creation_without_env_file(self) -> None:
        """
        .env 파일이 없어도 기본값으로 설정이 생성되는지 검증

        참고: 이 테스트는 전역 settings 인스턴스를 사용하므로
        .env 파일에서 로드된 값들이 있을 수 있습니다.
        """
        # 기본값으로 설정 생성 가능해야 함
        test_settings = Settings()

        # 설정이 올바르게 구성되었는지 확인 (값은 .env에서 로드될 수 있음)
        assert test_settings.retriever.default_k > 0
        assert test_settings.multi_query.num_queries > 0
        assert 0.0 <= test_settings.advanced_hybrid.base_hybrid_weight <= 1.0

        # 서브 설정도 올바르게 생성됨
        assert test_settings.naver_cloud_embeddings.model_url is None or isinstance(test_settings.naver_cloud_embeddings.model_url, str)
        assert isinstance(test_settings.qdrant_vector_store.collection_name, str)


class TestOpenAISettings:
    """OpenAISettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("OPENAI_"):
                monkeypatch.delenv(key, raising=False)

        openai_settings = OpenAISettings(_env_file=None)  # .env 파일 로드 방지

        assert openai_settings.api_key is None
        assert openai_settings.model_name == "gpt-4o-mini"
        assert openai_settings.temperature == 0.7
        assert openai_settings.max_tokens is None
        assert openai_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "1.0")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "2048")
        monkeypatch.setenv("OPENAI_ENABLED", "true")

        openai_settings = OpenAISettings()

        assert openai_settings.api_key is not None
        assert openai_settings.model_name == "gpt-4o"
        assert openai_settings.temperature == 1.0
        assert openai_settings.max_tokens == 2048
        assert openai_settings.enabled is True

    def test_temperature_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """temperature 범위 검증 (0.0~2.0)"""
        monkeypatch.setenv("OPENAI_TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            OpenAISettings()

        monkeypatch.setenv("OPENAI_TEMPERATURE", "2.1")
        with pytest.raises(ValidationError):
            OpenAISettings()

    def test_max_tokens_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_tokens는 -1 이상이어야 함"""
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "-2")
        with pytest.raises(ValidationError):
            OpenAISettings()


class TestOpenRouterSettings:
    """OpenRouterSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("OPENROUTER_"):
                monkeypatch.delenv(key, raising=False)

        openrouter_settings = OpenRouterSettings(_env_file=None)

        assert openrouter_settings.api_key is None
        assert openrouter_settings.base_url == "https://openrouter.ai/api/v1"
        assert openrouter_settings.model_name == "anthropic/claude-3.5-sonnet"
        assert openrouter_settings.temperature == 0.7
        assert openrouter_settings.max_tokens is None
        assert openrouter_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.openrouter.ai/api/v1")
        monkeypatch.setenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o")
        monkeypatch.setenv("OPENROUTER_TEMPERATURE", "1.2")
        monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "4096")
        monkeypatch.setenv("OPENROUTER_ENABLED", "true")

        openrouter_settings = OpenRouterSettings(_env_file=None)

        assert openrouter_settings.api_key is not None
        assert openrouter_settings.base_url == "https://custom.openrouter.ai/api/v1"
        assert openrouter_settings.model_name == "openai/gpt-4o"
        assert openrouter_settings.temperature == 1.2
        assert openrouter_settings.max_tokens == 4096
        assert openrouter_settings.enabled is True

    def test_temperature_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """temperature 범위 검증 (0.0~2.0)"""
        monkeypatch.setenv("OPENROUTER_TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            OpenRouterSettings()

        monkeypatch.setenv("OPENROUTER_TEMPERATURE", "2.1")
        with pytest.raises(ValidationError):
            OpenRouterSettings()


class TestNaverCloudChatSettings:
    """NaverCloudChatSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_CHAT_"):
                monkeypatch.delenv(key, raising=False)

        chat_settings = NaverCloudChatSettings(_env_file=None)

        assert chat_settings.endpoint is None
        assert chat_settings.api_key is None
        assert chat_settings.api_gateway_key is None
        assert chat_settings.model_name == "HCX-003"
        assert chat_settings.temperature == 0.7
        assert chat_settings.max_tokens == 1024
        assert chat_settings.top_k == 0
        assert chat_settings.top_p == 0.8
        assert chat_settings.repeat_penalty == 1.0
        assert chat_settings.request_timeout == 60.0
        assert chat_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_CHAT_ENDPOINT", "https://api.example.com/chat")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_MODEL_NAME", "HCX-DASH-001")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TEMPERATURE", "0.9")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_MAX_TOKENS", "2048")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TOP_K", "10")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TOP_P", "0.95")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_REPEAT_PENALTY", "1.2")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_REQUEST_TIMEOUT", "120.0")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_ENABLED", "true")

        chat_settings = NaverCloudChatSettings(_env_file=None)

        assert chat_settings.endpoint == "https://api.example.com/chat"
        assert chat_settings.api_key is not None
        assert chat_settings.api_gateway_key is not None
        assert chat_settings.model_name == "HCX-DASH-001"
        assert chat_settings.temperature == 0.9
        assert chat_settings.max_tokens == 2048
        assert chat_settings.top_k == 10
        assert chat_settings.top_p == 0.95
        assert chat_settings.repeat_penalty == 1.2
        assert chat_settings.request_timeout == 120.0
        assert chat_settings.enabled is True

    def test_validation_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """필드 제약조건 검증"""
        # temperature는 0.0~1.0 범위
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()

        monkeypatch.setenv("NAVER_CLOUD_CHAT_TEMPERATURE", "1.1")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()

        # max_tokens는 -1~8192 범위
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TEMPERATURE", "0.7")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_MAX_TOKENS", "-2")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()

        monkeypatch.setenv("NAVER_CLOUD_CHAT_MAX_TOKENS", "8193")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()

        # top_p는 0.0~1.0 범위
        monkeypatch.setenv("NAVER_CLOUD_CHAT_MAX_TOKENS", "1024")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TOP_P", "1.1")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()

        # repeat_penalty는 0.0~10.0 범위
        monkeypatch.setenv("NAVER_CLOUD_CHAT_TOP_P", "0.8")
        monkeypatch.setenv("NAVER_CLOUD_CHAT_REPEAT_PENALTY", "10.1")
        with pytest.raises(ValidationError):
            NaverCloudChatSettings()


class TestNaverCloudSegmentationSettings:
    """NaverCloudSegmentationSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_SEGMENTATION_"):
                monkeypatch.delenv(key, raising=False)

        segmentation_settings = NaverCloudSegmentationSettings(_env_file=None)

        assert segmentation_settings.endpoint is None
        assert segmentation_settings.api_key is None
        assert segmentation_settings.api_gateway_key is None
        assert segmentation_settings.default_alpha == -1
        assert segmentation_settings.request_timeout == 30.0
        assert segmentation_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_ENDPOINT", "https://api.example.com/segment")
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_DEFAULT_ALPHA", "50")
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_REQUEST_TIMEOUT", "60.0")
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_ENABLED", "true")

        segmentation_settings = NaverCloudSegmentationSettings(_env_file=None)

        assert segmentation_settings.endpoint == "https://api.example.com/segment"
        assert segmentation_settings.api_key is not None
        assert segmentation_settings.api_gateway_key is not None
        assert segmentation_settings.default_alpha == 50
        assert segmentation_settings.request_timeout == 60.0
        assert segmentation_settings.enabled is True

    def test_alpha_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_alpha 범위 검증 (-1~100)"""
        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_DEFAULT_ALPHA", "-2")
        with pytest.raises(ValidationError):
            NaverCloudSegmentationSettings()

        monkeypatch.setenv("NAVER_CLOUD_SEGMENTATION_DEFAULT_ALPHA", "101")
        with pytest.raises(ValidationError):
            NaverCloudSegmentationSettings()


class TestNaverCloudSummarizationSettings:
    """NaverCloudSummarizationSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_SUMMARIZATION_"):
                monkeypatch.delenv(key, raising=False)

        summarization_settings = NaverCloudSummarizationSettings(_env_file=None)

        assert summarization_settings.endpoint is None
        assert summarization_settings.api_key is None
        assert summarization_settings.api_gateway_key is None
        assert summarization_settings.default_length == "medium"
        assert summarization_settings.request_timeout == 30.0
        assert summarization_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_ENDPOINT", "https://api.example.com/summarize")
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_DEFAULT_LENGTH", "long")
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_REQUEST_TIMEOUT", "45.0")
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_ENABLED", "true")

        summarization_settings = NaverCloudSummarizationSettings(_env_file=None)

        assert summarization_settings.endpoint == "https://api.example.com/summarize"
        assert summarization_settings.api_key is not None
        assert summarization_settings.api_gateway_key is not None
        assert summarization_settings.default_length == "long"
        assert summarization_settings.request_timeout == 45.0
        assert summarization_settings.enabled is True

    def test_length_literal_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_length Literal 타입 검증"""
        monkeypatch.setenv("NAVER_CLOUD_SUMMARIZATION_DEFAULT_LENGTH", "invalid")
        with pytest.raises(ValidationError):
            NaverCloudSummarizationSettings()


class TestNaverCloudRAGReasoningSettings:
    """NaverCloudRAGReasoningSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_RAG_REASONING_"):
                monkeypatch.delenv(key, raising=False)

        rag_reasoning_settings = NaverCloudRAGReasoningSettings(_env_file=None)

        assert rag_reasoning_settings.endpoint is None
        assert rag_reasoning_settings.api_key is None
        assert rag_reasoning_settings.api_gateway_key is None
        assert rag_reasoning_settings.request_timeout == 60.0
        assert rag_reasoning_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_RAG_REASONING_ENDPOINT", "https://api.example.com/rag")
        monkeypatch.setenv("NAVER_CLOUD_RAG_REASONING_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_RAG_REASONING_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_RAG_REASONING_REQUEST_TIMEOUT", "90.0")
        monkeypatch.setenv("NAVER_CLOUD_RAG_REASONING_ENABLED", "true")

        rag_reasoning_settings = NaverCloudRAGReasoningSettings(_env_file=None)

        assert rag_reasoning_settings.endpoint == "https://api.example.com/rag"
        assert rag_reasoning_settings.api_key is not None
        assert rag_reasoning_settings.api_gateway_key is not None
        assert rag_reasoning_settings.request_timeout == 90.0
        assert rag_reasoning_settings.enabled is True


class TestNaverCloudOpenAICompatibleSettings:
    """NaverCloudOpenAICompatibleSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("NAVER_CLOUD_OPENAI_COMPATIBLE_"):
                monkeypatch.delenv(key, raising=False)

        openai_compatible_settings = NaverCloudOpenAICompatibleSettings(_env_file=None)

        assert openai_compatible_settings.base_url is None
        assert openai_compatible_settings.api_key is None
        assert openai_compatible_settings.api_gateway_key is None
        assert openai_compatible_settings.default_model == "HCX-007"
        assert openai_compatible_settings.enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("NAVER_CLOUD_OPENAI_COMPATIBLE_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("NAVER_CLOUD_OPENAI_COMPATIBLE_API_KEY", "test_api_key")
        monkeypatch.setenv("NAVER_CLOUD_OPENAI_COMPATIBLE_API_GATEWAY_KEY", "test_gateway_key")
        monkeypatch.setenv("NAVER_CLOUD_OPENAI_COMPATIBLE_DEFAULT_MODEL", "HCX-003")
        monkeypatch.setenv("NAVER_CLOUD_OPENAI_COMPATIBLE_ENABLED", "true")

        openai_compatible_settings = NaverCloudOpenAICompatibleSettings(_env_file=None)

        assert openai_compatible_settings.base_url == "https://api.example.com/v1"
        assert openai_compatible_settings.api_key is not None
        assert openai_compatible_settings.api_gateway_key is not None
        assert openai_compatible_settings.default_model == "HCX-003"
        assert openai_compatible_settings.enabled is True


class TestLoggingSettings:
    """LoggingSettings 기본값 및 검증 테스트"""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """기본값 검증"""
        for key in list(os.environ.keys()):
            if key.startswith("LOG_"):
                monkeypatch.delenv(key, raising=False)

        logging_settings = LoggingSettings(_env_file=None)

        assert logging_settings.level == "INFO"
        assert logging_settings.enable_console is True
        assert logging_settings.enable_file is True
        assert logging_settings.log_dir == "logs"
        assert logging_settings.rotation == "100 MB"
        assert logging_settings.retention == "30 days"
        assert logging_settings.compression == "zip"
        assert logging_settings.json_format is True
        assert logging_settings.serialize is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_ENABLE_CONSOLE", "false")
        monkeypatch.setenv("LOG_ENABLE_FILE", "false")
        monkeypatch.setenv("LOG_LOG_DIR", "custom_logs")
        monkeypatch.setenv("LOG_ROTATION", "50 MB")
        monkeypatch.setenv("LOG_RETENTION", "7 days")
        monkeypatch.setenv("LOG_COMPRESSION", "gz")
        monkeypatch.setenv("LOG_JSON_FORMAT", "false")
        monkeypatch.setenv("LOG_SERIALIZE", "true")

        logging_settings = LoggingSettings(_env_file=None)

        assert logging_settings.level == "DEBUG"
        assert logging_settings.enable_console is False
        assert logging_settings.enable_file is False
        assert logging_settings.log_dir == "custom_logs"
        assert logging_settings.rotation == "50 MB"
        assert logging_settings.retention == "7 days"
        assert logging_settings.compression == "gz"
        assert logging_settings.json_format is False
        assert logging_settings.serialize is True


class TestSlackSettings:
    """SlackSettings 기본값 및 검증 테스트"""

    def test_required_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """필수 필드 검증 (bot_token, signing_secret)"""
        # bot_token과 signing_secret이 없으면 ValidationError 발생
        for key in list(os.environ.keys()):
            if key.startswith("SLACK_"):
                monkeypatch.delenv(key, raising=False)

        with pytest.raises(ValidationError) as exc_info:
            SlackSettings()

        # 필수 필드가 누락되었는지 확인
        error_fields = {err["loc"][0] for err in exc_info.value.errors()}
        assert "bot_token" in error_fields or "signing_secret" in error_fields

    def test_default_values_with_required_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """필수 필드 제공 시 기본값 검증"""
        # 모든 SLACK_ 환경변수 초기화
        for key in list(os.environ.keys()):
            if key.startswith("SLACK_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test123")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test_secret")

        slack_settings = SlackSettings(_env_file=None)

        assert slack_settings.bot_token is not None
        assert slack_settings.signing_secret is not None
        assert slack_settings.app_token is None
        assert slack_settings.port == 3000

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수를 통한 값 오버라이드 검증"""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-custom123")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "custom_secret")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-custom456")
        monkeypatch.setenv("SLACK_PORT", "8080")

        slack_settings = SlackSettings()

        assert slack_settings.bot_token is not None
        assert slack_settings.signing_secret is not None
        assert slack_settings.app_token is not None
        assert slack_settings.port == 8080

    def test_port_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """port 범위 검증 (1~65535)"""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test123")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test_secret")
        monkeypatch.setenv("SLACK_PORT", "0")

        with pytest.raises(ValidationError):
            SlackSettings()

        monkeypatch.setenv("SLACK_PORT", "65536")
        with pytest.raises(ValidationError):
            SlackSettings()

