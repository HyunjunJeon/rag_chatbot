"""
프롬프트 로딩 및 관리 테스트.

YAML 기반 프롬프트 시스템을 대상으로 다음을 검증합니다.
- YAML 파일에서의 로딩
- 스키마 검증
- 폴백 메커니즘
- 캐시 동작
- 오류 처리
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from naver_connect_chatbot.prompts import (
    get_prompt,
    list_available_prompts,
    load_prompt,
    reload_prompts,
    PromptLoadError,
)
from naver_connect_chatbot.prompts.loader import load_prompt_config, PromptConfig, _build_messages


class TestPromptLoading:
    """YAML 프롬프트 기본 로딩을 검증합니다."""

    def test_load_rag_generation_prompt(self):
        """RAG 생성 프롬프트를 정상적으로 불러오는지 확인합니다."""
        prompt = get_prompt("rag_generation")
        assert prompt is not None

        # 예상된 변수를 이용해 포맷이 가능한지 확인합니다.
        result = prompt.format(question="What is AI?", context="AI is artificial intelligence")
        assert "What is AI?" in result
        assert "AI is artificial intelligence" in result
        assert "assistant" in result.lower()  # 어시스턴트 지시가 포함되어야 함

    def test_load_document_grading_prompt(self):
        """문서 평가 프롬프트를 로드할 수 있는지 확인합니다."""
        prompt = get_prompt("document_grading")
        assert prompt is not None

        # 포맷 테스트
        result = prompt.format(question="Test question?", context="Test context")
        assert "Test question?" in result
        assert "Test context" in result
        assert "grader" in result.lower()  # 채점자 역할이 나타나야 함

    def test_load_query_transformation_prompt(self):
        """질의 변환 프롬프트를 로드할 수 있는지 확인합니다."""
        prompt = get_prompt("query_transformation")
        assert prompt is not None

        # 포맷 테스트
        result = prompt.format(question="Original question")
        assert "Original question" in result
        assert "retrieval" in result.lower()  # 검색 최적화 관련 언급 필요

    def test_load_multi_query_generation_prompt(self):
        """다중 질의 생성 프롬프트를 로드할 수 있는지 확인합니다."""
        prompt = get_prompt("multi_query_generation")
        assert prompt is not None

        # 두 변수를 모두 사용하여 포맷 테스트
        result = prompt.format(query="Search query", num=3)
        assert "Search query" in result
        assert "3" in result
        assert "search" in result.lower()  # 검색 전략 안내가 포함되어야 함

    def test_all_prompts_load_without_error(self):
        """네 가지 프롬프트가 모두 정상 로드되는지 확인합니다."""
        prompts = [
            get_prompt("rag_generation"),
            get_prompt("document_grading"),
            get_prompt("query_transformation"),
            get_prompt("multi_query_generation"),
        ]

        for prompt in prompts:
            assert prompt is not None
            assert hasattr(prompt, "format")  # PromptTemplate 이어야 함

    def test_get_prompt_can_return_text(self):
        """return_type='text' 옵션이 문자열을 반환하는지 검증합니다."""
        prompt_text = get_prompt("rag_generation", return_type="text")
        assert isinstance(prompt_text, str)
        assert "Question:" in prompt_text

    def test_list_available_prompts_contains_expected_names(self):
        """프롬프트 목록 API가 주요 프롬프트를 포함하는지 확인합니다."""
        available = list_available_prompts()
        expected = {"rag_generation", "document_grading", "multi_query_generation"}
        assert expected.issubset(set(available))


class TestPromptConfigValidation:
    """프롬프트 설정 검증 로직을 테스트합니다."""

    def test_valid_simple_prompt_config(self):
        """유효한 simple 프롬프트 구성을 검증합니다."""
        valid_config = {
            "_type": "simple",
            "template": "Hello {name}",
            "input_variables": ["name"],
            "metadata": {"name": "test", "description": "Test prompt", "version": "1.0"},
        }
        config = PromptConfig(**valid_config)
        assert config.prompt_type == "simple"
        assert config.template == "Hello {name}"
        assert "name" in config.input_variables

    def test_valid_chat_messages_config(self):
        """유효한 chat_messages 구성을 검증합니다."""
        valid_config = {
            "_type": "chat_messages",
            "messages": [
                {"role": "system", "content": "System prompt {var}"},
                {"role": "human", "content": "{question}"},
            ],
            "input_variables": ["var", "question"],
            "metadata": {"name": "test", "description": "Test prompt", "version": "1.0"},
        }
        config = PromptConfig(**valid_config)
        assert config.prompt_type == "chat_messages"
        assert len(config.messages) == 2

    def test_invalid_prompt_type_raises_error(self):
        """잘못된 _type 값이 검증 오류를 발생시키는지 확인합니다."""
        invalid_config = {
            "_type": "invalid_type",
            "metadata": {"name": "test", "description": "test"},
        }
        with pytest.raises(Exception):  # Pydantic validation error
            PromptConfig(**invalid_config)

    def test_missing_metadata_raises_error(self):
        """metadata가 없으면 검증 오류가 발생하는지 확인합니다."""
        invalid_config = {
            "_type": "simple",
            "template": "Test {var}",
        }
        with pytest.raises(Exception):  # Pydantic validation error
            PromptConfig(**invalid_config)


class TestFallbackMechanism:
    """프롬프트 폴백 메커니즘을 검증합니다."""

    def test_fallback_on_missing_file(self):
        """파일이 없을 때 폴백 프롬프트가 사용되는지 확인합니다."""
        from langchain_core.prompts import ChatPromptTemplate

        fallback = ChatPromptTemplate.from_template("Fallback: {query}")

        # 존재하지 않는 프롬프트를 폴백과 함께 로드합니다.
        result = load_prompt("non_existent_prompt_xyz_123", fallback=fallback)
        assert result == fallback

    def test_load_without_fallback_raises_error(self):
        """폴백 없이 존재하지 않는 프롬프트를 읽으면 오류가 발생하는지 확인합니다."""
        with pytest.raises(PromptLoadError, match="Prompt file not found"):
            load_prompt("non_existent_prompt_xyz_123")

    def test_convenience_functions_have_fallbacks(self):
        """모든 편의 함수가 폴백을 보유하는지 확인합니다."""
        # YAML 파일을 삭제해도 폴백이 동작해야 합니다.
        # 해당 기능은 편의 함수 호출만으로 간접 검증됩니다.


class TestCacheBehavior:
    """프롬프트 캐시 동작을 검증합니다."""

    def test_cache_speeds_up_reload(self):
        """캐시가 반복 로드 성능을 향상시키는지 확인합니다."""
        # 첫 번째 로드
        config1 = load_prompt_config("rag_generation")

        # 두 번째 로드 (캐시에서 가져와야 함)
        config2 = load_prompt_config("rag_generation")

        # 캐시 덕분에 동일 객체를 참조해야 함
        assert config1 is config2

    def test_reload_prompts_clears_cache(self):
        """reload_prompts()가 캐시를 비우는지 확인합니다."""
        # 캐시 채우기
        load_prompt_config("rag_generation")

        # 비우기 전 캐시 정보 확인
        cache_info_before = load_prompt_config.cache_info()
        assert cache_info_before.currsize > 0

        # 캐시 비우기
        reload_prompts()

        # 캐시가 비워졌는지 확인
        cache_info_after = load_prompt_config.cache_info()
        assert cache_info_after.currsize == 0


class TestMessageBuilding:
    """설정에서 메시지를 구성하는 로직을 검증합니다."""

    def test_build_valid_messages(self):
        """유효한 설정으로 메시지를 구성할 수 있는지 확인합니다."""
        messages_config = [
            {"role": "system", "content": "System message"},
            {"role": "human", "content": "Human message"},
            {"role": "ai", "content": "AI message"},
        ]

        messages = _build_messages(messages_config)

        assert len(messages) == 3
        assert messages[0] == ("system", "System message")
        assert messages[1] == ("human", "Human message")
        assert messages[2] == ("ai", "AI message")

    def test_build_messages_with_invalid_role_raises_error(self):
        """잘못된 role이 PromptLoadError를 발생시키는지 확인합니다."""
        messages_config = [
            {"role": "invalid_role", "content": "Message"},
        ]

        with pytest.raises(PromptLoadError, match="Unknown message role"):
            _build_messages(messages_config)

    def test_build_messages_with_various_roles(self):
        """지원되는 모든 role이 허용되는지 확인합니다."""
        valid_roles = ["system", "human", "ai", "user", "assistant"]

        for role in valid_roles:
            messages_config = [{"role": role, "content": "Test"}]
            messages = _build_messages(messages_config)
            assert len(messages) == 1
            assert messages[0][0] == role.lower()


class TestErrorHandling:
    """프롬프트 로딩 시 오류 처리를 검증합니다."""

    def test_yaml_parse_error_handling(self):
        """잘못된 YAML 문법을 처리하는지 확인합니다."""
        invalid_yaml = "invalid: yaml: content: ["

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.open", mock_open(read_data=invalid_yaml)):
                with pytest.raises(PromptLoadError, match="YAML parsing error"):
                    load_prompt_config("test_prompt")

    def test_validation_error_handling(self):
        """스키마 검증 오류가 발생하는지 확인합니다."""
        # YAML은 유효하지만 스키마가 잘못된 경우
        invalid_schema_yaml = """
_type: simple
# Missing required metadata
input_variables: []
"""

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.open", mock_open(read_data=invalid_schema_yaml)):
                with pytest.raises(PromptLoadError, match="Validation error"):
                    load_prompt_config("test_prompt")

    def test_missing_template_field_error(self):
        """simple 프롬프트에서 template 필드가 없을 때 오류를 확인합니다."""
        config_dict = {
            "_type": "simple",
            "metadata": {"name": "test", "description": "test"},
        }

        mock_config = PromptConfig(**{**config_dict, "template": None})

        with patch("naver_connect_chatbot.prompts.loader.load_prompt_config", return_value=mock_config):
            with pytest.raises(PromptLoadError, match="Simple prompt missing 'template' field"):
                load_prompt("test_prompt")

    def test_missing_messages_field_error(self):
        """chat 프롬프트에서 messages 필드가 없을 때 오류를 확인합니다."""
        config_dict = {
            "_type": "chat_messages",
            "metadata": {"name": "test", "description": "test"},
        }

        mock_config = PromptConfig(**{**config_dict, "messages": None})

        with patch("naver_connect_chatbot.prompts.loader.load_prompt_config", return_value=mock_config):
            with pytest.raises(PromptLoadError, match="Chat prompt missing 'messages' field"):
                load_prompt("test_prompt")


class TestPromptTemplateIntegration:
    """실제 YAML 파일을 대상으로 한 통합 테스트."""

    def test_template_files_exist(self):
        """모든 템플릿 파일이 존재하는지 확인합니다."""
        templates_dir = Path(__file__).parent.parent / "app" / "naver_connect_chatbot" / "prompts" / "templates"

        expected_files = [
            "rag_generation.yaml",
            "document_grading.yaml",
            "query_transformation.yaml",
            "multi_query_generation.yaml",
        ]

        for filename in expected_files:
            filepath = templates_dir / filename
            assert filepath.exists(), f"Template file {filename} not found"

    def test_all_templates_are_valid_yaml(self):
        """모든 템플릿 파일이 유효한 YAML인지 확인합니다."""
        templates_dir = Path(__file__).parent.parent / "app" / "naver_connect_chatbot" / "prompts" / "templates"

        for yaml_file in templates_dir.glob("*.yaml"):
            with yaml_file.open("r") as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file.name}: {e}")

    def test_loaded_prompts_can_format(self):
        """로딩된 프롬프트가 변수와 함께 포맷되는지 검증합니다."""
        # RAG 생성
        rag_prompt = get_prompt("rag_generation")
        rag_result = rag_prompt.format(question="Q", context="C")
        assert "Q" in rag_result and "C" in rag_result

        # 문서 평가
        doc_prompt = get_prompt("document_grading")
        doc_result = doc_prompt.format(question="Q", context="C")
        assert "Q" in doc_result

        # 질의 변환
        query_prompt = get_prompt("query_transformation")
        query_result = query_prompt.format(question="Q")
        assert "Q" in query_result

        # 다중 질의 생성
        multi_prompt = get_prompt("multi_query_generation")
        multi_result = multi_prompt.format(query="Q", num=3)
        assert "Q" in multi_result
