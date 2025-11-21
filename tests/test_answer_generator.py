"""Tests for answer_generator agent."""
import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from naver_connect_chatbot.service.agents.answer_generator import (
    AnswerOutput,
    create_answer_generator,
    get_generation_strategy,
)


class TestAnswerOutput:
    """Test the AnswerOutput Pydantic model."""

    def test_valid_answer(self):
        """Valid answer should create AnswerOutput."""
        output = AnswerOutput(answer="PyTorch is a framework.")
        assert output.answer == "PyTorch is a framework."

    def test_empty_answer_allowed(self):
        """Empty string is valid (LLM might return empty)."""
        output = AnswerOutput(answer="")
        assert output.answer == ""

    def test_missing_answer_fails(self):
        """Missing answer field should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnswerOutput()  # No answer provided

    def test_model_dump(self):
        """Model should serialize correctly."""
        output = AnswerOutput(answer="Test answer")
        dumped = output.model_dump()
        assert dumped == {"answer": "Test answer"}

    def test_model_validate(self):
        """Model should validate from dict."""
        data = {"answer": "Validated answer"}
        output = AnswerOutput.model_validate(data)
        assert output.answer == "Validated answer"


class TestGenerationStrategy:
    """Test strategy mapping logic."""

    @pytest.mark.parametrize("intent,expected_strategy", [
        ("SIMPLE_QA", "simple"),
        ("COMPLEX_REASONING", "complex"),
        ("EXPLORATORY", "exploratory"),
        ("CLARIFICATION_NEEDED", "simple"),
        ("UNKNOWN_INTENT", "simple"),  # Fallback
        ("", "simple"),  # Empty string fallback
        (None, "simple"),  # None fallback
    ])
    def test_get_generation_strategy(self, intent, expected_strategy):
        """Test intent to strategy mapping."""
        assert get_generation_strategy(intent) == expected_strategy


class TestAnswerGeneratorAgent:
    """Test agent creation."""

    def test_create_answer_generator_simple(self):
        """Should create agent with simple strategy."""
        mock_llm = Mock()
        agent = create_answer_generator(mock_llm, strategy="simple")
        assert agent is not None

    def test_create_answer_generator_complex(self):
        """Should create agent with complex strategy."""
        mock_llm = Mock()
        agent = create_answer_generator(mock_llm, strategy="complex")
        assert agent is not None

    def test_create_answer_generator_exploratory(self):
        """Should create agent with exploratory strategy."""
        mock_llm = Mock()
        agent = create_answer_generator(mock_llm, strategy="exploratory")
        assert agent is not None

    def test_create_answer_generator_default_strategy(self):
        """Should default to simple strategy when not specified."""
        mock_llm = Mock()
        agent = create_answer_generator(mock_llm)
        assert agent is not None
