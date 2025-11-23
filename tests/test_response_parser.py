"""
Response Parser 유틸리티 테스트.

200+ 라인의 중복 파싱 로직을 대체하는 공통 유틸리티에 대한
포괄적인 테스트를 제공합니다.
"""

import pytest
import json
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from naver_connect_chatbot.service.agents.response_parser import (
    parse_agent_response,
    extract_tool_messages,
    get_last_tool_message,
)


# 테스트용 Pydantic 모델 (이름이 Test로 시작하면 pytest가 수집하려 하므로 _ 접두사 사용)
class _TestResult(BaseModel):
    """테스트용 결과 모델"""
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    message: str = ""


class _TestData(BaseModel):
    """다른 타입의 테스트 모델"""
    value: int
    label: str


class TestParseAgentResponse:
    """parse_agent_response 함수 테스트 그룹"""

    def test_already_correct_type(self):
        """이미 올바른 타입인 경우"""
        original = _TestResult(status="success", confidence=0.9, message="Done")
        result = parse_agent_response(original, _TestResult)

        assert result is original  # 동일한 인스턴스 반환
        assert result.status == "success"

    def test_convert_basemodel_type(self):
        """다른 BaseModel에서 변환"""
        # _TestData를 _TestResult처럼 파싱 시도 (실패해야 함)
        test_data = _TestData(value=42, label="test")

        with pytest.raises(ValueError):
            parse_agent_response(test_data, _TestResult)

    def test_parse_from_dict(self):
        """일반 dict 파싱"""
        data = {
            "status": "completed",
            "confidence": 0.85,
            "message": "All good"
        }

        result = parse_agent_response(data, _TestResult)

        assert result.status == "completed"
        assert result.confidence == 0.85
        assert result.message == "All good"

    def test_parse_from_agent_state_messages(self):
        """AgentState의 messages에서 ToolMessage 추출"""
        tool_msg = ToolMessage(
            content=json.dumps({
                "status": "extracted",
                "confidence": 0.95,
                "message": "From ToolMessage"
            }),
            tool_call_id="test-123"
        )

        response = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Processing..."),
                tool_msg
            ]
        }

        result = parse_agent_response(response, _TestResult)

        assert result.status == "extracted"
        assert result.confidence == 0.95
        assert result.message == "From ToolMessage"

    def test_parse_from_tool_message_dict_content(self):
        """ToolMessage의 content가 dict인 경우 (실제로는 이미 model_dump()된 상태)"""
        # 실제 LangChain에서는 content가 str이지만,
        # 간혹 dict가 들어올 수 있는 경우를 테스트
        content_dict = {
            "status": "direct_dict",
            "confidence": 0.88
        }

        # ToolMessage는 보통 content를 자동으로 직렬화하므로
        # 우리는 이미 직렬화된 것으로 가정
        tool_msg = ToolMessage(
            content=json.dumps(content_dict),  # JSON 문자열로 변환
            tool_call_id="test-456"
        )

        response = {"messages": [tool_msg]}
        result = parse_agent_response(response, _TestResult)

        assert result.status == "direct_dict"
        assert result.confidence == 0.88

    def test_parse_from_tool_message_basemodel_content(self):
        """ToolMessage의 content가 BaseModel인 경우 (model_dump_json() 사용)"""
        content = _TestResult(status="model_content", confidence=0.77)

        # BaseModel은 model_dump_json()으로 직렬화
        tool_msg = ToolMessage(
            content=content.model_dump_json(),  # JSON 문자열로 변환
            tool_call_id="test-789"
        )

        response = {"messages": [tool_msg]}
        result = parse_agent_response(response, _TestResult)

        assert result.status == "model_content"
        assert result.confidence == 0.77

    def test_parse_from_legacy_agent_output(self):
        """Legacy Agent의 output 키에서 추출"""
        response = {
            "output": {
                "status": "legacy",
                "confidence": 0.65
            }
        }

        result = parse_agent_response(response, _TestResult)

        assert result.status == "legacy"
        assert result.confidence == 0.65

    def test_parse_from_message_content(self):
        """AIMessage의 content에서 추출"""
        ai_message = AIMessage(
            content=json.dumps({
                "status": "ai_response",
                "confidence": 0.92
            })
        )

        result = parse_agent_response(ai_message, _TestResult)

        assert result.status == "ai_response"
        assert result.confidence == 0.92

    def test_parse_from_json_string(self):
        """JSON 문자열 파싱"""
        json_str = json.dumps({
            "status": "json_parsed",
            "confidence": 0.99,
            "message": "Perfect"
        })

        result = parse_agent_response(json_str, _TestResult)

        assert result.status == "json_parsed"
        assert result.confidence == 0.99
        assert result.message == "Perfect"

    def test_parse_invalid_json_string(self):
        """잘못된 JSON 문자열"""
        invalid_json = "not a json {invalid syntax"

        with pytest.raises(ValueError):
            parse_agent_response(invalid_json, _TestResult)

    def test_parse_with_fallback_on_error(self):
        """파싱 실패 시 fallback 사용"""
        fallback = _TestResult(
            status="fallback",
            confidence=0.5,
            message="Default value"
        )

        invalid_data = "completely invalid"
        result = parse_agent_response(invalid_data, _TestResult, fallback=fallback)

        assert result.status == "fallback"
        assert result.confidence == 0.5
        assert result.message == "Default value"

    def test_parse_missing_required_field(self):
        """필수 필드 누락 시"""
        incomplete_data = {
            "status": "incomplete"
            # confidence 누락
        }

        with pytest.raises(ValueError):
            parse_agent_response(incomplete_data, _TestResult)

    def test_parse_missing_required_field_with_fallback(self):
        """필수 필드 누락 + fallback 사용"""
        incomplete_data = {"status": "incomplete"}
        fallback = _TestResult(status="default", confidence=0.0)

        result = parse_agent_response(incomplete_data, _TestResult, fallback=fallback)

        assert result.status == "default"
        assert result.confidence == 0.0

    def test_parse_multiple_tool_messages_uses_last(self):
        """여러 ToolMessage가 있을 때 마지막 것 사용"""
        first_msg = ToolMessage(
            content=json.dumps({"status": "first", "confidence": 0.1}),
            tool_call_id="1"
        )
        second_msg = ToolMessage(
            content=json.dumps({"status": "second", "confidence": 0.2}),
            tool_call_id="2"
        )
        last_msg = ToolMessage(
            content=json.dumps({"status": "last", "confidence": 0.9}),
            tool_call_id="3"
        )

        response = {
            "messages": [
                HumanMessage(content="Hi"),
                first_msg,
                AIMessage(content="Thinking..."),
                second_msg,
                last_msg
            ]
        }

        result = parse_agent_response(response, _TestResult)

        # 마지막 ToolMessage 사용
        assert result.status == "last"
        assert result.confidence == 0.9

    def test_parse_validation_error_with_invalid_value(self):
        """유효성 검증 실패 (confidence 범위 초과)"""
        invalid_data = {
            "status": "test",
            "confidence": 1.5  # 1.0 초과 (Field(le=1.0))
        }

        with pytest.raises(ValueError):
            parse_agent_response(invalid_data, _TestResult)


class TestExtractToolMessages:
    """extract_tool_messages 함수 테스트 그룹"""

    def test_extract_single_tool_message(self):
        """단일 ToolMessage 추출"""
        tool_msg = ToolMessage(content="test", tool_call_id="123")
        messages = [
            HumanMessage(content="Hi"),
            tool_msg
        ]

        result = extract_tool_messages(messages)

        assert len(result) == 1
        assert result[0] is tool_msg

    def test_extract_multiple_tool_messages(self):
        """여러 ToolMessage 추출"""
        tool1 = ToolMessage(content="first", tool_call_id="1")
        tool2 = ToolMessage(content="second", tool_call_id="2")
        tool3 = ToolMessage(content="third", tool_call_id="3")

        messages = [
            HumanMessage(content="Start"),
            tool1,
            AIMessage(content="Processing"),
            tool2,
            HumanMessage(content="Continue"),
            tool3
        ]

        result = extract_tool_messages(messages)

        assert len(result) == 3
        assert result[0] is tool1
        assert result[1] is tool2
        assert result[2] is tool3

    def test_extract_no_tool_messages(self):
        """ToolMessage가 없는 경우"""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="World")
        ]

        result = extract_tool_messages(messages)

        assert len(result) == 0

    def test_extract_from_empty_list(self):
        """빈 리스트"""
        result = extract_tool_messages([])
        assert len(result) == 0


class TestGetLastToolMessage:
    """get_last_tool_message 함수 테스트 그룹"""

    def test_get_last_from_multiple(self):
        """여러 ToolMessage 중 마지막 반환"""
        tool1 = ToolMessage(content="first", tool_call_id="1")
        tool2 = ToolMessage(content="second", tool_call_id="2")
        tool3 = ToolMessage(content="third", tool_call_id="3")

        messages = [tool1, AIMessage(content="..."), tool2, tool3]

        result = get_last_tool_message(messages)

        assert result is tool3

    def test_get_last_from_single(self):
        """단일 ToolMessage"""
        tool_msg = ToolMessage(content="only", tool_call_id="1")
        messages = [HumanMessage(content="Hi"), tool_msg]

        result = get_last_tool_message(messages)

        assert result is tool_msg

    def test_get_last_when_none(self):
        """ToolMessage가 없을 때 None 반환"""
        messages = [HumanMessage(content="Hi"), AIMessage(content="Hello")]

        result = get_last_tool_message(messages)

        assert result is None

    def test_get_last_from_empty(self):
        """빈 리스트에서 None 반환"""
        result = get_last_tool_message([])
        assert result is None
