"""
B-3: 노드 유틸리티 함수 단위 테스트.

_build_document_label(), _format_chat_history(), _extract_text_response(),
_extract_text_from_content()을 검증합니다.
API 키나 VectorDB 불필요.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from naver_connect_chatbot.service.graph.nodes import (
    _build_document_label,
    _format_chat_history,
    _extract_text_response,
    _extract_text_from_content,
)


# ============================================================================
# B-3-1~3: _build_document_label() 테스트
# ============================================================================


class TestBuildDocumentLabel:
    """_build_document_label() 메타데이터 기반 식별자 생성 테스트"""

    def test_pdf_with_full_metadata(self):
        """PDF + course + lecture_num → '[강의자료: CV 이론/3강]'"""
        doc = Document(
            page_content="내용",
            metadata={"doc_type": "pdf", "course": "CV 이론", "lecture_num": "3"},
        )
        result = _build_document_label(doc, 0)
        assert result == "[강의자료: CV 이론/3강]"

    def test_slack_qa_with_full_metadata(self):
        """Slack QA + course + generation → '[Slack Q&A: PyTorch/1기]'"""
        doc = Document(
            page_content="질답 내용",
            metadata={"doc_type": "slack_qa", "course": "PyTorch", "generation": "1기"},
        )
        result = _build_document_label(doc, 0)
        # generation은 필터용이고, label에는 course만 포함
        assert result == "[Slack Q&A: PyTorch]"

    def test_slack_qa_course_only(self):
        """Slack Q&A + course만 → '[Slack Q&A: PyTorch]'"""
        doc = Document(
            page_content="질답",
            metadata={"doc_type": "slack_qa", "course": "PyTorch"},
        )
        result = _build_document_label(doc, 0)
        assert result == "[Slack Q&A: PyTorch]"

    def test_notebook_with_topic(self):
        """notebook + topic → '[실습노트북: NLP 기초/Tokenization]'"""
        doc = Document(
            page_content="코드",
            metadata={"doc_type": "notebook", "course": "NLP 기초", "topic": "Tokenization"},
        )
        result = _build_document_label(doc, 0)
        assert result == "[실습노트북: NLP 기초/Tokenization]"

    def test_no_metadata_fallback(self):
        """메타데이터 없음 → '[문서 N]' 폴백 (1-based index)"""
        doc = Document(page_content="내용", metadata={})
        result = _build_document_label(doc, 0)
        assert result == "[문서 1]"

    def test_no_metadata_index_2(self):
        """index=1 → '[문서 2]'"""
        doc = Document(page_content="내용", metadata={})
        result = _build_document_label(doc, 1)
        assert result == "[문서 2]"

    def test_no_metadata_empty_dict(self):
        """metadata={} → '[문서 1]' 폴백"""
        doc = Document(page_content="내용", metadata={})
        result = _build_document_label(doc, 0)
        assert result == "[문서 1]"

    def test_doc_type_only_no_course(self):
        """doc_type만 있고 course 없음 → '[강의자료]'"""
        doc = Document(page_content="내용", metadata={"doc_type": "pdf"})
        result = _build_document_label(doc, 0)
        assert result == "[강의자료]"

    def test_lecture_transcript_type(self):
        """lecture_transcript doc_type → '[강의녹취록: ...]'"""
        doc = Document(
            page_content="녹취",
            metadata={"doc_type": "lecture_transcript", "course": "CV", "lecture_num": "5"},
        )
        result = _build_document_label(doc, 0)
        assert result == "[강의녹취록: CV/5강]"

    def test_lecture_num_preferred_over_topic(self):
        """lecture_num과 topic 둘 다 있으면 lecture_num 우선"""
        doc = Document(
            page_content="내용",
            metadata={
                "doc_type": "pdf",
                "course": "ML",
                "lecture_num": "2",
                "topic": "Linear Regression",
            },
        )
        result = _build_document_label(doc, 0)
        assert result == "[강의자료: ML/2강]"


# ============================================================================
# B-3-4~6: _format_chat_history() 테스트
# ============================================================================


class TestFormatChatHistory:
    """_format_chat_history() 대화 히스토리 포맷팅 테스트"""

    def test_empty_messages_returns_empty_string(self):
        """빈 메시지 리스트 → 빈 문자열"""
        result = _format_chat_history([])
        assert result == ""

    def test_single_turn_includes_turn_number(self):
        """단일 턴 → 턴 번호 포함"""
        messages = [
            HumanMessage(content="PyTorch란?"),
            AIMessage(content="PyTorch는 딥러닝 프레임워크입니다."),
        ]
        result = _format_chat_history(messages)
        assert "[턴 1]" in result
        assert "PyTorch란?" in result
        assert "PyTorch는 딥러닝 프레임워크입니다." in result

    def test_multiple_turns_numbered_correctly(self):
        """다중 턴 → 순서대로 번호 부여"""
        messages = [
            HumanMessage(content="질문1"),
            AIMessage(content="답변1"),
            HumanMessage(content="질문2"),
            AIMessage(content="답변2"),
        ]
        result = _format_chat_history(messages)
        assert "[턴 1]" in result
        assert "[턴 2]" in result

    def test_ai_response_truncated_over_500_chars(self):
        """AI 응답 500자 초과 시 '...'으로 잘림"""
        long_response = "A" * 600
        messages = [
            HumanMessage(content="질문"),
            AIMessage(content=long_response),
        ]
        result = _format_chat_history(messages)
        assert "..." in result
        # 원본 600자 전체가 포함되면 안 됨
        assert "A" * 600 not in result

    def test_ai_response_exactly_500_chars_not_truncated(self):
        """AI 응답 정확히 500자는 잘리지 않음"""
        exact_response = "B" * 500
        messages = [
            HumanMessage(content="질문"),
            AIMessage(content=exact_response),
        ]
        result = _format_chat_history(messages)
        # 500자는 truncation 없음
        assert "..." not in result

    def test_max_turns_limit(self):
        """max_turns=1 → 가장 최근 1턴만 포함"""
        messages = [
            HumanMessage(content="오래된 질문"),
            AIMessage(content="오래된 답변"),
            HumanMessage(content="최근 질문"),
            AIMessage(content="최근 답변"),
        ]
        result = _format_chat_history(messages, max_turns=1)
        assert "최근 질문" in result
        assert "최근 답변" in result
        # 오래된 턴은 제외됨
        assert "오래된 질문" not in result

    def test_includes_note_about_no_repetition(self):
        """결과에 중복 방지 주의 문구 포함"""
        messages = [
            HumanMessage(content="질문"),
            AIMessage(content="답변"),
        ]
        result = _format_chat_history(messages)
        assert "이미 답변한 내용을 반복하지 마세요" in result

    def test_history_starts_with_header(self):
        """히스토리 문자열이 '[이전 대화]' 헤더로 시작"""
        messages = [
            HumanMessage(content="질문"),
            AIMessage(content="답변"),
        ]
        result = _format_chat_history(messages)
        assert result.startswith("[이전 대화]")


# ============================================================================
# B-3-7~9: _extract_text_response() 및 _extract_text_from_content() 테스트
# ============================================================================


class TestExtractTextFromContent:
    """_extract_text_from_content() 내부 유틸리티 테스트"""

    def test_string_content_returned_as_is(self):
        """문자열 content → 그대로 반환"""
        result = _extract_text_from_content("Simple answer")
        assert result == "Simple answer"

    def test_list_with_thinking_and_text_blocks(self):
        """Gemini thinking blocks: type='text' 블록만 추출"""
        content = [
            {"type": "thinking", "text": "Let me think about this..."},
            {"type": "text", "text": "The answer is 42."},
        ]
        result = _extract_text_from_content(content)
        assert result == "The answer is 42."
        assert "Let me think" not in result

    def test_list_with_multiple_text_blocks(self):
        """여러 text 블록 → 줄바꿈으로 결합"""
        content = [
            {"type": "thinking", "text": "thinking..."},
            {"type": "text", "text": "Part 1."},
            {"type": "text", "text": "Part 2."},
        ]
        result = _extract_text_from_content(content)
        assert "Part 1." in result
        assert "Part 2." in result

    def test_list_with_only_thinking_blocks_falls_back(self):
        """text 블록 없이 thinking 블록만 있으면 폴백 (첫 번째 블록의 text 사용)"""
        content = [
            {"type": "thinking", "text": "Only thinking here."},
        ]
        result = _extract_text_from_content(content)
        assert result == "Only thinking here."

    def test_list_with_string_items(self):
        """리스트 안에 문자열 아이템 → 직접 사용"""
        content = ["item1", "item2"]
        result = _extract_text_from_content(content)
        assert "item1" in result
        assert "item2" in result

    def test_empty_list_returns_str_of_empty_list(self):
        """빈 리스트 → str([]) 반환"""
        result = _extract_text_from_content([])
        # 빈 리스트는 text_parts가 없어서 str([]) 반환
        assert result == str([])

    def test_block_without_type_field(self):
        """type 필드 없는 dict 블록 → text 키로 추출"""
        content = [{"text": "No type field here."}]
        result = _extract_text_from_content(content)
        assert result == "No type field here."


class TestExtractTextResponse:
    """_extract_text_response() 통합 추출 테스트"""

    def test_ai_message_string_content(self):
        """AIMessage string content → 그대로 반환"""
        msg = AIMessage(content="Simple answer")
        result = _extract_text_response(msg)
        assert result == "Simple answer"

    def test_ai_message_with_thinking_blocks(self):
        """AIMessage list content (Gemini thinking blocks) → text만 추출"""
        msg = AIMessage(
            content=[
                {"type": "thinking", "text": "Let me think about this..."},
                {"type": "text", "text": "The answer is 42."},
            ]
        )
        result = _extract_text_response(msg)
        assert result == "The answer is 42."
        assert "Let me think" not in result

    def test_ai_message_list_no_text_type_falls_back(self):
        """AIMessage list에 text 타입 없으면 폴백 처리"""
        msg = AIMessage(
            content=[
                {"type": "thinking", "text": "Thinking only."},
            ]
        )
        result = _extract_text_response(msg)
        # thinking 블록의 text 폴백
        assert result == "Thinking only."

    def test_dict_with_output_key(self):
        """dict 응답 + 'output' 키 → output 값 반환"""
        response = {"output": "The output answer."}
        result = _extract_text_response(response)
        assert result == "The output answer."

    def test_dict_with_content_key(self):
        """dict 응답 + 'content' 키 → content 값 반환"""
        response = {"content": "The content answer."}
        result = _extract_text_response(response)
        assert result == "The content answer."

    def test_plain_string_returned_as_is(self):
        """순수 문자열 → 그대로 반환"""
        result = _extract_text_response("Direct string response")
        assert result == "Direct string response"

    def test_output_key_takes_priority_over_content(self):
        """'output'과 'content' 둘 다 있으면 'output' 우선"""
        response = {"output": "output value", "content": "content value"}
        result = _extract_text_response(response)
        assert result == "output value"

    def test_object_with_content_attribute(self):
        """content 속성을 가진 일반 객체 → content 추출"""

        class FakeMessage:
            content = "Attribute content"

        result = _extract_text_response(FakeMessage())
        assert result == "Attribute content"

    def test_non_string_output_in_dict_falls_through(self):
        """dict의 output이 문자열이 아니면 content 시도"""
        response = {"output": 42, "content": "content fallback"}
        result = _extract_text_response(response)
        assert result == "content fallback"
