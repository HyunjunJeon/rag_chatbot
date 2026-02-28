"""
B-4: OOD 패턴 매칭 단위 테스트.

classify_intent_node()의 패턴 매칭 경로 및
generate_ood_response_node()의 응답 생성을 검증합니다.
LLM 호출 없이 패턴 매칭만 테스트합니다.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from naver_connect_chatbot.service.graph.nodes import (
    classify_intent_node,
    generate_ood_response_node,
)


def _make_no_call_llm():
    """LLM이 호출되면 AssertionError를 발생시키는 mock LLM."""
    mock_llm = MagicMock()
    # with_structured_output이 호출되면 에러
    mock_llm.with_structured_output = MagicMock(
        side_effect=AssertionError("LLM should not be called for pattern-matched OOD")
    )
    # ainvoke도 에러
    mock_llm.ainvoke = MagicMock(
        side_effect=AssertionError("LLM should not be called for pattern-matched OOD")
    )
    return mock_llm


# ============================================================================
# B-4-1: greeting 패턴
# ============================================================================


class TestGreetingPatterns:
    """인사 패턴 매칭 테스트"""

    def test_annyeong_is_ood(self):
        """'안녕하세요' → OUT_OF_DOMAIN (LLM 호출 없음)"""
        mock_llm = _make_no_call_llm()
        state = {"question": "안녕하세요"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"
        assert result["domain_relevance"] == 0.0

    def test_hello_english_is_ood(self):
        """'hello there' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "hello there, how are you?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"
        assert result["domain_relevance"] == 0.0

    def test_good_morning_is_ood(self):
        """'좋은 아침이에요' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "좋은 아침이에요!"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_hi_with_space_is_ood(self):
        """'hi ' (공백 포함) → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "hi how are you"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_pattern_matched_has_high_confidence(self):
        """패턴 매칭 시 confidence=0.95"""
        mock_llm = _make_no_call_llm()
        state = {"question": "안녕"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent_confidence"] == 0.95


# ============================================================================
# B-4-2: self_intro 패턴
# ============================================================================


class TestSelfIntroPatterns:
    """챗봇 자기소개 패턴 테스트"""

    def test_neon_nugoo_is_ood(self):
        """'넌 누구야' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "넌 누구야?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_what_can_you_do_korean_is_ood(self):
        """'뭘 할 수 있어?' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "뭘 할 수 있어?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_who_are_you_english_is_ood(self):
        """'who are you' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "who are you?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_introduce_yourself_is_ood(self):
        """'자기소개 해줘' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "자기소개 해줘"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"


# ============================================================================
# B-4-3: chitchat 패턴
# ============================================================================


class TestChitchatPatterns:
    """잡담 패턴 테스트"""

    def test_simsim_is_ood(self):
        """'심심해' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "심심해"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_baegopa_is_ood(self):
        """'배고파' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "배고파 뭐 먹지?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_pigeون_is_ood(self):
        """'피곤해' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "피곤해서 공부가 안돼"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"


# ============================================================================
# B-4-4: off_topic 정밀화 테스트
# ============================================================================


class TestOffTopicPatterns:
    """오프토픽 패턴 정밀화 테스트"""

    def test_lunch_menu_is_ood(self):
        """'점심 메뉴' → OUT_OF_DOMAIN (LLM 호출 없음)"""
        mock_llm = _make_no_call_llm()
        state = {"question": "점심 메뉴 추천해줘"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_weather_is_ood(self):
        """'날씨' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "오늘 날씨 어때?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_stock_is_ood(self):
        """'주식' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "삼성 주식 어떻게 생각해?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"

    def test_soccer_game_is_ood(self):
        """'축구 경기' → OUT_OF_DOMAIN"""
        mock_llm = _make_no_call_llm()
        state = {"question": "어제 축구 경기 봤어?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "OUT_OF_DOMAIN"


# ============================================================================
# B-4-5: 기술 질문은 패턴 매칭 OOD가 아님
# ============================================================================


class TestTechnicalQuestionsNotPatternOOD:
    """기술 질문은 패턴 매칭으로 OOD가 되지 않음 → LLM 호출"""

    def _make_mock_llm_with_result(self):
        """LLM이 호출되면 SIMPLE_QA를 반환하는 mock."""
        from naver_connect_chatbot.service.agents.intent_classifier import IntentClassification

        mock_result = IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.9,
            reasoning="Technical question about Python",
            domain_relevance=0.9,
        )
        mock_structured = MagicMock()
        mock_structured.ainvoke = MagicMock(return_value=mock_result)

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
        return mock_llm

    def test_python_decorator_not_pattern_ood(self):
        """'Python 데코레이터' → 패턴 매칭 OOD가 아님 (LLM 호출됨)"""
        mock_llm = self._make_mock_llm_with_result()
        state = {"question": "Python 데코레이터가 뭔가요?"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        # LLM이 호출되어 SIMPLE_QA 반환
        assert result["intent"] == "SIMPLE_QA"

    def test_pytorch_question_not_pattern_ood(self):
        """'PyTorch DataLoader' → 패턴 매칭 OOD 아님"""
        mock_llm = self._make_mock_llm_with_result()
        state = {"question": "PyTorch DataLoader 사용법 알려줘"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "SIMPLE_QA"

    def test_transformer_question_not_pattern_ood(self):
        """'Transformer Attention' → 패턴 매칭 OOD 아님"""
        mock_llm = self._make_mock_llm_with_result()
        state = {"question": "Transformer의 Attention 메커니즘 설명해줘"}
        result = asyncio.run(classify_intent_node(state, llm=mock_llm))
        assert result["intent"] == "SIMPLE_QA"


# ============================================================================
# B-4-6: generate_ood_response_node - self_intro
# ============================================================================


class TestGenerateOODResponseNodeSelfIntro:
    """generate_ood_response_node 자기소개 응답 테스트"""

    def test_self_intro_response_contains_bot_description(self):
        """자기소개 요청 → 봇 설명 포함 응답"""
        state = {"question": "넌 누구야?", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        assert result["intent"] == "OUT_OF_DOMAIN" if "intent" in result else True
        assert "is_out_of_domain" in result
        assert result["is_out_of_domain"] is True
        answer = result["answer"]
        # 자기소개 응답에 봇 설명 포함
        assert "부스트캠프" in answer or "AI Tech" in answer or "도우미" in answer

    def test_self_intro_response_mentions_capabilities(self):
        """자기소개 응답에 기능 소개 포함"""
        state = {"question": "자기소개 해줘", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        answer = result["answer"]
        # 도움 가능 영역 언급
        assert "PyTorch" in answer or "AI" in answer or "ML" in answer

    def test_self_intro_strategy_is_ood_decline(self):
        """generation_strategy=ood_decline"""
        state = {"question": "뭘 할 수 있어?", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        assert result["generation_strategy"] == "ood_decline"


# ============================================================================
# B-4-7: generate_ood_response_node - greeting
# ============================================================================


class TestGenerateOODResponseNodeGreeting:
    """generate_ood_response_node 인사 응답 테스트"""

    def test_greeting_response_is_friendly(self):
        """인사 → 친근한 응답"""
        state = {"question": "안녕하세요!", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        answer = result["answer"]
        assert "안녕" in answer

    def test_greeting_response_contains_guidance(self):
        """인사 응답에 도움 가능 영역 안내 포함"""
        state = {"question": "안녕", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        answer = result["answer"]
        assert "질문" in answer or "도움" in answer or "AI" in answer

    def test_ood_response_includes_ai_message(self):
        """응답에 AIMessage가 messages 필드에 포함됨"""
        from langchain_core.messages import AIMessage

        state = {"question": "안녕하세요", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_non_greeting_non_selfintro_ood_response(self):
        """인사/자기소개 아닌 OOD → 거절 응답"""
        state = {"question": "오늘 날씨 어때?", "domain_relevance": 0.0}
        result = asyncio.run(generate_ood_response_node(state))
        answer = result["answer"]
        assert "죄송" in answer or "답변드리기 어렵" in answer
