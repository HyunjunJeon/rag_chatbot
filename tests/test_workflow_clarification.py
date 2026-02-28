"""
Workflow clarification 단위 테스트.

should_clarify 라우팅 함수와 filter_confidence 처리를 테스트합니다:
- Threshold boundary (off-by-one 버그 방지)
- Enable/disable clarification
- Default values
"""


import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from naver_connect_chatbot.service.graph.workflow import should_clarify


# ============================================================================
# should_clarify Function Tests
# ============================================================================


class TestShouldClarify:
    """should_clarify 라우팅 함수 테스트"""

    def test_disabled_returns_continue(self):
        """비활성화 시 항상 continue"""
        state = {"filter_confidence": 0.0}  # 매우 낮은 confidence
        result = should_clarify(state, enable_clarification=False)
        assert result == "continue"

    def test_high_confidence_returns_continue(self):
        """높은 confidence는 continue"""
        state = {"filter_confidence": 0.9}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "continue"

    def test_low_confidence_returns_clarify(self):
        """낮은 confidence는 clarify"""
        state = {"filter_confidence": 0.3}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "clarify"


class TestThresholdBoundary:
    """Threshold 경계값 테스트 (off-by-one 버그 방지)"""

    def test_exactly_at_threshold_triggers_clarify(self):
        """confidence == threshold일 때 clarify 트리거 (경계값 포함)"""
        state = {"filter_confidence": 0.5}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        # 중요: <= 비교이므로 정확히 같을 때도 clarify
        assert result == "clarify"

    def test_just_above_threshold_continues(self):
        """confidence > threshold일 때 continue"""
        state = {"filter_confidence": 0.51}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "continue"

    def test_just_below_threshold_clarifies(self):
        """confidence < threshold일 때 clarify"""
        state = {"filter_confidence": 0.49}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "clarify"

    def test_zero_confidence(self):
        """confidence = 0.0일 때 clarify"""
        state = {"filter_confidence": 0.0}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "clarify"

    def test_one_confidence(self):
        """confidence = 1.0일 때 continue"""
        state = {"filter_confidence": 1.0}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        assert result == "continue"


class TestDefaultValues:
    """기본값 처리 테스트"""

    def test_missing_filter_confidence_defaults_to_one(self):
        """filter_confidence 없으면 1.0으로 간주"""
        state = {}  # filter_confidence 없음
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.5
        )
        # 기본값 1.0 > 0.5 이므로 continue
        assert result == "continue"

    def test_default_threshold(self):
        """기본 threshold 값 사용"""
        state = {"filter_confidence": 0.4}
        # clarification_threshold 기본값 = 0.5
        result = should_clarify(state, enable_clarification=True)
        assert result == "clarify"


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_custom_threshold_zero(self):
        """threshold = 0.0일 때"""
        state = {"filter_confidence": 0.0}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=0.0
        )
        # confidence(0.0) <= threshold(0.0) 이므로 clarify
        assert result == "clarify"

    def test_custom_threshold_one(self):
        """threshold = 1.0일 때"""
        state = {"filter_confidence": 0.9}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=1.0
        )
        # confidence(0.9) <= threshold(1.0) 이므로 clarify
        assert result == "clarify"

    def test_confidence_equal_to_one_at_threshold_one(self):
        """confidence = 1.0, threshold = 1.0일 때"""
        state = {"filter_confidence": 1.0}
        result = should_clarify(
            state, enable_clarification=True, clarification_threshold=1.0
        )
        # 1.0 <= 1.0 이므로 clarify
        assert result == "clarify"
