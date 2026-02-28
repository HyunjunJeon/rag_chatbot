"""
B-2: 워크플로 라우팅 단위 테스트.

route_after_intent() Hard/Soft OOD 라우팅 로직을 검증합니다.
LLM 호출 없이 순수 상태(dict) 기반으로 테스트합니다.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from naver_connect_chatbot.service.graph.workflow import (
    route_after_intent,
    OOD_HARD_THRESHOLD,
    OOD_SOFT_THRESHOLD,
)


# ============================================================================
# B-2-1: Hard OOD 라우팅
# ============================================================================


class TestRouteAfterIntentHardOOD:
    """Hard OOD: intent=OUT_OF_DOMAIN, domain_relevance < 0.2 → generate_ood_response"""

    def test_hard_ood_routes_to_ood_response(self):
        """확실한 OOD → generate_ood_response"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.1}
        result = route_after_intent(state)
        assert result == "generate_ood_response"

    def test_hard_ood_with_zero_relevance(self):
        """domain_relevance=0.0 → generate_ood_response"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.0}
        result = route_after_intent(state)
        assert result == "generate_ood_response"

    def test_hard_ood_just_below_threshold(self):
        """domain_relevance=0.19 (Hard OOD 임계값 직전) → generate_ood_response"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.19}
        result = route_after_intent(state)
        assert result == "generate_ood_response"

    def test_hard_ood_greeting_pattern_relevance(self):
        """인사말 패턴 매칭 시 relevance=0.0 → generate_ood_response"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.0}
        result = route_after_intent(state)
        assert result == "generate_ood_response"


# ============================================================================
# B-2-2: Soft OOD 라우팅
# ============================================================================


class TestRouteAfterIntentSoftOOD:
    """Soft OOD: intent=OUT_OF_DOMAIN, 0.2 ≤ domain_relevance < 0.5 → analyze_query"""

    def test_soft_ood_routes_to_analyze_query(self):
        """Soft OOD → analyze_query (검색 먼저 시도)"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.3}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_soft_ood_at_hard_threshold_boundary(self):
        """domain_relevance=0.2 (Hard 임계값 경계, Soft OOD 시작) → analyze_query"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.2}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_soft_ood_just_below_soft_threshold(self):
        """domain_relevance=0.49 (Soft 임계값 직전) → analyze_query"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.49}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_soft_ood_midpoint(self):
        """domain_relevance=0.35 (Soft OOD 중간) → analyze_query"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.35}
        result = route_after_intent(state)
        assert result == "analyze_query"


# ============================================================================
# B-2-3: In-domain 라우팅
# ============================================================================


class TestRouteAfterIntentInDomain:
    """In-domain: domain_relevance ≥ 0.5 → analyze_query"""

    def test_in_domain_simple_qa(self):
        """SIMPLE_QA, 높은 relevance → analyze_query"""
        state = {"intent": "SIMPLE_QA", "domain_relevance": 0.8}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_in_domain_complex_reasoning(self):
        """COMPLEX_REASONING → analyze_query"""
        state = {"intent": "COMPLEX_REASONING", "domain_relevance": 0.9}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_in_domain_exploratory(self):
        """EXPLORATORY → analyze_query"""
        state = {"intent": "EXPLORATORY", "domain_relevance": 0.7}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_in_domain_at_soft_threshold(self):
        """domain_relevance=0.5 (Soft 임계값 경계) → analyze_query"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.5}
        result = route_after_intent(state)
        assert result == "analyze_query"


# ============================================================================
# B-2-4: intent가 OUT_OF_DOMAIN이 아닌 경우 → 항상 analyze_query
# ============================================================================


class TestRouteAfterIntentNonOOD:
    """intent가 OUT_OF_DOMAIN이 아닌 경우 → 항상 analyze_query"""

    def test_simple_qa_with_low_relevance(self):
        """SIMPLE_QA + 낮은 relevance → OOD가 아니므로 analyze_query"""
        state = {"intent": "SIMPLE_QA", "domain_relevance": 0.1}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_complex_reasoning_with_zero_relevance(self):
        """COMPLEX_REASONING + domain_relevance=0.0 → analyze_query"""
        state = {"intent": "COMPLEX_REASONING", "domain_relevance": 0.0}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_clarification_needed_with_low_relevance(self):
        """CLARIFICATION_NEEDED + 낮은 relevance → analyze_query"""
        state = {"intent": "CLARIFICATION_NEEDED", "domain_relevance": 0.05}
        result = route_after_intent(state)
        assert result == "analyze_query"


# ============================================================================
# B-2-5: 경계값 (Boundary) 테스트
# ============================================================================


class TestRouteAfterIntentBoundary:
    """경계값 테스트 (off-by-one 버그 방지)"""

    def test_exactly_at_hard_threshold_is_soft_ood(self):
        """domain_relevance=0.2 (Hard 임계값 정확히) → analyze_query (Hard OOD 조건 미충족)"""
        # Hard OOD 조건: domain_relevance < 0.2 (strictly less than)
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": OOD_HARD_THRESHOLD}
        result = route_after_intent(state)
        # 0.2 < 0.2 는 False이므로 Hard OOD 아님 → analyze_query
        assert result == "analyze_query"

    def test_just_below_hard_threshold_is_hard_ood(self):
        """domain_relevance=0.199... → Hard OOD"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": 0.199}
        result = route_after_intent(state)
        assert result == "generate_ood_response"

    def test_exactly_at_soft_threshold_goes_to_analyze(self):
        """domain_relevance=0.5 (Soft 임계값 정확히) → analyze_query"""
        state = {"intent": "OUT_OF_DOMAIN", "domain_relevance": OOD_SOFT_THRESHOLD}
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_default_intent_and_relevance(self):
        """상태에 intent/domain_relevance 없으면 기본값 사용 → analyze_query"""
        state = {}  # 빈 상태 → 기본값 intent="SIMPLE_QA", domain_relevance=1.0
        result = route_after_intent(state)
        assert result == "analyze_query"

    def test_thresholds_have_expected_values(self):
        """임계값 상수가 기대값인지 확인"""
        assert OOD_HARD_THRESHOLD == 0.2
        assert OOD_SOFT_THRESHOLD == 0.5
