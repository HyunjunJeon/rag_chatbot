"""
LangGraph Node 반환 타입 정의.

각 노드 함수의 반환 타입을 TypedDict로 명시하여 타입 안전성을 강화합니다.
이를 통해 IDE 자동완성, 타입 체커, 그리고 명확한 계약(contract)을 제공합니다.
"""

from typing import TypedDict, Literal
from langchain_core.documents import Document


# ============================================================================
# 필터 타입 (다른 타입들이 참조하므로 먼저 정의)
# ============================================================================


class RetrievalFilters(TypedDict, total=False):
    """메타 기반 검색 필터.

    의도 분석 결과를 기반으로 검색 범위를 좁히는 필터입니다.
    Qdrant payload 필터와 BM25 후처리 필터에 사용됩니다.

    속성:
        doc_type: 문서 유형 필터 (예: ["slack_qa", "pdf", "notebook", "mission"])
        course: 코스 이름 필터 (예: "데이터분석")
        course_level: 코스 난이도 필터 (예: "초급", "중급", "고급")
        course_topic: 코스 주제 필터
        generation: 기수 필터 (예: "1기", "2기")
        year: 연도 필터 (예: "2024")
        year_month: 연월 필터 (예: "2024-01")
    """

    doc_type: list[str]
    course: str
    course_level: str
    course_topic: str
    generation: str
    year: str
    year_month: str


# ============================================================================
# Node 반환 타입
# ============================================================================


class IntentUpdate(TypedDict, total=False):
    """classify_intent_node 반환 타입.

    Intent 분류 결과를 state에 업데이트합니다.
    """

    intent: Literal["SIMPLE_QA", "COMPLEX_REASONING", "EXPLORATORY", "CLARIFICATION_NEEDED"]
    intent_confidence: float
    intent_reasoning: str


class QueryAnalysisUpdate(TypedDict, total=False):
    """analyze_query_node 반환 타입.

    Query 분석 결과를 state에 업데이트합니다.
    """

    refined_query: str
    refined_queries: list[str]
    original_query: str
    query_analysis: dict
    retrieval_filters: RetrievalFilters


class RetrievalUpdate(TypedDict, total=False):
    """retrieve_node 반환 타입.

    검색된 문서와 메타데이터를 state에 업데이트합니다.
    """

    documents: list[Document]
    context: list[Document]  # 하위 호환성
    retrieval_strategy: str
    retrieval_filters_applied: bool
    retrieval_fallback_used: bool
    retrieval_metadata: dict


class DocumentEvaluationUpdate(TypedDict, total=False):
    """evaluate_documents_node 반환 타입.

    문서 평가 결과를 state에 업데이트합니다.
    """

    sufficient_context: bool
    relevant_doc_count: int
    document_evaluation: dict


class AnswerUpdate(TypedDict, total=False):
    """generate_answer_node 반환 타입.

    생성된 답변과 메타데이터를 state에 업데이트합니다.
    """

    answer: str
    answer_metadata: dict


class ErrorUpdate(TypedDict, total=False):
    """에러 발생 시 공통 반환 타입.

    에러 정보와 폴백 사용 여부를 state에 업데이트합니다.
    """

    error: str
    error_node: str
    fallback_used: bool
