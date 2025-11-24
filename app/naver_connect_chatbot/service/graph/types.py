"""
LangGraph Node 반환 타입 정의.

각 노드 함수의 반환 타입을 TypedDict로 명시하여 타입 안전성을 강화합니다.
이를 통해 IDE 자동완성, 타입 체커, 그리고 명확한 계약(contract)을 제공합니다.
"""

from typing import TypedDict, Literal
from langchain_core.documents import Document


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
    query_analysis: dict


class RetrievalUpdate(TypedDict, total=False):
    """retrieve_node 반환 타입.

    검색된 문서와 메타데이터를 state에 업데이트합니다.
    """
    documents: list[Document]
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
