"""
Adaptive RAG 시스템에서 사용하는 상태 정의.

기본 AgentState와 확장된 AdaptiveRAGState 구조를 제공합니다.
"""

from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
import operator

from naver_connect_chatbot.service.graph.types import RetrievalFilters


# 하위 호환성을 위한 기본 AgentState
class AgentState(TypedDict):
    """
    단순 RAG 워크플로에서 사용하는 기본 상태 구조입니다.

    속성:
        messages: 대화 메시지 시퀀스
        context: 검색된 문서
        answer: 생성된 답변
        question: 사용자 질문
        retry_count: 질의 변환 재시도 횟수
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: List[Document]
    answer: str
    question: str
    retry_count: int


class AdaptiveRAGState(TypedDict, total=False):
    """
    의도 분류, 질의 분석, 검색, 답변 생성을 포함한 Adaptive RAG 확장 상태입니다.

    의도 분류부터 질의 분석, 검색, 답변 생성까지의 전체 워크플로를 추적합니다.

    속성:
        # Input
        question: 사용자의 원본 질문
        messages: 대화 메시지 기록

        # Intent Classification
        intent: 분류된 의도 (SIMPLE_QA | COMPLEX_REASONING | EXPLORATORY | CLARIFICATION_NEEDED | OUT_OF_DOMAIN)
        intent_confidence: 의도 분류 신뢰도 (0.0 ~ 1.0)
        intent_reasoning: 의도 분류 근거
        domain_relevance: 도메인 관련성 점수 (0.0 ~ 1.0, 낮으면 OUT_OF_DOMAIN)

        # Query Processing
        original_query: 사용자의 원본 질의
        refined_queries: 개선·확장된 질의 목록
        query_analysis: 상세 질의 분석 결과

        # Retrieval
        context: 검색된 문서 (documents 별칭)
        documents: 검색된 문서
        retrieval_strategy: 사용된 검색 전략

        # Answer Generation
        answer: 생성된 답변
        generation_metadata: 답변 생성 관련 메타데이터
        generation_strategy: 사용된 생성 전략

        # Out-of-Domain
        is_out_of_domain: OOD 질문 여부

        # Control Flow
        retry_count: 검색 재시도 횟수
        max_retries: 허용되는 최대 재시도 수
        workflow_stage: 현재 워크플로 단계
    """

    # Input
    question: str
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Intent Classification
    intent: str
    intent_confidence: float
    intent_reasoning: str
    domain_relevance: float  # 도메인 관련성 점수 (0.0 ~ 1.0)

    # Query Processing
    original_query: str
    refined_queries: List[str]
    query_analysis: Dict[str, Any]
    filter_confidence: float  # 필터 추출 신뢰도 (0.0 ~ 1.0, 낮으면 명확화 필요)

    # Retrieval
    context: List[Document]  # 하위 호환성 유지용 필드
    documents: List[Document]
    retrieval_strategy: str
    retrieval_filters: RetrievalFilters  # 메타 기반 검색 필터
    retrieval_filters_applied: bool  # 필터가 실제 적용되었는지 여부
    retrieval_fallback_used: bool  # 0건으로 인한 폴백 사용 여부

    # Answer Generation
    answer: str
    generation_metadata: Dict[str, Any]
    generation_strategy: str

    # Out-of-Domain
    is_out_of_domain: bool  # OOD 질문 여부

    # Control Flow
    retry_count: int
    max_retries: int
    workflow_stage: str
