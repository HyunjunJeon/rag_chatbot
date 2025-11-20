"""
Adaptive RAG 시스템에서 사용하는 상태 정의.

기본 AgentState와 확장된 AdaptiveRAGState 구조를 제공합니다.
"""

from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
import operator


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
    의도 분류와 검증 단계를 포함한 Adaptive RAG 확장 상태입니다.
    
    의도 분류부터 질의 분석, 검색, 생성, 검증, 교정까지 전체 워크플로를 추적합니다.
    
    속성:
        # Input
        question: 사용자의 원본 질문
        messages: 대화 메시지 기록
        
        # Intent Classification
        intent: 분류된 의도 (SIMPLE_QA | COMPLEX_REASONING | EXPLORATORY | CLARIFICATION_NEEDED)
        intent_confidence: 의도 분류 신뢰도 (0.0 ~ 1.0)
        intent_reasoning: 의도 분류 근거
        
        # Query Processing
        original_query: 사용자의 원본 질의
        refined_queries: 개선·확장된 질의 목록
        query_analysis: 상세 질의 분석 결과
        
        # Retrieval
        context: 검색된 문서 (documents 별칭)
        documents: 검색된 문서
        retrieval_strategy: 사용된 검색 전략
        
        # Document Evaluation
        document_evaluation: 문서 평가 결과
        sufficient_context: 문맥이 충분한지 여부
        relevant_doc_count: 관련 문서 수
        
        # Answer Generation
        answer: 생성된 답변
        generation_metadata: 답변 생성 관련 메타데이터
        generation_strategy: 사용된 생성 전략
        
        # Validation
        validation_result: 상세 검증 결과
        has_hallucination: 환각 여부
        is_grounded: 답변이 문맥에 근거했는지 여부
        is_complete: 답변이 완전한지 여부
        quality_score: 전반적 품질 점수 (0.0 ~ 1.0)
        validation_issues: 발견된 문제 목록
        
        # Correction
        correction_count: 교정 반복 횟수
        correction_feedback: 교정 피드백 메시지 목록
        correction_action: 권장 교정 액션
        
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
    
    # Query Processing
    original_query: str
    refined_queries: List[str]
    query_analysis: Dict[str, Any]
    
    # Retrieval
    context: List[Document]  # 하위 호환성 유지용 필드
    documents: List[Document]
    retrieval_strategy: str
    
    # Document Evaluation
    document_evaluation: Dict[str, Any]
    sufficient_context: bool
    relevant_doc_count: int
    
    # Answer Generation
    answer: str
    generation_metadata: Dict[str, Any]
    generation_strategy: str
    
    # Validation
    validation_result: Dict[str, Any]
    has_hallucination: bool
    is_grounded: bool
    is_complete: bool
    quality_score: float
    validation_issues: List[str]
    
    # Correction
    correction_count: int
    correction_feedback: List[str]
    correction_action: str
    
    # Control Flow
    retry_count: int
    max_retries: int
    workflow_stage: str
