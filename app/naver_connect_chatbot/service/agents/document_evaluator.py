"""
Adaptive RAG용 문서 평가 에이전트 구현.

검색된 문서의 관련성과 충분성을 판별합니다.
"""

from typing import Any, List
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_document_evaluation_prompt
from naver_connect_chatbot.config import logger


class DocumentEvaluation(BaseModel):
    """
    문서 관련성과 충분성 평가 결과를 담는 모델입니다.
    
    속성:
        relevant_count: 관련 문서 개수
        irrelevant_count: 비관련 문서 개수
        sufficient: 문서가 답변에 충분한지 여부
        confidence: 평가 신뢰도 (0.0 ~ 1.0)
        improvement_suggestions: 검색 개선 제안
    """
    relevant_count: int = Field(
        description="Number of relevant documents",
        ge=0
    )
    irrelevant_count: int = Field(
        description="Number of irrelevant documents",
        ge=0
    )
    sufficient: bool = Field(
        description="Whether documents are sufficient to answer the question"
    )
    confidence: float = Field(
        description="Confidence in the evaluation (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    improvement_suggestions: List[str] = Field(
        description="Suggestions for improving retrieval",
        default_factory=list
    )


def create_document_evaluator(llm: Runnable) -> Any:
    """
    문서 평가 에이전트를 생성합니다.
    
    다음 기준으로 문서를 평가합니다.
    - 사용자 질문과의 관련성
    - 충분한 답변 제공 여부
    - 정보의 품질과 명확성
    
    매개변수:
        llm: 평가에 사용할 언어 모델
    
    반환값:
        문서를 평가하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> evaluator = create_document_evaluator(llm)
        >>> docs = [Document(page_content="PyTorch is a deep learning framework")]
        >>> result = evaluator.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": f"question: What is PyTorch?\ndocuments: {docs}"
        ...     }]
        ... })
    """
    try:
        # 평가 프롬프트를 불러옵니다.
        system_prompt = get_document_evaluation_prompt()
        
        # 구조화된 출력을 반환하도록 LLM을 구성합니다.
        structured_llm = llm.with_structured_output(DocumentEvaluation)
        
        # 도구 없이 평가만 수행하는 에이전트를 생성합니다.
        agent = create_agent(
            model=structured_llm,
            tools=[],
            system_prompt=system_prompt,
        )
        
        logger.debug("Document evaluator agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create document evaluator: {e}")
        raise


def evaluate_documents(
    question: str,
    documents: List[Document],
    llm: Runnable
) -> DocumentEvaluation:
    """
    단일 질문에 대해 문서를 평가하는 편의 함수입니다.
    
    매개변수:
        question: 사용자 질문
        documents: 평가할 검색 문서
        llm: 사용할 언어 모델
    
    반환값:
        DocumentEvaluation 평가 결과
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.documents import Document
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> docs = [Document(page_content="PyTorch is a framework")]
        >>> result = evaluate_documents("What is PyTorch?", docs, llm)
        >>> print(result.sufficient)
        True
    """
    evaluator = create_document_evaluator(llm)
    
    # 평가에 사용할 문서를 포맷합니다.
    doc_text = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content[:500]}..."  # Limit length
        for i, doc in enumerate(documents)
    ])
    
    response = evaluator.invoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\n\ndocuments:\n{doc_text}"
        }]
    })
    
    # 응답에서 구조화된 출력을 추출합니다.
    if hasattr(response, "content") and isinstance(response.content, DocumentEvaluation):
        return response.content
    elif isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        # 폴백: 기본 평가 결과를 반환합니다.
        logger.warning(f"Unexpected response format from document evaluator: {type(response)}")
        return DocumentEvaluation(
            relevant_count=len(documents),
            irrelevant_count=0,
            sufficient=len(documents) > 0,
            confidence=0.5,
            improvement_suggestions=["Unable to evaluate documents properly"]
        )

