"""
Adaptive RAG용 답변 검증 에이전트 구현.

생성된 답변의 환각, 근거, 완전성, 품질을 점검합니다.
"""

from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_prompt
from naver_connect_chatbot.config import logger


class AnswerValidation(BaseModel):
    """
    다양한 품질 검사 결과를 담는 답변 검증 모델입니다.
    
    속성:
        has_hallucination: 답변에 환각 정보가 포함되었는지 여부
        is_grounded: 제공된 문맥에 근거했는지 여부
        is_complete: 질문에 완전히 답했는지 여부
        quality_score: 전체 품질 점수 (0.0 ~ 1.0)
        issues: 발견된 문제 목록
    """
    has_hallucination: bool = Field(
        description="Whether the answer contains hallucinations"
    )
    is_grounded: bool = Field(
        description="Whether the answer is grounded in context"
    )
    is_complete: bool = Field(
        description="Whether the answer is complete"
    )
    quality_score: float = Field(
        description="Overall quality score (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    issues: list[str] = Field(
        description="List of identified issues",
        default_factory=list
    )


@tool(args_schema=AnswerValidation)
def emit_answer_validation(
    has_hallucination: Annotated[bool, Field(description="Whether the answer contains hallucinations")],
    is_grounded: Annotated[bool, Field(description="Whether the answer is grounded in context")],
    is_complete: Annotated[bool, Field(description="Whether the answer is complete")],
    quality_score: Annotated[float, Field(description="Overall quality score (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    issues: Annotated[list[str], Field(description="List of identified issues")],
) -> AnswerValidation:
    """
    Emit structured answer validation results.
    
    Call this tool after validating the generated answer to return quality assessment
    including hallucination detection, grounding, and completeness checks.
    
    Args:
        has_hallucination: True if answer contains unsupported claims
        is_grounded: True if answer is grounded in provided context
        is_complete: True if answer fully addresses the question
        quality_score: Overall quality score (0.0 ~ 1.0)
        issues: List of specific issues identified
    
    Returns:
        AnswerValidation instance with all validation results
    """
    return AnswerValidation(
        has_hallucination=has_hallucination,
        is_grounded=is_grounded,
        is_complete=is_complete,
        quality_score=quality_score,
        issues=issues,
    )


def create_answer_validator(llm: Runnable) -> Any:
    """
    답변을 검증하는 에이전트를 생성합니다.
    
    다음 항목을 확인하여 품질을 점검합니다.
    - Hallucinations: 문맥에 없는 정보가 포함되었는가?
    - Grounding: 제공된 문맥에 기반했는가?
    - Completeness: 질문에 충분히 답했는가?
    - Quality: 명확하고 일관된 구조인가?
    
    매개변수:
        llm: 검증에 사용할 언어 모델
    
    반환값:
        답변을 검증하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> validator = create_answer_validator(llm)
        >>> result = validator.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "question: What is PyTorch?\ncontext: ...\nanswer: ..."
        ...     }]
        ... })
    """
    try:        
        prompt_template = get_prompt("answer_validation")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        
        enhanced_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: After validating the answer, you MUST call the emit_answer_validation tool "
            "with all required parameters to return your validation results."
        )
        
        agent = create_agent(
            model=llm,
            tools=[emit_answer_validation],
            system_prompt=enhanced_prompt,
            name="answer_validator",
        )
        
        logger.debug("Answer validator agent created successfully with emit_answer_validation tool")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create answer validator: {e}")
        raise


def validate_answer(
    question: str,
    context: list[Document],
    answer: str,
    llm: Runnable
) -> AnswerValidation:
    """
    단일 답변을 검증하는 편의 함수입니다.
    
    매개변수:
        question: 사용자 질문
        context: 검색된 문서 컨텍스트
        answer: 검증 대상 답변
        llm: 사용할 언어 모델
    
    반환값:
        AnswerValidation 검증 결과
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.documents import Document
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> docs = [Document(page_content="PyTorch is a framework")]
        >>> answer = "PyTorch is a deep learning framework for Python"
        >>> result = validate_answer("What is PyTorch?", docs, answer, llm)
        >>> print(result.has_hallucination)
        False
    """
    validator = create_answer_validator(llm)
    
    # 검증에 사용할 문맥을 포맷합니다.
    context_text = "\n\n".join([
        f"[문서 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(context)
    ])
    
    response: AnswerValidation = validator.invoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\n\ncontext:\n{context_text}\n\nanswer:\n{answer}"
        }]
    })
    
    return response

