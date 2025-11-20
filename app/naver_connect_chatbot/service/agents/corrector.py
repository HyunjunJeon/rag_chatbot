"""
Adaptive RAG용 교정 에이전트 구현.

검증 실패 원인을 분석하여 적절한 교정 전략을 제안합니다.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_correction_prompt
from naver_connect_chatbot.config import logger


class CorrectionStrategy(BaseModel):
    """
    교정 액션과 피드백을 담는 모델입니다.
    
    속성:
        action: 추천 교정 액션 (REGENERATE | REFINE_QUERY | ADD_CONTEXT | CLARIFY)
        feedback: 개선을 위한 상세 피드백
        priority: 우선순위 (HIGH | MEDIUM | LOW)
    """
    action: str = Field(
        description="Correction action: REGENERATE | REFINE_QUERY | ADD_CONTEXT | CLARIFY"
    )
    feedback: str = Field(
        description="Detailed feedback for improvement"
    )
    priority: str = Field(
        description="Priority level: HIGH | MEDIUM | LOW"
    )


def create_corrector(llm: Runnable) -> Any:
    """
    교정 전략을 제안하는 에이전트를 생성합니다.
    
    검증 실패 원인에 따라 다음 전략을 결정합니다.
    - REGENERATE: 답변을 완전히 새로 생성
    - REFINE_QUERY: 질의를 개선하고 재검색
    - ADD_CONTEXT: 추가 문맥 검색
    - CLARIFY: 특정 부분을 명확히 설명
    
    매개변수:
        llm: 교정 분석에 사용할 언어 모델
    
    반환값:
        교정 전략을 산출하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> corrector = create_corrector(llm)
        >>> validation_result = {"has_hallucination": True, "issues": ["..."]}
        >>> result = corrector.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": f"validation_result: {validation_result}\nanswer: ..."
        ...     }]
        ... })
    """
    try:
        # 교정 프롬프트를 불러옵니다.
        system_prompt = get_correction_prompt()
        
        # 구조화된 출력을 반환하도록 LLM을 구성합니다.
        structured_llm = llm.with_structured_output(CorrectionStrategy)
        
        # 도구 없이 분석만 수행하는 에이전트를 생성합니다.
        agent = create_agent(
            model=structured_llm,
            tools=[],
            system_prompt=system_prompt,
        )
        
        logger.debug("Corrector agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create corrector: {e}")
        raise


def determine_correction_strategy(
    validation_result: Dict[str, Any],
    answer: str,
    llm: Runnable
) -> CorrectionStrategy:
    """
    교정 전략을 계산하는 편의 함수입니다.
    
    매개변수:
        validation_result: 답변 검증 결과
        answer: 검증에 실패한 답변
        llm: 사용할 언어 모델
    
    반환값:
        추천 교정 전략
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> validation = {
        ...     "has_hallucination": True,
        ...     "is_grounded": False,
        ...     "quality_score": 0.3,
        ...     "issues": ["Contains unsupported claims"]
        ... }
        >>> strategy = determine_correction_strategy(validation, "Answer text", llm)
        >>> print(strategy.action)
        REGENERATE
    """
    corrector = create_corrector(llm)
    
    # 검증 결과를 문자열로 정리합니다.
    validation_text = "\n".join([
        f"{key}: {value}"
        for key, value in validation_result.items()
    ])
    
    response = corrector.invoke({
        "messages": [{
            "role": "user",
            "content": f"validation_result:\n{validation_text}\n\nanswer:\n{answer}"
        }]
    })
    
    # 응답에서 구조화된 출력을 추출합니다.
    if hasattr(response, "content") and isinstance(response.content, CorrectionStrategy):
        return response.content
    elif isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        # 폴백: 기본 전략을 반환합니다.
        logger.warning(f"Unexpected response format from corrector: {type(response)}")
        
        # 검증 결과를 기반으로 액션을 추론합니다.
        if validation_result.get("has_hallucination"):
            action = "REGENERATE"
            priority = "HIGH"
        elif not validation_result.get("is_complete"):
            action = "ADD_CONTEXT"
            priority = "MEDIUM"
        else:
            action = "CLARIFY"
            priority = "LOW"
        
        return CorrectionStrategy(
            action=action,
            feedback="Unable to analyze correction strategy properly",
            priority=priority
        )

