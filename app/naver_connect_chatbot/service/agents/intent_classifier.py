"""
Adaptive RAG용 의도 분류 에이전트 구현.

사용자 질문을 다양한 의도 카테고리로 분류해 적응형 검색 전략을 돕습니다.
"""

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_intent_classification_prompt
from naver_connect_chatbot.config import logger


class IntentClassification(BaseModel):
    """
    의도 분류 결과와 근거를 담는 모델입니다.
    
    속성:
        intent: SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, CLARIFICATION_NEEDED 중 하나
        confidence: 0.0~1.0 사이 신뢰도
        reasoning: 분류 결정 근거
    """
    intent: str = Field(
        description="Classified intent: SIMPLE_QA | COMPLEX_REASONING | EXPLORATORY | CLARIFICATION_NEEDED"
    )
    confidence: float = Field(
        description="Confidence score (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation for the classification"
    )


def create_intent_classifier(llm: Runnable) -> Any:
    """
    의도 분류 에이전트를 생성합니다.
    
    사용자 질문을 다음 네 가지 카테고리로 분류합니다.
    - SIMPLE_QA: 단순 사실형 질문
    - COMPLEX_REASONING: 복잡한 추론이 필요한 질문
    - EXPLORATORY: 탐색형 질문
    - CLARIFICATION_NEEDED: 모호하거나 추가 설명이 필요한 질문
    
    매개변수:
        llm: 분류에 사용할 언어 모델
    
    반환값:
        사용자 의도를 분류하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> classifier = create_intent_classifier(llm)
        >>> result = classifier.invoke({"messages": [{"role": "user", "content": "What is PyTorch?"}]})
    """
    try:
        # 분류 프롬프트를 불러옵니다.
        system_prompt = get_intent_classification_prompt()
        
        # 구조화된 출력을 반환하도록 LLM을 구성합니다.
        structured_llm = llm.with_structured_output(IntentClassification)
        
        # 도구 없이 분류만 수행하는 에이전트를 생성합니다.
        agent = create_agent(
            model=structured_llm,
            tools=[],
            system_prompt=system_prompt,
        )
        
        logger.debug("Intent classifier agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create intent classifier: {e}")
        raise


def classify_intent(question: str, llm: Runnable) -> IntentClassification:
    """
    단일 질문을 분류하는 편의 함수입니다.
    
    매개변수:
        question: 분류할 사용자 질문
        llm: 사용할 언어 모델
    
    반환값:
        IntentClassification 결과
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> result = classify_intent("What is PyTorch?", llm)
        >>> print(result.intent)
        SIMPLE_QA
    """
    classifier = create_intent_classifier(llm)
    response = classifier.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    
    # 응답에서 구조화된 출력을 추출합니다.
    if hasattr(response, "content") and isinstance(response.content, IntentClassification):
        return response.content
    elif isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        # 폴백: 기본 분류 결과를 반환합니다.
        logger.warning(f"Unexpected response format from intent classifier: {type(response)}")
        return IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.5,
            reasoning="Default classification due to unexpected response format"
        )

