"""
Adaptive RAG용 의도 분류 에이전트 구현.

사용자 질문을 다양한 의도 카테고리로 분류해 적응형 검색 전략을 돕습니다.
"""

import json
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_prompt
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


@tool(args_schema=IntentClassification)
def emit_intent_classification(
    intent: Annotated[str, Field(description="Classified intent: SIMPLE_QA | COMPLEX_REASONING | EXPLORATORY | CLARIFICATION_NEEDED")],
    confidence: Annotated[float, Field(description="Confidence score (0.0 ~ 1.0)", ge=0.0, le=1.0)],
    reasoning: Annotated[str, Field(description="Explanation for the classification")],
) -> IntentClassification:
    """
    Emit structured intent classification results.
    
    Call this tool after analyzing the user's question to return the classification
    with confidence score and reasoning.
    
    Args:
        intent: The classified intent category
        confidence: Classification confidence (0.0 ~ 1.0)
        reasoning: Detailed explanation for why this intent was chosen
    
    Returns:
        IntentClassification instance with all classification results
    """
    return IntentClassification(
        intent=intent,
        confidence=confidence,
        reasoning=reasoning,
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
        prompt_template = get_prompt("intent_classification")
        system_prompt = prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        
        # 도구 사용 가이드 추가
        schema_text = json.dumps(
            IntentClassification.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        enhanced_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: After classifying the intent, you MUST call the emit_intent_classification tool "
            "with all required parameters (intent, confidence, reasoning) to return your classification."
            "\n\nReturn outputs that match this JSON schema exactly (no extra fields, no prose):\n"
            f"{schema_text}"
        )
        
        agent = create_agent(
            model=llm,
            tools=[emit_intent_classification],
            system_prompt=enhanced_prompt,
            name="intent_classifier",
        )
        
        logger.debug("Intent classifier agent created successfully with emit_intent_classification tool")
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
    
    # 에이전트 응답에서 ToolMessage를 찾아 IntentClassification을 추출합니다.
    try:
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            for msg in reversed(messages):
                is_tool_msg = (
                    msg.__class__.__name__ == "ToolMessage" or
                    (hasattr(msg, "type") and msg.type == "tool")
                )
                
                if is_tool_msg:
                    tool_content = msg.content
                    if isinstance(tool_content, IntentClassification):
                        logger.debug("Successfully extracted IntentClassification from ToolMessage")
                        return tool_content
                    elif isinstance(tool_content, dict):
                        logger.debug("Converting dict content to IntentClassification")
                        return IntentClassification(**tool_content)
                    elif isinstance(tool_content, str):
                        try:
                            logger.debug("Attempting to parse string content as IntentClassification")
                            import re
                            data = {}
                            
                            # intent 추출
                            intent_match = re.search(r"intent='?\"?([^'\"]+)'?\"?", tool_content)
                            if intent_match:
                                data["intent"] = intent_match.group(1)
                            
                            # confidence 추출
                            conf_match = re.search(r"confidence=([\d.]+)", tool_content)
                            if conf_match:
                                data["confidence"] = float(conf_match.group(1))
                            
                            # reasoning 추출
                            reasoning_match = re.search(r"reasoning='?\"?([^'\"]+(?:[^'\"]*[^'\"]+)?)'?\"?", tool_content)
                            if reasoning_match:
                                data["reasoning"] = reasoning_match.group(1)
                            
                            if data:
                                logger.debug(f"Successfully parsed IntentClassification from string: {data}")
                                return IntentClassification(**data)
                        except Exception as e:
                            logger.warning(f"Failed to parse string content: {e}")
        
        # 직접 반환된 경우
        if isinstance(response, IntentClassification):
            return response
        
        logger.warning(f"Unable to extract IntentClassification from response: {type(response)}")
        
    except Exception as e:
        logger.error(f"Error extracting IntentClassification from response: {e}")
    
    # 최종 폴백
    return IntentClassification(
        intent="CLARIFICATION_NEEDED",
        confidence=0.5,
        reasoning="Unable to classify intent properly"
    )
