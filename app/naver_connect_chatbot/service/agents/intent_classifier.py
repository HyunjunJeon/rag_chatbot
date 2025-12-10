"""
Adaptive RAG용 의도 분류 에이전트 구현.

사용자 질문을 다양한 의도 카테고리로 분류해 적응형 검색 전략을 돕습니다.

Note:
    CLOVA HCX-007은 tools와 reasoning을 동시에 지원하지 않으므로,
    with_structured_output() 패턴을 사용하여 구조화된 출력을 생성합니다.
"""

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.prompts import get_prompt


class IntentClassification(BaseModel):
    """
    의도 분류 결과와 근거를 담는 모델입니다.

    속성:
        intent: SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, CLARIFICATION_NEEDED, OUT_OF_DOMAIN 중 하나
        confidence: 0.0~1.0 사이 신뢰도
        reasoning: 분류 결정 근거
        domain_relevance: 도메인 관련성 점수 (0.0~1.0)
    """

    intent: str = Field(
        description="Classified intent: SIMPLE_QA | COMPLEX_REASONING | EXPLORATORY | CLARIFICATION_NEEDED | OUT_OF_DOMAIN"
    )
    confidence: float = Field(description="Confidence score (0.0 ~ 1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation for the classification")
    domain_relevance: float = Field(
        default=1.0,
        description="Domain relevance score (0.0 ~ 1.0). Low score (<0.3) indicates OUT_OF_DOMAIN",
        ge=0.0,
        le=1.0,
    )


def classify_intent(question: str, llm: Runnable) -> IntentClassification:
    """
    사용자 질문의 의도를 분류합니다.

    매개변수:
        question: 분류할 사용자 질문
        llm: 분류에 사용할 언어 모델

    반환값:
        IntentClassification 결과

    예시:
        >>> from naver_connect_chatbot.config import get_chat_model
        >>> llm = get_chat_model()
        >>> result = classify_intent("What is PyTorch?", llm)
        >>> print(result.intent)
        SIMPLE_QA
    """
    try:
        # 프롬프트 템플릿 로드
        prompt_template = get_prompt("intent_classification")
        system_prompt = (
            prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        )

        # 전체 프롬프트 구성
        full_prompt = f"{system_prompt}\n\nUser question: {question}"

        # with_structured_output 사용하여 LLM 직접 호출
        structured_llm = _get_structured_llm(llm, IntentClassification)
        result = structured_llm.invoke(full_prompt)

        if isinstance(result, IntentClassification):
            return result

        # fallback
        logger.warning(f"Unexpected result type: {type(result)}")
        return IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.5,
            reasoning="Unable to classify intent properly",
            domain_relevance=1.0,
        )

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.5,
            reasoning=f"Classification error: {str(e)}",
            domain_relevance=1.0,
        )


async def aclassify_intent(question: str, llm: Runnable) -> IntentClassification:
    """
    사용자 질문의 의도를 비동기로 분류합니다.

    매개변수:
        question: 분류할 사용자 질문
        llm: 분류에 사용할 언어 모델

    반환값:
        IntentClassification 결과
    """
    try:
        # 프롬프트 템플릿 로드
        prompt_template = get_prompt("intent_classification")
        system_prompt = (
            prompt_template.messages[0].prompt.template if prompt_template.messages else ""
        )

        # 전체 프롬프트 구성
        full_prompt = f"{system_prompt}\n\nUser question: {question}"

        # with_structured_output 사용하여 LLM 직접 호출
        structured_llm = _get_structured_llm(llm, IntentClassification)
        result = await structured_llm.ainvoke(full_prompt)

        if isinstance(result, IntentClassification):
            return result

        # fallback
        logger.warning(f"Unexpected result type: {type(result)}")
        return IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.5,
            reasoning="Unable to classify intent properly",
            domain_relevance=1.0,
        )

    except Exception as e:
        logger.error(f"Async intent classification failed: {e}")
        return IntentClassification(
            intent="SIMPLE_QA",
            confidence=0.5,
            reasoning=f"Classification error: {str(e)}",
            domain_relevance=1.0,
        )


def _get_structured_llm(llm: Runnable, schema: type[BaseModel]) -> Runnable:
    """
    LLM에 structured output을 적용합니다.

    with_structured_output()을 지원하지 않는 LLM의 경우
    fallback으로 JSON 파싱을 시도합니다.

    매개변수:
        llm: 언어 모델
        schema: 출력 스키마 (Pydantic 모델)

    반환값:
        구조화된 출력을 생성하는 Runnable
    """
    with_structured = getattr(llm, "with_structured_output", None)
    if callable(with_structured):
        return with_structured(schema)

    # Fallback: PydanticOutputParser 사용
    logger.warning(
        "LLM does not support with_structured_output, using PydanticOutputParser fallback"
    )
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=schema)

    # 파서와 함께 호출하는 래퍼 반환
    class ParserWrapper:
        def __init__(self, llm, parser):
            self._llm = llm
            self._parser = parser

        def invoke(self, prompt):
            result = self._llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            return self._parser.parse(content)

        async def ainvoke(self, prompt):
            result = await self._llm.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            return self._parser.parse(content)

    return ParserWrapper(llm, parser)


# Deprecated: 이전 버전과의 호환성을 위해 유지
def create_intent_classifier(llm: Runnable):
    """
    Deprecated: classify_intent() 또는 aclassify_intent()를 직접 사용하세요.

    이전 버전 호환성을 위한 래퍼 클래스를 반환합니다.
    """
    logger.warning(
        "create_intent_classifier() is deprecated. "
        "Use classify_intent() or aclassify_intent() directly."
    )

    class IntentClassifierWrapper:
        def __init__(self, llm):
            self._llm = llm

        def invoke(self, input_dict):
            question = input_dict.get("question", "")
            return classify_intent(question, self._llm)

        async def ainvoke(self, input_dict):
            question = input_dict.get("question", "")
            return await aclassify_intent(question, self._llm)

    return IntentClassifierWrapper(llm)
