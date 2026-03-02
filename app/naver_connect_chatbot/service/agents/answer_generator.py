"""
Adaptive RAG용 답변 생성 에이전트 구현.

검색된 문맥과 질문 의도에 따라 서로 다른 생성 전략을 적용합니다.

Note:
    tools/function calling을 사용하지 않고 LLM을 직접 호출하여
    CLOVA HCX-007의 reasoning 모드와 호환됩니다.
"""

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.prompts import get_prompt


def _render_prompt_text(prompt_template: ChatPromptTemplate, **kwargs) -> str:
    """템플릿에 선언된 변수만 사용해 프롬프트를 문자열로 렌더링합니다."""
    prompt_inputs = {k: v for k, v in kwargs.items() if k in prompt_template.input_variables}
    return prompt_template.format_prompt(**prompt_inputs).to_string()


def generate_answer(
    question: str,
    context: str,
    llm: Runnable,
    strategy: str = "simple",
    conversation_history: str = "",
) -> str:
    """
    검색된 문맥을 기반으로 답변을 생성합니다.

    매개변수:
        question: 사용자 질문
        context: 검색된 문맥
        llm: 답변 생성에 사용할 언어 모델
        strategy: 생성 전략 (simple, complex, exploratory)

    반환값:
        생성된 답변 텍스트

    예시:
        >>> from naver_connect_chatbot.config import get_chat_model
        >>> llm = get_chat_model()
        >>> answer = generate_answer(
        ...     "What is PyTorch?",
        ...     "PyTorch is a framework...",
        ...     llm,
        ...     strategy="simple"
        ... )
    """
    try:
        # 프롬프트 템플릿 로드
        prompt_template: ChatPromptTemplate = get_prompt(
            prompt_name=f"answer_generation_{strategy}",
            return_type="template",
        )
        history_text = conversation_history.strip() or "No prior conversation."
        full_prompt = _render_prompt_text(
            prompt_template,
            question=question,
            context=context,
            conversation_history=history_text,
        )

        # LLM 직접 호출
        result: AIMessage = llm.invoke(full_prompt)

        # 결과에서 content 추출
        if hasattr(result, "content"):
            return result.content
        return str(result)

    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"


async def agenerate_answer(
    question: str,
    context: str,
    llm: Runnable,
    strategy: str = "simple",
    conversation_history: str = "",
) -> str:
    """
    검색된 문맥을 기반으로 비동기로 답변을 생성합니다.

    매개변수:
        question: 사용자 질문
        context: 검색된 문맥
        llm: 답변 생성에 사용할 언어 모델
        strategy: 생성 전략 (simple, complex, exploratory)

    반환값:
        생성된 답변 텍스트
    """
    try:
        # 프롬프트 템플릿 로드
        prompt_template: ChatPromptTemplate = get_prompt(
            prompt_name=f"answer_generation_{strategy}",
            return_type="template",
        )
        history_text = conversation_history.strip() or "No prior conversation."
        full_prompt = _render_prompt_text(
            prompt_template,
            question=question,
            context=context,
            conversation_history=history_text,
        )

        # LLM 직접 호출
        result: AIMessage = await llm.ainvoke(full_prompt)

        # 결과에서 content 추출
        if hasattr(result, "content"):
            return result.content
        return str(result)

    except Exception as e:
        logger.error(f"Async answer generation failed: {e}", exc_info=True)
        return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"


def get_generation_strategy(intent: str) -> str:
    """
    주어진 의도에 대응하는 생성 전략을 반환합니다.

    매개변수:
        intent: 질문 의도 코드

    반환값:
        생성 전략 이름

    예시:
        >>> get_generation_strategy("COMPLEX_REASONING")
        'complex'
    """
    strategy_map = {
        "SIMPLE_QA": "simple",
        "COMPLEX_REASONING": "complex",
        "EXPLORATORY": "exploratory",
        "CLARIFICATION_NEEDED": "simple",
    }
    return strategy_map.get(intent, "simple")


# Deprecated: 이전 버전과의 호환성을 위해 유지
# def create_answer_generator(llm: Runnable, strategy: str = "simple"):
#     """
#     Deprecated: generate_answer() 또는 agenerate_answer()를 직접 사용하세요.

#     이전 버전 호환성을 위한 래퍼 클래스를 반환합니다.
#     """
#     logger.warning("create_answer_generator() is deprecated. Use generate_answer() or agenerate_answer() directly.")

#     class AnswerGeneratorWrapper:
#         def __init__(self, llm, strategy):
#             self._llm = llm
#             self._strategy = strategy

#         def invoke(self, input_dict):
#             question = input_dict.get("question", "")
#             context = input_dict.get("context", "")
#             answer = generate_answer(question, context, self._llm, self._strategy)
#             # AIMessage 형태로 반환 (호환성)
#             from langchain_core.messages import AIMessage

#             return AIMessage(content=answer)

#         async def ainvoke(self, input_dict):
#             question = input_dict.get("question", "")
#             context = input_dict.get("context", "")
#             answer = await agenerate_answer(question, context, self._llm, self._strategy)
#             # AIMessage 형태로 반환 (호환성)
#             from langchain_core.messages import AIMessage

#             return AIMessage(content=answer)

#     return AnswerGeneratorWrapper(llm, strategy)
