"""
Adaptive RAG용 답변 생성 에이전트 구현.

검색된 문맥과 질문 의도에 따라 서로 다른 생성 전략을 적용합니다.
"""

from langchain.agents import create_agent

from langchain_core.runnables import Runnable

from naver_connect_chatbot.config import logger
from naver_connect_chatbot.prompts import get_prompt


def create_answer_generator(llm: Runnable, strategy: str = "simple") -> Runnable:
    """
    답변 생성 에이전트를 초기화합니다.

    검색된 문맥과 질문 의도에 따라 다음 전략을 사용합니다.
    - simple: SIMPLE_QA용 간결하고 직접적인 답변
    - complex: COMPLEX_REASONING용 단계별 분석
    - exploratory: EXPLORATORY용 구조화된 개요

    매개변수:
        llm: 답변 생성을 수행할 언어 모델
        strategy: 생성 전략 (simple, complex, exploratory)

    반환값:
        AnswerOutput을 도구 호출을 통해 반환하는 Agent (Runnable)

    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> generator = create_answer_generator(llm, strategy="simple")
        >>> result = await generator.ainvoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "question: What is PyTorch?\\ncontext: PyTorch is a framework..."
        ...     }]
        ... })
    """
    try:
        # 프롬프트 로드
        prompt_template = get_prompt(f"answer_generation_{strategy}")

        # System prompt 추출
        system_prompt = ""
        if prompt_template.messages:
            system_prompt = prompt_template.messages[0].prompt.template

        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=str(system_prompt),
            name=f"answer_generator_{strategy}",
        )

        return agent

    except Exception as e:
        logger.error(f"Failed to create Answer agent: {e}", exc_info=True)
        raise


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
