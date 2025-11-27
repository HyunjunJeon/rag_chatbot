"""
Adaptive RAG용 답변 생성 에이전트 구현.

검색된 문맥과 질문 의도에 따라 서로 다른 생성 전략을 적용합니다.
"""

from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from naver_connect_chatbot.config import logger


class AnswerOutput(BaseModel):
    """Generated answer based on retrieved context and question intent.

    This model wraps the answer text to enable validation and maintain
    consistency with other agents in the Adaptive RAG workflow.
    """

    answer: str = Field(
        description="The generated answer to the user's question based on retrieved context"
    )


def create_answer_generator(llm: Runnable, strategy: str = "simple") -> Runnable:
    """
    답변 생성 에이전트를 초기화합니다 (구조화된 출력 패턴 사용).

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
    from naver_connect_chatbot.service.agents.factory import create_structured_agent

    return create_structured_agent(
        llm=llm,
        agent_name=f"answer_generator_{strategy}",
        prompt_key=f"answer_generation_{strategy}",
        output_model=AnswerOutput,
        tool_name=f"emit_answer_{strategy}",
        tool_description="Emit your generated answer after analyzing the context and question",
    )


def generate_answer(question: str, context: list[Document], intent: str, llm: Runnable) -> str:
    """
    단일 질문에 대한 답변을 생성하는 편의 함수입니다.

    매개변수:
        question: 사용자 질문
        context: 검색된 문서 컨텍스트
        intent: 전략 결정을 위한 질문 의도
        llm: 사용할 언어 모델

    반환값:
        생성된 답변 문자열

    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.documents import Document
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> docs = [Document(page_content="PyTorch is a deep learning framework")]
        >>> answer = generate_answer("What is PyTorch?", docs, "SIMPLE_QA", llm)
        >>> print(answer)
        PyTorch is a deep learning framework...
    """
    # 의도별로 생성 전략을 매핑합니다.
    strategy_map = {
        "SIMPLE_QA": "simple",
        "COMPLEX_REASONING": "complex",
        "EXPLORATORY": "exploratory",
        "CLARIFICATION_NEEDED": "simple",  # Fallback to simple
    }
    strategy = strategy_map.get(intent, "simple")

    generator = create_answer_generator(llm, strategy=strategy)

    # 생성에 사용할 문맥을 포맷합니다.
    context_text = "\n\n".join(
        [f"[문서 {i + 1}]\n{doc.page_content}" for i, doc in enumerate(context)]
    )

    response_raw = generator.invoke(
        {
            "messages": [
                {"role": "user", "content": f"question: {question}\n\ncontext:\n{context_text}"}
            ]
        }
    )

    # Use response parser for structured extraction
    from naver_connect_chatbot.service.agents.response_parser import parse_agent_response

    try:
        response = parse_agent_response(
            response_raw,
            model_type=AnswerOutput,
            fallback=AnswerOutput(answer="죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."),
        )
        return response.answer
    except Exception as e:
        logger.error(f"Failed to parse answer response: {e}")
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."


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
