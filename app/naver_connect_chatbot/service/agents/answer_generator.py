"""
Adaptive RAG용 답변 생성 에이전트 구현.

검색된 문맥과 질문 의도에 따라 서로 다른 생성 전략을 적용합니다.
"""

from typing import Any, List
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_answer_generation_prompt
from naver_connect_chatbot.config import logger


def create_answer_generator(
    llm: Runnable,
    strategy: str = "simple"
) -> Any:
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
        문맥 기반 답변을 생성하는 에이전트
        
    예시:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> generator = create_answer_generator(llm, strategy="simple")
        >>> result = generator.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "question: What is PyTorch?\ncontext: PyTorch is a framework..."
        ...     }]
        ... })
    """
    try:
        # 전략에 맞는 생성 프롬프트를 불러옵니다.
        system_prompt = get_answer_generation_prompt(strategy=strategy)
        
        # 도구 없이 순수 생성만 수행하는 에이전트를 구성합니다.
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=system_prompt,
        )
        
        logger.debug(f"Answer generator agent created successfully with strategy: {strategy}")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create answer generator: {e}")
        raise


def generate_answer(
    question: str,
    context: List[Document],
    intent: str,
    llm: Runnable
) -> str:
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
    context_text = "\n\n".join([
        f"[문서 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(context)
    ])
    
    response = generator.invoke({
        "messages": [{
            "role": "user",
            "content": f"question: {question}\n\ncontext:\n{context_text}"
        }]
    })
    
    # 응답 객체에서 실제 답변을 추출합니다.
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "output" in response:
        return response["output"]
    elif isinstance(response, str):
        return response
    else:
        # 폴백 처리
        logger.warning(f"Unexpected response format from answer generator: {type(response)}")
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

