"""
Agent 생성 팩토리 함수.

이 모듈은 100+ 라인의 중복된 agent 생성 코드를 단일 팩토리로 통합하여
일관성을 보장하고 유지보수를 용이하게 합니다.
"""

import json
from typing import Type, Callable
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from naver_connect_chatbot.prompts import get_prompt
from naver_connect_chatbot.config import logger


def create_structured_agent(
    llm: Runnable,
    *,
    agent_name: str,
    prompt_key: str,
    output_model: Type[BaseModel],
    tool_name: str | None = None,
    tool_description: str | None = None,
    additional_instructions: str = "",
) -> Runnable:
    """
    구조화된 출력을 가진 에이전트를 생성합니다.

    이 함수는 모든 agent 생성 로직을 표준화합니다:
    1. 프롬프트 로드 (prompts/{prompt_key}.yaml)
    2. Tool 자동 생성 (emit_{agent_name}_tool)
    3. System prompt 구성 (원본 + tool 사용 지시)
    4. LangChain Agent 생성
    5. 로깅 및 에러 핸들링

    매개변수:
        llm: 에이전트에 사용할 언어 모델
        agent_name: 에이전트 이름 (예: "intent_classifier")
        prompt_key: 프롬프트 파일 키 (예: "intent_classification")
        output_model: Tool의 출력 Pydantic 모델
        tool_name: Tool 이름 (기본값: f"emit_{agent_name}")
        tool_description: Tool 설명 (기본값: f"Emit {agent_name} result")
        additional_instructions: System prompt에 추가할 지시사항

    반환값:
        구성된 Agent (Runnable - CompiledStateGraph)

    예외:
        Exception: 에이전트 생성 실패 시

    예시:
        >>> from pydantic import BaseModel
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> class IntentResult(BaseModel):
        ...     intent: str
        ...     confidence: float
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> agent = create_structured_agent(
        ...     llm=llm,
        ...     agent_name="intent_classifier",
        ...     prompt_key="intent_classification",
        ...     output_model=IntentResult,
        ... )
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    """
    try:
        # 1. 프롬프트 로드
        logger.debug(f"Loading prompt for {agent_name} (key: {prompt_key})")
        prompt_template = get_prompt(prompt_key)

        # System prompt 추출
        system_prompt = ""
        if prompt_template.messages:
            system_prompt = prompt_template.messages[0].prompt.template

        # 2. Tool 생성
        if tool_name is None:
            tool_name = f"emit_{agent_name}"

        if tool_description is None:
            tool_description = f"Emit {agent_name} result with structured output"

        # Tool 함수 동적 생성
        def _emit_result(**kwargs) -> output_model:
            """Emit structured result"""
            return output_model(**kwargs)

        # StructuredTool 사용 (name과 description을 명시적으로 설정 가능)
        emit_tool = StructuredTool.from_function(
            func=_emit_result,
            name=tool_name,
            description=tool_description,
        )

        logger.debug(f"Created tool: {tool_name}")

        # 3. System prompt 구성
        schema_text = json.dumps(
            output_model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        tool_usage_instruction = (
            f"\n\nIMPORTANT: After completing your task, you MUST call the '{tool_name}' tool "
            f"with all required parameters to return your structured result."
        )
        schema_instruction = (
            "\n\nReturn outputs that match this JSON schema exactly (no extra fields, no prose):\n"
            f"{schema_text}"
        )

        enhanced_prompt = system_prompt
        if additional_instructions:
            enhanced_prompt += f"\n\n{additional_instructions}"
        enhanced_prompt += tool_usage_instruction
        enhanced_prompt += schema_instruction

        # 4. Agent 생성
        logger.debug(f"Creating agent: {agent_name}")
        agent = create_agent(
            model=llm,
            tools=[emit_tool],
            system_prompt=enhanced_prompt,
            name=agent_name,
        )

        logger.info(f"✓ {agent_name} agent created successfully")
        return agent

    except Exception as e:
        logger.error(
            f"Failed to create {agent_name} agent",
            error=str(e),
            exc_info=True
        )
        raise


def create_simple_tool_from_function(
    func: Callable,
    name: str,
    description: str,
) -> Callable:
    """
    일반 함수를 LangChain Tool로 변환합니다.

    매개변수:
        func: 변환할 함수
        name: Tool 이름
        description: Tool 설명

    반환값:
        LangChain Tool

    예시:
        >>> def my_function(x: int, y: int) -> int:
        ...     return x + y
        >>>
        >>> tool = create_simple_tool_from_function(
        ...     my_function,
        ...     "add_numbers",
        ...     "Add two numbers together"
        ... )
    """
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
    )


__all__ = [
    "create_structured_agent",
    "create_simple_tool_from_function",
]
