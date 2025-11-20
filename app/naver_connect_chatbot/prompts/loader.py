"""
YAML 기반 프롬프트를 로드하고 관리하기 위한 도우미.

이 모듈은 YAML 파일에서 ChatPromptTemplate 인스턴스를 생성하고,
캐싱·검증·폴백 메커니즘을 제공합니다.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from naver_connect_chatbot.config import logger

# 프롬프트 템플릿 디렉터리
PROMPTS_DIR = Path(__file__).parent / "templates"


class PromptMetadata(BaseModel):
    """프롬프트 템플릿 메타데이터 스키마."""

    name: str
    description: str
    version: str = "1.0"
    author: str | None = None
    last_updated: str | None = None


class PromptConfig(BaseModel):
    """YAML 프롬프트 파일 구성을 검증하는 스키마."""

    prompt_type: Literal["simple", "chat_messages"] = Field(alias="_type")
    template: str | None = None
    messages: List[Dict[str, str]] | None = None
    input_variables: List[str] = Field(default_factory=list)
    output_parser: Dict[str, Any] | None = None
    metadata: PromptMetadata

    model_config = {"populate_by_name": True}


class PromptLoadError(Exception):
    """프롬프트 로딩 실패 시 발생하는 예외."""

    pass


@lru_cache(maxsize=32)
def load_prompt_config(prompt_name: str) -> PromptConfig:
    """
    YAML 파일에서 프롬프트 구성을 읽어 검증합니다.

    매개변수:
        prompt_name: 확장자를 제외한 프롬프트 이름

    반환값:
        PromptConfig: 검증된 설정 인스턴스

    예외:
        PromptLoadError: 파일이 없거나 검증에 실패한 경우
    """
    yaml_path = PROMPTS_DIR / f"{prompt_name}.yaml"

    if not yaml_path.exists():
        raise PromptLoadError(f"Prompt file not found: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = PromptConfig(**data)
        logger.debug(f"Loaded prompt config: {config.metadata.name}")
        return config

    except yaml.YAMLError as e:
        raise PromptLoadError(f"YAML parsing error in {yaml_path}: {e}") from e
    except ValidationError as e:
        raise PromptLoadError(f"Validation error in {yaml_path}: {e}") from e


def _build_messages(messages_config: List[Dict[str, str]]) -> List[tuple[str, str]]:
    """
    구성 정보를 LangChain 메시지 튜플로 변환합니다.

    매개변수:
        messages_config: role·content 키를 가진 메시지 목록

    반환값:
        ChatPromptTemplate.from_messages()에 전달할 (role, content) 목록

    예외:
        PromptLoadError: 지원되지 않는 role이 포함된 경우
    """
    valid_roles = {"system", "human", "ai", "user", "assistant"}
    messages = []

    for msg_config in messages_config:
        role = msg_config.get("role", "").lower()
        content = msg_config.get("content", "")

        if role not in valid_roles:
            raise PromptLoadError(f"Unknown message role: {role}. Must be one of {valid_roles}")

        messages.append((role, content))

    return messages


def load_prompt(prompt_name: str, fallback: ChatPromptTemplate | None = None) -> ChatPromptTemplate:
    """
    YAML 기반 ChatPromptTemplate을 로드하며 폴백을 지원합니다.

    YAML 구성을 읽어 적절한 ChatPromptTemplate을 생성하며,
    실패 시 폴백 템플릿이 있으면 이를 반환합니다.

    매개변수:
        prompt_name: 확장자를 제외한 프롬프트 이름
        fallback: 로드 실패 시 사용할 선택적 폴백 템플릿

    반환값:
        ChatPromptTemplate 인스턴스

    예외:
        PromptLoadError: 로드에 실패했고 폴백도 없는 경우

    예시:
        >>> prompt = load_prompt("rag_generation")
        >>> result = prompt.format(question="What is AI?", context="AI is...")

        >>> # With fallback
        >>> fallback = ChatPromptTemplate.from_template("Fallback: {query}")
        >>> prompt = load_prompt("missing_prompt", fallback=fallback)
    """
    try:
        config = load_prompt_config(prompt_name)

        if config.prompt_type == "simple":
            if not config.template:
                raise PromptLoadError("Simple prompt missing 'template' field")

            # ChatPromptTemplate.from_template() auto-detects input_variables from template
            # Don't pass input_variables explicitly to avoid conflicts
            return ChatPromptTemplate.from_template(config.template)

        elif config.prompt_type == "chat_messages":
            if not config.messages:
                raise PromptLoadError("Chat prompt missing 'messages' field")

            messages = _build_messages(config.messages)
            return ChatPromptTemplate.from_messages(messages)

        else:
            raise PromptLoadError(f"Unknown prompt type: {config.prompt_type}")

    except PromptLoadError as e:
        logger.error(f"Failed to load prompt '{prompt_name}': {e}")

        if fallback is not None:
            logger.warning(f"Using fallback prompt for '{prompt_name}'")
            return fallback

        raise


def reload_prompts() -> None:
    """
    프롬프트 캐시를 비워 다음 접근 시 강제로 다시 로드합니다.

    애플리케이션을 재시작하지 않고 프롬프트 파일을 수정할 때 유용합니다.

    예시:
        >>> reload_prompts()
        >>> prompt = load_prompt("rag_generation")  # Will reload from disk
    """
    load_prompt_config.cache_clear()
    logger.info("Prompt cache cleared")


# 프롬프트별 편의 함수


def get_rag_generation_prompt() -> ChatPromptTemplate:
    """
    RAG 답변 생성 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 문맥 기반 답변 생성을 위한 프롬프트

    예시:
        >>> prompt = get_rag_generation_prompt()
        >>> result = prompt.format(question="What is AI?", context="AI is...")
    """
    fallback = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question}\nContext: {context}\nAnswer:"
    )
    return load_prompt("rag_generation", fallback=fallback)


def get_document_grading_prompt() -> ChatPromptTemplate:
    """
    문서 관련성 평가 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 문서 관련성을 평가하는 프롬프트

    예시:
        >>> prompt = get_document_grading_prompt()
        >>> result = prompt.format(question="Test?", context="Doc content")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document to a user question. \nHere is the retrieved document: \n\n {context} \n\nHere is the user question: {question} \nIf the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \nGive a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
            ),
            ("human", "{question}"),
        ]
    )
    return load_prompt("document_grading", fallback=fallback)


def get_query_transformation_prompt() -> ChatPromptTemplate:
    """
    질의 변환 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 질의를 재구성하기 위한 프롬프트

    예시:
        >>> prompt = get_query_transformation_prompt()
        >>> result = prompt.format(question="Original question")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are generating a question that is well optimized for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.",
            ),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    return load_prompt("query_transformation", fallback=fallback)


def get_multi_query_generation_prompt() -> ChatPromptTemplate:
    """
    다중 질의 생성 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 복수 검색 질의를 생성하기 위한 프롬프트

    예시:
        >>> prompt = get_multi_query_generation_prompt()
        >>> result = prompt.format(query="Search query", num=3)
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior search-strategy analyst.\n\n[Goal]\n- Surface both explicit and implicit intents in the user's request.\n- Produce {num} precise search queries that cover the discovered intents.\n\n[Process (invisible to the user)]\n- Step 1: Parse the user's wording to capture explicit objectives.\n- Step 2: Infer hidden or contextual needs that may be implied.\n- Step 3: Draft focused search directions that cover complementary\n  viewpoints (e.g., definitions, causes, comparisons, best practices,\n  tooling).\n- Step 4: Validate that queries are mutually distinct and collectively\n  exhaustive for the user's research task.\n\n[Output Requirements]\n- Return only the final rewritten queries.\n- Place exactly one query per line with no numbering or commentary.\n- Use concrete terminology; avoid metaphors, rhetorical questions,\n  pronouns without referents, or vague qualifiers.\n- Keep each query self-contained so it stands alone without the\n  original prompt.\n\nThink step by step and output the final queries.",
            ),
            ("human", "{query}"),
        ]
    )
    return load_prompt("multi_query_generation", fallback=fallback)


def get_intent_classification_prompt() -> ChatPromptTemplate:
    """
    의도 분류 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 사용자 의도를 분류하기 위한 프롬프트

    예시:
        >>> prompt = get_intent_classification_prompt()
        >>> result = prompt.format(question="What is PyTorch?")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's question intent into one of: SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, CLARIFICATION_NEEDED",
            ),
            ("human", "Question: {question}"),
        ]
    )
    return load_prompt("intent_classification", fallback=fallback)


def get_query_analysis_prompt() -> ChatPromptTemplate:
    """
    질의 분석 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 질의 품질을 분석하는 프롬프트

    예시:
        >>> prompt = get_query_analysis_prompt()
        >>> result = prompt.format(question="Question", intent="SIMPLE_QA")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Analyze the query quality and suggest improvements for better retrieval.",
            ),
            ("human", "Question: {question}\nIntent: {intent}"),
        ]
    )
    return load_prompt("query_analysis", fallback=fallback)


def get_document_evaluation_prompt() -> ChatPromptTemplate:
    """
    문서 평가 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 검색된 문서를 평가하는 프롬프트

    예시:
        >>> prompt = get_document_evaluation_prompt()
        >>> result = prompt.format(question="Question", documents="Docs")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Evaluate the relevance and sufficiency of retrieved documents.",
            ),
            ("human", "Question: {question}\n\nDocuments:\n{documents}"),
        ]
    )
    return load_prompt("document_evaluation", fallback=fallback)


def get_answer_generation_prompt(strategy: str = "simple") -> ChatPromptTemplate:
    """
    전략에 따라 답변 생성 프롬프트를 반환합니다.

    매개변수:
        strategy: simple/complex/exploratory 중 사용할 생성 전략

    반환값:
        ChatPromptTemplate: 답변 생성을 위한 프롬프트

    예시:
        >>> prompt = get_answer_generation_prompt("simple")
        >>> result = prompt.format(question="Q", context="Context")
    """
    prompt_name = f"answer_generation_{strategy}"
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the question based on the provided context.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    return load_prompt(prompt_name, fallback=fallback)


def get_answer_validation_prompt() -> ChatPromptTemplate:
    """
    답변 검증 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 답변 품질을 검증하는 프롬프트

    예시:
        >>> prompt = get_answer_validation_prompt()
        >>> result = prompt.format(question="Q", context="C", answer="A")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Validate the answer for hallucinations and quality issues.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
        ]
    )
    return load_prompt("answer_validation", fallback=fallback)


def get_correction_prompt() -> ChatPromptTemplate:
    """
    교정 전략 프롬프트를 반환합니다.

    반환값:
        ChatPromptTemplate: 교정 전략을 결정하기 위한 프롬프트

    예시:
        >>> prompt = get_correction_prompt()
        >>> result = prompt.format(validation_result={...}, answer="A")
    """
    fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Analyze the validation result and suggest a correction strategy.",
            ),
            ("human", "Validation Result: {validation_result}\n\nAnswer: {answer}"),
        ]
    )
    return load_prompt("correction", fallback=fallback)


__all__ = [
    "load_prompt",
    "reload_prompts",
    "get_rag_generation_prompt",
    "get_document_grading_prompt",
    "get_query_transformation_prompt",
    "get_multi_query_generation_prompt",
    "get_intent_classification_prompt",
    "get_query_analysis_prompt",
    "get_document_evaluation_prompt",
    "get_answer_generation_prompt",
    "get_answer_validation_prompt",
    "get_correction_prompt",
    "PromptLoadError",
    "PromptConfig",
    "PromptMetadata",
]
