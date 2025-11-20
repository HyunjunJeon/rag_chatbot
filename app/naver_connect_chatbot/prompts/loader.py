"""
YAML 기반 프롬프트를 로드하고 관리하기 위한 도우미.

이 모듈은 YAML 파일에서 ChatPromptTemplate 인스턴스를 생성하고,
캐싱·검증·폴백 메커니즘을 제공합니다.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from naver_connect_chatbot.config import logger

# 프롬프트 템플릿 디렉터리
PROMPTS_DIR = Path(__file__).parent / "templates"

PromptMessageConfig = dict[str, str]
PromptReturnFormat = Literal["template", "text"]


@dataclass(frozen=True)
class PromptFallback:
    """프롬프트 폴백 정의를 표현하는 불변 객체."""

    prompt_type: Literal["simple", "chat_messages"]
    template: str | None = None
    messages: tuple[PromptMessageConfig, ...] | None = None

    def build_template(self) -> ChatPromptTemplate:
        """폴백으로 사용할 ChatPromptTemplate을 생성합니다."""
        if self.prompt_type == "simple" and self.template:
            return ChatPromptTemplate.from_template(self.template)

        if self.prompt_type == "chat_messages" and self.messages:
            return ChatPromptTemplate.from_messages(_build_messages(list(self.messages)))

        raise PromptLoadError("Fallback definition is incomplete")

    def render_text(self) -> str:
        """폴백 프롬프트의 텍스트 표현을 반환합니다."""
        if self.prompt_type == "simple" and self.template:
            return self.template

        if self.prompt_type == "chat_messages" and self.messages:
            messages = _build_messages(list(self.messages))
            return _format_messages_as_text(messages)

        raise PromptLoadError("Fallback definition is incomplete")


DEFAULT_ANSWER_GENERATION_FALLBACK = PromptFallback(
    prompt_type="chat_messages",
    messages=(
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question based on the "
                "provided context."
            ),
        },
        {
            "role": "human",
            "content": "Question: {question}\n\nContext:\n{context}",
        },
    ),
)

PROMPT_FALLBACKS: Dict[str, PromptFallback] = {
    "rag_generation": PromptFallback(
        prompt_type="simple",
        template=(
            "You are an assistant for question-answering tasks. Use the "
            "following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. Use "
            "three sentences maximum and keep the answer concise.\n\n"
            "Question: {question}\nContext: {context}\nAnswer:"
        ),
    ),
    "document_grading": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "You are a grader assessing relevance of a retrieved "
                    "document to a user question.\nHere is the retrieved "
                    "document:\n\n{context}\n\nHere is the user "
                    "question: {question}\nIf the document contains "
                    "keyword(s) or semantic meaning related to the user "
                    "question, grade it as relevant.\nGive a binary score "
                    "'yes' or 'no' score to indicate whether the document is "
                    "relevant to the question."
                ),
            },
            {"role": "human", "content": "{question}"},
        ),
    ),
    "query_transformation": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "You are generating a question that is well optimized for "
                    "retrieval. Look at the input and try to reason about the "
                    "underlying semantic intent / meaning."
                ),
            },
            {
                "role": "human",
                "content": (
                    "Here is the initial question:\n\n {question} \n Formulate "
                    "an improved question."
                ),
            },
        ),
    ),
    "multi_query_generation": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "You are a senior search-strategy analyst.\n\n[Goal]\n- "
                    "Surface both explicit and implicit intents in the user's "
                    "request.\n- Produce {num} precise search queries that "
                    "cover the discovered intents.\n\n[Process (invisible to "
                    "the user)]\n- Step 1: Parse the user's wording to capture "
                    "explicit objectives.\n- Step 2: Infer hidden or "
                    "contextual needs that may be implied.\n- Step 3: Draft "
                    "focused search directions that cover complementary "
                    "viewpoints (e.g., definitions, causes, comparisons, best "
                    "practices, tooling).\n- Step 4: Validate that queries are "
                    "mutually distinct and collectively exhaustive for the "
                    "user's research task.\n\n[Output Requirements]\n- Return "
                    "only the final rewritten queries.\n- Place exactly one "
                    "query per line with no numbering or commentary.\n- Use "
                    "concrete terminology; avoid metaphors, rhetorical "
                    "questions, pronouns without referents, or vague "
                    "qualifiers.\n- Keep each query self-contained so it "
                    "stands alone without the original prompt.\n\nThink step "
                    "by step and output the final queries."
                ),
            },
            {"role": "human", "content": "{query}"},
        ),
    ),
    "intent_classification": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "Classify the user's question intent into one of: "
                    "SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, "
                    "CLARIFICATION_NEEDED"
                ),
            },
            {"role": "human", "content": "Question: {question}"},
        ),
    ),
    "query_analysis": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "Analyze the query quality and suggest improvements for "
                    "better retrieval."
                ),
            },
            {"role": "human", "content": "Question: {question}\nIntent: {intent}"},
        ),
    ),
    "document_evaluation": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "Evaluate the relevance and sufficiency of retrieved "
                    "documents."
                ),
            },
            {
                "role": "human",
                "content": "Question: {question}\n\nDocuments:\n{documents}",
            },
        ),
    ),
    "answer_generation_simple": DEFAULT_ANSWER_GENERATION_FALLBACK,
    "answer_generation_complex": DEFAULT_ANSWER_GENERATION_FALLBACK,
    "answer_generation_exploratory": DEFAULT_ANSWER_GENERATION_FALLBACK,
    "answer_validation": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "Validate the answer for hallucinations and quality issues."
                ),
            },
            {
                "role": "human",
                "content": (
                    "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
                ),
            },
        ),
    ),
    "correction": PromptFallback(
        prompt_type="chat_messages",
        messages=(
            {
                "role": "system",
                "content": (
                    "Analyze the validation result and suggest a correction "
                    "strategy."
                ),
            },
            {
                "role": "human",
                "content": (
                    "Validation Result: {validation_result}\n\nAnswer: {answer}"
                ),
            },
        ),
    ),
}


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
    messages: list[dict[str, str]] | None = None
    input_variables: list[str] = Field(default_factory=list)
    output_parser: dict[str, Any] | None = None
    metadata: PromptMetadata

    model_config = {"populate_by_name": True}


class PromptLoadError(Exception):
    """프롬프트 로딩 실패 시 발생하는 예외."""

    pass

@lru_cache(maxsize=None)
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


def _build_messages(messages_config: Sequence[PromptMessageConfig]) -> list[tuple[str, str]]:
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


def _format_messages_as_text(messages: Sequence[tuple[str, str]]) -> str:
    """
    메시지 튜플을 사람이 읽을 수 있는 문자열로 변환합니다.

    매개변수:
        messages: (role, content) 튜플 목록

    반환값:
        역할 헤더와 본문이 포함된 문자열
    """
    formatted_blocks: list[str] = []

    for role, content in messages:
        header = f"[{role.upper()}]"
        formatted_blocks.append(f"{header}\n{content}".strip())

    return "\n\n".join(formatted_blocks)


def _build_template_from_config(config: PromptConfig) -> ChatPromptTemplate:
    """PromptConfig를 LangChain ChatPromptTemplate으로 변환합니다."""
    if config.prompt_type == "simple":
        if not config.template:
            raise PromptLoadError("Simple prompt missing 'template' field")
        return ChatPromptTemplate.from_template(config.template)

    if config.prompt_type == "chat_messages":
        if not config.messages:
            raise PromptLoadError("Chat prompt missing 'messages' field")
        messages = _build_messages(config.messages)
        return ChatPromptTemplate.from_messages(messages)

    raise PromptLoadError(f"Unknown prompt type: {config.prompt_type}")


def _render_text_from_config(config: PromptConfig) -> str:
    """PromptConfig를 순수 텍스트 형태로 직렬화합니다."""
    if config.prompt_type == "simple":
        if not config.template:
            raise PromptLoadError("Simple prompt missing 'template' field")
        return config.template

    if config.prompt_type == "chat_messages":
        if not config.messages:
            raise PromptLoadError("Chat prompt missing 'messages' field")
        messages = _build_messages(config.messages)
        return _format_messages_as_text(messages)

    raise PromptLoadError(f"Unknown prompt type: {config.prompt_type}")


def _load_prompt_template(
    prompt_name: str,
    *,
    fallback: ChatPromptTemplate | None = None,
) -> ChatPromptTemplate:
    """
    PromptConfig를 템플릿으로 변환하고 폴백을 적용합니다.
    """
    try:
        config = load_prompt_config(prompt_name)
        return _build_template_from_config(config)
    except PromptLoadError as exc:
        logger.error("Failed to load prompt '%s': %s", prompt_name, exc)
        if fallback is not None:
            logger.warning("Using fallback prompt for '%s'", prompt_name)
            return fallback
        raise


def _load_prompt_text(
    prompt_name: str,
    *,
    fallback_text: str | None = None,
) -> str:
    """
    PromptConfig를 문자열 표현으로 변환하고 폴백을 적용합니다.
    """
    try:
        config = load_prompt_config(prompt_name)
        return _render_text_from_config(config)
    except PromptLoadError as exc:
        logger.error("Failed to load prompt '%s': %s", prompt_name, exc)
        if fallback_text is not None:
            logger.warning("Using fallback text for '%s'", prompt_name)
            return fallback_text
        raise


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
    return _load_prompt_template(prompt_name, fallback=fallback)


def list_available_prompts() -> list[str]:
    """
    디스크와 폴백 레지스트리에 존재하는 프롬프트 이름 목록을 반환합니다.
    """
    disk_prompts = {path.stem for path in PROMPTS_DIR.glob("*.yaml")}
    registry_prompts = set(PROMPT_FALLBACKS.keys())
    return sorted(disk_prompts | registry_prompts)


def get_prompt(
    prompt_name: str,
    *,
    return_type: PromptReturnFormat = "template",
) -> ChatPromptTemplate | str:
    """
    프롬프트를 템플릿 또는 순수 텍스트로 반환합니다.

    매개변수:
        prompt_name: 로드할 프롬프트 파일 이름(확장자 제외)
        return_type: 'template'이면 ChatPromptTemplate, 'text'이면 문자열 반환

    반환값:
        ChatPromptTemplate 또는 str
    """
    if return_type == "template":
        fallback_def = PROMPT_FALLBACKS.get(prompt_name)
        fallback_template = fallback_def.build_template() if fallback_def else None
        return _load_prompt_template(prompt_name, fallback=fallback_template)

    if return_type == "text":
        fallback_def = PROMPT_FALLBACKS.get(prompt_name)
        fallback_text = fallback_def.render_text() if fallback_def else None
        return _load_prompt_text(prompt_name, fallback_text=fallback_text)

    raise ValueError(f"Unsupported return_type: {return_type}")


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
