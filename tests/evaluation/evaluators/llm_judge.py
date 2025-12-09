"""
LLM-as-Judge 평가기 구현.

HyperClovaX HCX-007을 사용하여 RAG 답변 품질을 평가합니다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import JudgeEvaluation


def load_prompt(filename: str) -> dict:
    """YAML 프롬프트 파일 로드.

    Args:
        filename: 프롬프트 파일명 (prompts/ 디렉토리 기준)

    Returns:
        프롬프트 딕셔너리 (role, content/template 포함)
    """
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_path = prompts_dir / filename

    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_documents(documents: list[Document], max_docs: int = 5) -> str:
    """검색된 문서를 프롬프트용 문자열로 포맷.

    Args:
        documents: 검색된 문서 리스트
        max_docs: 최대 포함 문서 수

    Returns:
        포맷된 문서 문자열
    """
    if not documents:
        return "(검색된 문서 없음)"

    formatted = []
    for i, doc in enumerate(documents[:max_docs], 1):
        meta = doc.metadata
        source = meta.get("source_file", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        course = meta.get("course", "unknown")

        content = doc.page_content[:500]  # 500자 제한
        if len(doc.page_content) > 500:
            content += "..."

        formatted.append(
            f"[문서 {i}]\n"
            f"- 출처: {source}\n"
            f"- 유형: {doc_type}\n"
            f"- 과정: {course}\n"
            f"- 내용:\n{content}\n"
        )

    if len(documents) > max_docs:
        formatted.append(f"(... 외 {len(documents) - max_docs}개 문서 생략)")

    return "\n".join(formatted)


def extract_json_from_response(response_text: str) -> dict:
    """LLM 응답에서 JSON 추출.

    Args:
        response_text: LLM 응답 텍스트

    Returns:
        파싱된 JSON 딕셔너리

    Raises:
        ValueError: JSON 파싱 실패 시
    """
    # 1. 전체가 JSON인 경우
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` 블록 추출
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. { ... } 패턴 추출
    brace_match = re.search(r"\{[\s\S]*\}", response_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSON 추출 실패: {response_text[:200]}...")


class LLMJudgeEvaluator:
    """HCX-007 기반 LLM-as-Judge 평가기.

    RAG 시스템의 답변 품질을 LLM이 평가합니다.

    Attributes:
        llm: 평가에 사용할 LLM (기본: HCX-007)
        system_prompt: 시스템 프롬프트
        user_template: 사용자 프롬프트 템플릿

    Example:
        >>> evaluator = LLMJudgeEvaluator()
        >>> result = await evaluator.evaluate(question_data, answer, docs)
        >>> print(result.overall_score)
        0.8
    """

    def __init__(self, llm: BaseChatModel | None = None):
        """평가기 초기화.

        Args:
            llm: 평가용 LLM. None이면 기본 HCX-007 사용.
        """
        if llm is None:
            from naver_connect_chatbot.config.llm import get_chat_model
            self.llm = get_chat_model()
        else:
            self.llm = llm

        # 프롬프트 로드
        system_data = load_prompt("judge_system.yaml")
        user_data = load_prompt("judge_user.yaml")

        self.system_prompt = system_data["content"]
        self.user_template = user_data["template"]

    def _format_user_prompt(
        self,
        question_data: dict,
        answer: str,
        documents: list[Document],
    ) -> str:
        """사용자 프롬프트 포맷.

        Args:
            question_data: 질문 데이터 (데이터셋 항목)
            answer: RAG 시스템 답변
            documents: 검색된 문서 리스트

        Returns:
            포맷된 사용자 프롬프트
        """
        ground_truth = question_data.get("ground_truth", {})
        metadata = question_data.get("metadata", {})

        # 기대 행동 결정
        expected_behavior = ground_truth.get("expected_behavior", "provide_answer")
        if question_data["category"] == "in_domain":
            expected_behavior = "provide_answer"

        # 키워드 포맷
        keywords = ground_truth.get("answer_keywords", [])
        keywords_str = ", ".join(keywords) if keywords else "(없음)"

        return self.user_template.format(
            question=question_data["question"],
            category=question_data["category"],
            subcategory=question_data["subcategory"],
            expected_behavior=expected_behavior,
            difficulty=metadata.get("difficulty", "medium"),
            doc_count=len(documents),
            documents=format_documents(documents),
            answer=answer or "(답변 없음)",
            expected_keywords=keywords_str,
        )

    async def evaluate(
        self,
        question_data: dict,
        answer: str,
        documents: list[Document],
    ) -> JudgeEvaluation:
        """답변 품질 평가 실행.

        Args:
            question_data: 질문 데이터 (데이터셋 항목)
            answer: RAG 시스템 답변
            documents: 검색된 문서 리스트

        Returns:
            JudgeEvaluation 평가 결과

        Raises:
            ValueError: LLM 응답 파싱 실패 시
        """
        user_prompt = self._format_user_prompt(question_data, answer, documents)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # JSON 추출 및 파싱
        eval_data = extract_json_from_response(response_text)

        return JudgeEvaluation(**eval_data)

    def evaluate_sync(
        self,
        question_data: dict,
        answer: str,
        documents: list[Document],
    ) -> JudgeEvaluation:
        """동기 버전 평가 (테스트용).

        Args:
            question_data: 질문 데이터
            answer: RAG 시스템 답변
            documents: 검색된 문서 리스트

        Returns:
            JudgeEvaluation 평가 결과
        """
        import asyncio
        return asyncio.run(self.evaluate(question_data, answer, documents))
