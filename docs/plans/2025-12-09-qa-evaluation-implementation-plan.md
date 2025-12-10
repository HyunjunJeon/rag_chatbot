# Q&A 평가 데이터셋 확장 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** RAG Q&A 평가 데이터셋을 27개에서 80개로 확장하고 LLM-as-Judge 기반 평가 시스템 구축

**Architecture:** 기존 `tests/evaluation/` 구조를 확장하여 v2 데이터셋과 HCX-007 기반 LLM Judge 추가. Pydantic 스키마로 타입 안전성 확보, YAML 프롬프트로 유지보수성 향상.

**Tech Stack:** Python 3.11+, pytest-asyncio, Pydantic v2, LangChain, HyperClovaX HCX-007, YAML

---

## Task 1: 디렉토리 구조 생성

**Files:**
- Create: `tests/evaluation/config/`
- Create: `tests/evaluation/prompts/`
- Create: `tests/evaluation/evaluators/`

**Step 1: 디렉토리 생성**

```bash
mkdir -p tests/evaluation/config
mkdir -p tests/evaluation/prompts
mkdir -p tests/evaluation/evaluators
```

**Step 2: __init__.py 파일 생성**

```bash
touch tests/evaluation/evaluators/__init__.py
```

**Step 3: 확인**

```bash
ls -la tests/evaluation/
```

Expected:
```
config/
evaluators/
prompts/
reports/
qa_dataset.json
test_rag_evaluation.py
README.md
```

---

## Task 2: Pydantic 평가 스키마 정의

**Files:**
- Create: `tests/evaluation/evaluators/schemas.py`

**Step 1: 스키마 파일 작성**

```python
"""
LLM-as-Judge 평가 스키마 정의.

Pydantic v2 모델로 평가 결과의 타입 안전성을 보장합니다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeEvaluation(BaseModel):
    """LLM-as-Judge 평가 결과.

    HCX-007이 반환하는 구조화된 평가 결과입니다.
    """

    faithfulness: int = Field(
        ge=1, le=5,
        description="충실성 점수 (1-5): 검색 문서 기반 답변 여부"
    )
    relevance: int = Field(
        ge=1, le=5,
        description="관련성 점수 (1-5): 질문에 대한 답변 적절성"
    )
    completeness: int = Field(
        ge=1, le=5,
        description="완전성 점수 (1-5): 답변의 충분성"
    )
    hallucination_detected: bool = Field(
        description="환각 탐지 여부"
    )
    behavior_correct: bool = Field(
        description="기대 행동 수행 여부 (OOD/Edge Case용)"
    )
    reasoning: str = Field(
        description="평가 근거 (2-3문장)"
    )

    @property
    def overall_score(self) -> float:
        """종합 점수 (0-1 스케일).

        환각 탐지 시 0.2 페널티 적용.
        """
        base = (self.faithfulness + self.relevance + self.completeness) / 15
        penalty = 0.2 if self.hallucination_detected else 0
        return max(0.0, base - penalty)

    @property
    def is_passing(self) -> bool:
        """통과 여부 (overall_score >= 0.6)"""
        return self.overall_score >= 0.6 and not self.hallucination_detected


class QuestionEvaluation(BaseModel):
    """개별 질문 평가 결과.

    질문 메타데이터와 Judge 평가를 통합합니다.
    """

    question_id: str
    question: str
    category: str
    subcategory: str

    # RAG 결과
    answer: str = ""
    retrieved_docs_count: int = 0
    retrieval_time_ms: float = 0.0
    filter_applied: bool = False

    # Judge 평가
    judge_evaluation: JudgeEvaluation | None = None

    # 최종 결과
    passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        """딕셔너리 변환 (리포트용)"""
        result = {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "subcategory": self.subcategory,
            "answer": self.answer[:500] if self.answer else "",
            "retrieved_docs_count": self.retrieved_docs_count,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "filter_applied": self.filter_applied,
            "passed": self.passed,
            "error": self.error,
        }

        if self.judge_evaluation:
            result["judge"] = {
                "faithfulness": self.judge_evaluation.faithfulness,
                "relevance": self.judge_evaluation.relevance,
                "completeness": self.judge_evaluation.completeness,
                "hallucination_detected": self.judge_evaluation.hallucination_detected,
                "behavior_correct": self.judge_evaluation.behavior_correct,
                "overall_score": round(self.judge_evaluation.overall_score, 3),
                "reasoning": self.judge_evaluation.reasoning,
            }

        return result


class EvaluationReport(BaseModel):
    """전체 평가 리포트.

    카테고리별/과정별 통계를 집계합니다.
    """

    dataset_version: str = "2.0.0"
    timestamp: str = ""
    total_questions: int = 0
    passed_questions: int = 0

    # 카테고리별 통계
    by_category: dict[str, dict[str, int | float]] = Field(default_factory=dict)
    by_subcategory: dict[str, dict[str, int | float]] = Field(default_factory=dict)
    by_course: dict[str, dict[str, int | float]] = Field(default_factory=dict)

    # 세부 결과
    results: list[dict] = Field(default_factory=list)

    # 실패 질문 목록
    failed_questions: list[str] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """전체 통과율"""
        if self.total_questions == 0:
            return 0.0
        return self.passed_questions / self.total_questions

    @property
    def overall_score(self) -> float:
        """평균 overall_score"""
        scores = [
            r.get("judge", {}).get("overall_score", 0)
            for r in self.results
            if r.get("judge")
        ]
        return sum(scores) / len(scores) if scores else 0.0
```

**Step 2: 타입 체크 실행**

```bash
cd /Users/jhj/Desktop/personal/naver_connect_chatbot && uv run mypy tests/evaluation/evaluators/schemas.py --ignore-missing-imports
```

Expected: Success (no errors)

---

## Task 3: LLM-as-Judge 프롬프트 작성

**Files:**
- Create: `tests/evaluation/prompts/judge_system.yaml`
- Create: `tests/evaluation/prompts/judge_user.yaml`

**Step 1: 시스템 프롬프트 작성**

```yaml
# tests/evaluation/prompts/judge_system.yaml
# LLM-as-Judge 시스템 프롬프트
# HyperClovaX HCX-007 용

role: system
content: |
  당신은 네이버 부스트캠프 AI Tech 교육 챗봇의 RAG 시스템 평가 전문가입니다.

  ## 평가 기준

  ### 1. Faithfulness (충실성) - 1~5점
  검색된 문서에 기반한 답변인지 평가합니다.
  - 5점: 답변의 모든 정보가 검색 문서에서 직접 도출됨
  - 4점: 대부분의 정보가 문서 기반이며, 일부 합리적 추론 포함
  - 3점: 핵심 정보는 문서 기반이나, 일부 불확실한 내용 포함
  - 2점: 문서와 관련 있으나 상당 부분이 문서에 없는 내용
  - 1점: 문서와 무관하거나 명백한 환각(hallucination) 포함

  ### 2. Relevance (관련성) - 1~5점
  사용자 질문에 적절히 답변했는지 평가합니다.
  - 5점: 질문의 핵심을 정확히 파악하고 완벽히 답변
  - 4점: 질문에 잘 답변했으나 약간의 부가 정보 부족
  - 3점: 질문과 관련된 답변이나 핵심을 놓침
  - 2점: 질문과 부분적으로만 관련된 답변
  - 1점: 질문과 무관한 답변 또는 질문 오해

  ### 3. Completeness (완전성) - 1~5점
  답변이 충분히 상세한지 평가합니다.
  - 5점: 필요한 모든 정보를 포함한 완전한 답변
  - 4점: 대부분의 정보 포함, 사소한 세부사항 누락
  - 3점: 기본적인 답변이나 중요한 세부사항 누락
  - 2점: 불완전하여 추가 질문이 필요한 답변
  - 1점: 매우 불완전하거나 답변 거부

  ### 4. Hallucination Detection (환각 탐지)
  답변에 다음이 포함되면 환각으로 판정합니다:
  - 검색 문서에 없는 구체적인 수치, 날짜, 이름
  - 존재하지 않는 강의, 마스터, 커리큘럼 언급
  - 문서와 모순되는 정보
  - 근거 없는 단정적 주장

  ### 5. Behavior Correctness (행동 정확성)
  특수 케이스에서 시스템이 올바르게 행동했는지 평가합니다.

  #### Out-of-Domain 질문:
  - "politely_decline": 범위 외 질문임을 정중히 안내해야 함
  - "ask_clarification": 모호한 질문에 명확화 요청해야 함
  - "acknowledge_no_info": 정보 없음을 솔직히 인정해야 함
  - "acknowledge_limitation": 한계를 인정하며 가능한 정보 제공

  #### Edge Case 질문:
  - "multi_doc_synthesis": 여러 문서 정보를 종합해야 함
  - "temporal_reasoning": 시간/순서 관계를 올바르게 설명해야 함
  - "negation_handling": 제외 조건을 올바르게 처리해야 함
  - "code_explanation": 코드 동작을 정확히 설명해야 함
  - "meta_info_retrieval": 자료 메타정보를 정확히 제공해야 함

  ## 출력 형식
  반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:

  ```json
  {
    "faithfulness": <1-5 정수>,
    "relevance": <1-5 정수>,
    "completeness": <1-5 정수>,
    "hallucination_detected": <true 또는 false>,
    "behavior_correct": <true 또는 false>,
    "reasoning": "<평가 근거를 2-3문장 한국어로 설명>"
  }
  ```
```

**Step 2: 사용자 프롬프트 템플릿 작성**

```yaml
# tests/evaluation/prompts/judge_user.yaml
# LLM-as-Judge 사용자 프롬프트 템플릿

role: user
template: |
  ## 평가 대상

  ### 사용자 질문
  {question}

  ### 질문 메타데이터
  - 카테고리: {category}
  - 서브카테고리: {subcategory}
  - 기대 행동: {expected_behavior}
  - 난이도: {difficulty}

  ### 검색된 문서 (총 {doc_count}개)
  ---
  {documents}
  ---

  ### 시스템 답변
  {answer}

  ### 참고: 예상 키워드 (동의어/관련 표현도 인정)
  {expected_keywords}

  ---

  위 정보를 바탕으로 평가를 수행하고, 반드시 JSON 형식으로만 응답하세요.
```

**Step 3: 파일 확인**

```bash
ls -la tests/evaluation/prompts/
```

Expected:
```
judge_system.yaml
judge_user.yaml
```

---

## Task 4: LLM Judge 평가기 구현

**Files:**
- Create: `tests/evaluation/evaluators/llm_judge.py`

**Step 1: 평가기 클래스 작성**

```python
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
```

**Step 2: __init__.py 업데이트**

```python
# tests/evaluation/evaluators/__init__.py
"""RAG 평가기 모듈."""

from .schemas import JudgeEvaluation, QuestionEvaluation, EvaluationReport
from .llm_judge import LLMJudgeEvaluator, load_prompt, format_documents

__all__ = [
    "JudgeEvaluation",
    "QuestionEvaluation",
    "EvaluationReport",
    "LLMJudgeEvaluator",
    "load_prompt",
    "format_documents",
]
```

**Step 3: 타입 체크**

```bash
cd /Users/jhj/Desktop/personal/naver_connect_chatbot && uv run mypy tests/evaluation/evaluators/ --ignore-missing-imports
```

Expected: Success

---

## Task 5: 확장된 Q&A 데이터셋 생성 (80개)

**Files:**
- Create: `tests/evaluation/qa_dataset_v2.json`

**Step 1: 데이터셋 구조 생성**

(이 태스크는 별도 서브에이전트가 실제 80개 질문을 생성합니다)

데이터셋 구조:
```json
{
  "version": "2.0.0",
  "created_at": "2025-12-09",
  "description": "Naver Connect Boost Camp RAG 평가 데이터셋 v2",
  "statistics": {
    "total_questions": 80,
    "by_category": {
      "in_domain": 50,
      "out_of_domain": 15,
      "edge_case": 15
    }
  },
  "categories": {
    "in_domain": {...},
    "out_of_domain": {...},
    "edge_case": {...}
  },
  "questions": [...]
}
```

**질문 분배 목표:**
- In-Domain (50개): concept(10), implementation(10), troubleshooting(6), comparison(6), course_specific(10), source_specific(8)
- Out-of-Domain (15개): unrelated(4), ambiguous(4), hallucination_inducing(4), boundary(3)
- Edge Case (15개): multi_hop(4), temporal(3), negation(3), code_execution(2), meta_question(3)

**Step 2: 기존 27개 질문 마이그레이션**

기존 `qa_dataset.json`의 질문들을 v2 스키마에 맞게 변환하여 포함

**Step 3: 신규 53개 질문 생성**

과정별 커버리지 매핑에 따라 새 질문 추가

---

## Task 6: 테스트 코드 v2 작성

**Files:**
- Create: `tests/evaluation/test_rag_evaluation_v2.py`

**Step 1: 테스트 파일 작성**

```python
"""
RAG Q&A 평가 테스트 v2.

LLM-as-Judge 기반 평가 시스템으로 확장된 데이터셋을 테스트합니다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")

from tests.evaluation.evaluators import (
    JudgeEvaluation,
    QuestionEvaluation,
    EvaluationReport,
    LLMJudgeEvaluator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def qa_dataset_v2() -> dict:
    """Q&A 평가 데이터셋 v2 로드"""
    dataset_path = Path(__file__).parent / "qa_dataset_v2.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def llm_judge() -> LLMJudgeEvaluator:
    """LLM Judge 평가기"""
    return LLMJudgeEvaluator()


@pytest.fixture
def embeddings():
    """NaverEmbeddings 인스턴스 생성"""
    from naver_connect_chatbot.config.embedding import get_embeddings
    return get_embeddings()


@pytest.fixture
def hybrid_retriever(embeddings):
    """Hybrid Retriever 인스턴스 생성"""
    from naver_connect_chatbot.rag.retriever_factory import (
        build_dense_sparse_hybrid_from_saved,
    )
    from naver_connect_chatbot.rag.retriever.hybrid_retriever import HybridMethod

    bm25_path = PROJECT_ROOT / "sparse_index" / "unified_bm25"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "naver_connect_docs")

    return build_dense_sparse_hybrid_from_saved(
        bm25_index_path=str(bm25_path),
        embedding_model=embeddings,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        weights=[0.5, 0.5],
        k=10,
        method=HybridMethod.RRF,
        rrf_c=60,
    )


@pytest.fixture
def llm():
    """LLM 인스턴스 생성"""
    from naver_connect_chatbot.config.llm import get_chat_model
    try:
        return get_chat_model()
    except ValueError:
        pytest.skip("사용 가능한 LLM이 설정되지 않았습니다")


@pytest.fixture
def reasoning_llm():
    """Reasoning LLM 인스턴스 생성"""
    from naver_connect_chatbot.config.llm import get_chat_model
    try:
        return get_chat_model(
            model="HCX-007",
            use_reasoning=True,
            reasoning_effort="medium",
        )
    except ValueError:
        pytest.skip("사용 가능한 Reasoning LLM이 설정되지 않았습니다")


@pytest.fixture
def rag_graph(hybrid_retriever, llm, reasoning_llm):
    """RAG 그래프 생성"""
    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph
    return build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================


async def evaluate_single_question(
    question_data: dict,
    rag_graph: Any,
    evaluator: LLMJudgeEvaluator,
) -> QuestionEvaluation:
    """단일 질문 평가 실행.

    Args:
        question_data: 질문 데이터
        rag_graph: RAG 워크플로 그래프
        evaluator: LLM Judge 평가기

    Returns:
        QuestionEvaluation 평가 결과
    """
    import time

    result = QuestionEvaluation(
        question_id=question_data["id"],
        question=question_data["question"],
        category=question_data["category"],
        subcategory=question_data["subcategory"],
    )

    try:
        start = time.time()

        # RAG 실행
        rag_result = await rag_graph.ainvoke(
            {"question": question_data["question"], "max_retries": 1},
            config=RunnableConfig(run_name=f"eval_v2_{question_data['id']}"),
        )

        elapsed = (time.time() - start) * 1000
        result.retrieval_time_ms = elapsed

        # 결과 추출
        result.answer = rag_result.get("answer", "")
        result.retrieved_docs_count = len(rag_result.get("documents", []))
        result.filter_applied = rag_result.get("retrieval_filters_applied", False)

        # LLM Judge 평가
        documents = rag_result.get("documents", [])
        judge_result = await evaluator.evaluate(
            question_data,
            result.answer,
            documents,
        )
        result.judge_evaluation = judge_result

        # 통과 여부 결정
        if question_data["category"] == "in_domain":
            # In-Domain: overall_score >= 0.6 + 환각 없음
            result.passed = judge_result.is_passing
        else:
            # OOD/Edge Case: behavior_correct가 핵심
            result.passed = judge_result.behavior_correct

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# ============================================================================
# Tests - Dataset Coverage
# ============================================================================


def test_dataset_v2_coverage(qa_dataset_v2):
    """데이터셋 v2 커버리지 검증"""
    questions = qa_dataset_v2["questions"]

    # 총 질문 수 검증
    assert len(questions) >= 80, f"질문 수 부족: {len(questions)}/80"

    # 카테고리별 검증
    categories = {q["category"] for q in questions}
    assert {"in_domain", "out_of_domain", "edge_case"} == categories

    # In-Domain 서브카테고리
    in_domain = [q for q in questions if q["category"] == "in_domain"]
    in_domain_subcats = {q["subcategory"] for q in in_domain}
    expected_in = {"concept", "implementation", "troubleshooting", "comparison", "course_specific", "source_specific"}
    assert expected_in == in_domain_subcats, f"In-Domain 서브카테고리 누락: {expected_in - in_domain_subcats}"

    # Edge Case 서브카테고리
    edge_cases = [q for q in questions if q["category"] == "edge_case"]
    edge_subcats = {q["subcategory"] for q in edge_cases}
    expected_edge = {"multi_hop", "temporal", "negation", "code_execution", "meta_question"}
    assert expected_edge == edge_subcats, f"Edge Case 서브카테고리 누락: {expected_edge - edge_subcats}"

    print(f"\n✅ 데이터셋 v2 커버리지 검증 완료")
    print(f"   총 질문: {len(questions)}")
    print(f"   In-Domain: {len(in_domain)}")
    print(f"   Out-of-Domain: {len([q for q in questions if q['category'] == 'out_of_domain'])}")
    print(f"   Edge Case: {len(edge_cases)}")


# ============================================================================
# Tests - Category Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("subcategory", [
    "concept", "implementation", "troubleshooting",
    "comparison", "course_specific", "source_specific"
])
async def test_in_domain_by_subcategory(
    qa_dataset_v2, llm_judge, rag_graph, subcategory
):
    """In-Domain 서브카테고리별 테스트"""
    questions = [
        q for q in qa_dataset_v2["questions"]
        if q["category"] == "in_domain" and q["subcategory"] == subcategory
    ]

    if not questions:
        pytest.skip(f"No questions for subcategory: {subcategory}")

    print(f"\n{'='*60}")
    print(f"Testing In-Domain / {subcategory} ({len(questions)} questions)")
    print(f"{'='*60}")

    results = []
    for q in questions[:3]:  # 서브카테고리당 최대 3개 테스트 (비용 절감)
        result = await evaluate_single_question(q, rag_graph, llm_judge)
        results.append(result)

        status = "✅" if result.passed else "❌"
        score = result.judge_evaluation.overall_score if result.judge_evaluation else 0
        print(f"{status} {q['id']}: score={score:.2f}")

    # 통과율 계산
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    avg_score = sum(
        r.judge_evaluation.overall_score for r in results if r.judge_evaluation
    ) / len(results)

    print(f"\n{subcategory} 결과: pass_rate={pass_rate:.0%}, avg_score={avg_score:.2f}")

    # 카테고리별 최소 통과 기준: 50%
    assert pass_rate >= 0.5, f"{subcategory} 통과율 미달: {pass_rate:.0%}"


@pytest.mark.asyncio
@pytest.mark.parametrize("subcategory", [
    "multi_hop", "temporal", "negation", "code_execution", "meta_question"
])
async def test_edge_cases_by_subcategory(
    qa_dataset_v2, llm_judge, rag_graph, subcategory
):
    """Edge Case 서브카테고리별 테스트"""
    questions = [
        q for q in qa_dataset_v2["questions"]
        if q["category"] == "edge_case" and q["subcategory"] == subcategory
    ]

    if not questions:
        pytest.skip(f"No questions for subcategory: {subcategory}")

    print(f"\n{'='*60}")
    print(f"Testing Edge Case / {subcategory} ({len(questions)} questions)")
    print(f"{'='*60}")

    results = []
    for q in questions[:2]:  # 서브카테고리당 최대 2개 테스트
        result = await evaluate_single_question(q, rag_graph, llm_judge)
        results.append(result)

        correct = result.judge_evaluation.behavior_correct if result.judge_evaluation else False
        status = "✅" if correct else "⚠️"
        print(f"{status} {q['id']}: behavior_correct={correct}")

    # Edge Case는 behavior_correct 기준
    correct_rate = sum(
        1 for r in results
        if r.judge_evaluation and r.judge_evaluation.behavior_correct
    ) / len(results)

    print(f"\n{subcategory} behavior_correct rate: {correct_rate:.0%}")


# ============================================================================
# Tests - Full Evaluation
# ============================================================================


@pytest.mark.asyncio
async def test_full_evaluation_v2(qa_dataset_v2, llm_judge, rag_graph):
    """전체 평가 실행 및 리포트 생성"""
    print(f"\n{'='*80}")
    print("Full RAG Evaluation v2 with LLM-as-Judge")
    print(f"{'='*80}")

    report = EvaluationReport(
        dataset_version=qa_dataset_v2["version"],
        timestamp=datetime.now().isoformat(),
    )

    questions = qa_dataset_v2["questions"]

    for i, q_data in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Evaluating: {q_data['id']}")

        result = await evaluate_single_question(q_data, rag_graph, llm_judge)

        # 리포트 업데이트
        report.total_questions += 1
        if result.passed:
            report.passed_questions += 1
        else:
            report.failed_questions.append(q_data["id"])

        # 카테고리별 통계
        cat = result.category
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "passed": 0, "avg_score": 0.0}
        report.by_category[cat]["total"] += 1
        if result.passed:
            report.by_category[cat]["passed"] += 1

        # 서브카테고리별 통계
        subcat = f"{result.category}/{result.subcategory}"
        if subcat not in report.by_subcategory:
            report.by_subcategory[subcat] = {"total": 0, "passed": 0}
        report.by_subcategory[subcat]["total"] += 1
        if result.passed:
            report.by_subcategory[subcat]["passed"] += 1

        # 결과 추가
        report.results.append(result.to_dict())

        # 진행 상황 출력
        status = "✅" if result.passed else "❌"
        score = result.judge_evaluation.overall_score if result.judge_evaluation else 0
        print(f"  {status} score={score:.2f}, docs={result.retrieved_docs_count}")

    # 카테고리별 통과율 계산
    for cat in report.by_category:
        stats = report.by_category[cat]
        stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0

    # 리포트 저장
    report_path = Path(__file__).parent / "reports" / f"evaluation_report_v2_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = report.model_dump()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print(f"\n{'='*80}")
    print("Evaluation Report v2")
    print(f"{'='*80}")
    print(f"Total: {report.total_questions}")
    print(f"Passed: {report.passed_questions}")
    print(f"Pass Rate: {report.pass_rate:.1%}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"\nBy Category:")
    for cat, stats in report.by_category.items():
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")

    print(f"\nReport saved to: {report_path}")

    # 성공 기준 검증
    assert report.pass_rate >= 0.65, f"전체 통과율 미달: {report.pass_rate:.1%}"
    assert report.overall_score >= 0.6, f"평균 점수 미달: {report.overall_score:.2f}"


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

**Step 2: 데이터셋 커버리지 테스트 실행**

```bash
cd /Users/jhj/Desktop/personal/naver_connect_chatbot && uv run pytest tests/evaluation/test_rag_evaluation_v2.py::test_dataset_v2_coverage -v
```

Expected: PASS (데이터셋 구조 검증)

---

## Task 7: 통합 검증

**Step 1: 전체 파일 구조 확인**

```bash
ls -laR tests/evaluation/
```

Expected structure:
```
tests/evaluation/
├── config/
├── evaluators/
│   ├── __init__.py
│   ├── schemas.py
│   └── llm_judge.py
├── prompts/
│   ├── judge_system.yaml
│   └── judge_user.yaml
├── reports/
├── qa_dataset.json
├── qa_dataset_v2.json
├── test_rag_evaluation.py
├── test_rag_evaluation_v2.py
└── README.md
```

**Step 2: 타입 체크 전체 실행**

```bash
cd /Users/jhj/Desktop/personal/naver_connect_chatbot && uv run mypy tests/evaluation/ --ignore-missing-imports
```

Expected: Success

**Step 3: 샘플 평가 테스트 (In-Domain concept 3개)**

```bash
cd /Users/jhj/Desktop/personal/naver_connect_chatbot && uv run pytest tests/evaluation/test_rag_evaluation_v2.py::test_in_domain_by_subcategory[concept] -v -s
```

Expected: 테스트 실행 및 LLM Judge 평가 결과 출력

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | 디렉토리 구조 생성 | `evaluators/`, `prompts/`, `config/` |
| 2 | Pydantic 스키마 정의 | `evaluators/schemas.py` |
| 3 | Judge 프롬프트 작성 | `prompts/judge_*.yaml` |
| 4 | LLM Judge 평가기 구현 | `evaluators/llm_judge.py` |
| 5 | 80개 Q&A 데이터셋 생성 | `qa_dataset_v2.json` |
| 6 | 테스트 코드 v2 작성 | `test_rag_evaluation_v2.py` |
| 7 | 통합 검증 | All files |

**예상 소요 시간:** Task 1-4 (30분), Task 5 (60분), Task 6-7 (30분) = 총 2시간
