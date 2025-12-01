"""
LLM 기반 Q&A 품질 평가기.

Clova X HCX-007 모델을 사용합니다.
프롬프트 기반 JSON 응답을 파싱하는 방식으로 동작합니다.
"""

import asyncio
import json

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from naver_connect_chatbot.config import get_chat_model, logger

from .quality_schemas import EvaluationInput, QualityEvaluation, DimensionScore

SYSTEM_PROMPT = """# Role
You are an expert evaluator for educational Q&A data quality.
Evaluate Slack Q&A data from Naver Boostcamp AI training course.

# Purpose
Select high-quality Q&A pairs for RAG system.
Good Q&A: Should provide useful answers to students with similar questions.

# Evaluation Criteria (3 dimensions)

## 1. completeness
| Score | Criteria |
|-------|----------|
| 5 | Perfect answer + useful extra info |
| 4 | Fully answers all parts |
| 3 | Answers core, some details missing |
| 2 | Partial answer, important parts missing |
| 1 | No real answer or irrelevant |

## 2. context_independence
| Score | Criteria |
|-------|----------|
| 5 | Fully standalone with background |
| 4 | Mostly understandable |
| 3 | Some external context needed |
| 2 | Significant context needed ("that", "earlier") |
| 1 | Cannot understand without context |

## 3. technical_accuracy
| Score | Criteria |
|-------|----------|
| 5 | Perfectly accurate + best practices |
| 4 | Accurate, working solution |
| 3 | Mostly accurate, minor errors possible |
| 2 | Partially inaccurate |
| 1 | Seriously inaccurate |

# overall_quality
- high: avg >= 4.0 AND min >= 3
- medium: avg >= 3.0 AND min >= 2
- low: avg >= 2.0 OR min=1 but valuable
- remove: avg < 2.0 OR 2+ dimensions scored 1

# Important
1. Be strict
2. reasoning: MAX 5 words in English
3. Korean informal style is NOT a penalty

# Output Format
Respond with ONLY valid JSON. No other text.
reasoning must be MAX 5 words:
```json
{{
  "completeness": {{"score": 3, "reasoning": "core answer only"}},
  "context_independence": {{"score": 4, "reasoning": "mostly clear"}},
  "technical_accuracy": {{"score": 5, "reasoning": "accurate"}},
  "overall_quality": "high"
}}
```"""


def emit_quality_evaluation(
    completeness_score: int,
    completeness_reasoning: str,
    context_independence_score: int,
    context_independence_reasoning: str,
    technical_accuracy_score: int,
    technical_accuracy_reasoning: str,
    overall_quality: str,
    improvement_suggestion: str | None = None,
) -> QualityEvaluation:
    """
    Q&A 품질 평가 결과를 구조화된 형식으로 반환합니다.

    Args:
        completeness_score: 답변 완전성 점수 (1-5)
        completeness_reasoning: 완전성 점수의 근거
        context_independence_score: 맥락 독립성 점수 (1-5)
        context_independence_reasoning: 맥락 독립성 점수의 근거
        technical_accuracy_score: 기술적 정확성 점수 (1-5)
        technical_accuracy_reasoning: 기술적 정확성 점수의 근거
        overall_quality: 종합 등급 (high/medium/low/remove)
        improvement_suggestion: 품질 개선 제안 (선택사항)

    Returns:
        QualityEvaluation: 구조화된 평가 결과
    """
    return QualityEvaluation(
        completeness=DimensionScore(
            score=completeness_score,
            reasoning=completeness_reasoning,
        ),
        context_independence=DimensionScore(
            score=context_independence_score,
            reasoning=context_independence_reasoning,
        ),
        technical_accuracy=DimensionScore(
            score=technical_accuracy_score,
            reasoning=technical_accuracy_reasoning,
        ),
        overall_quality=overall_quality,  # type: ignore
        improvement_suggestion=improvement_suggestion,
    )


def _parse_json_response(response: str) -> dict | None:
    """LLM 응답에서 JSON 추출."""
    # JSON 블록 추출 시도
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()

    # JSON 파싱
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # 중괄호로 시작하는 부분 찾기
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # score 값을 정수로 변환 (LLM이 소수점을 반환할 수 있음)
    for key in ["completeness", "context_independence", "technical_accuracy"]:
        if key in data and isinstance(data[key], dict) and "score" in data[key]:
            data[key]["score"] = int(round(data[key]["score"]))

    return data


def _create_evaluation_chain(llm: Runnable) -> Runnable:
    """품질 평가 Chain 생성 (프롬프트 기반 JSON 응답)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "다음 Q&A를 평가해주세요.\n\n{qa_content}"),
    ])

    # Clova X는 structured output을 지원하지 않으므로 문자열로 받아서 파싱
    return prompt | llm | StrOutputParser()


class QualityEvaluator:
    """
    LLM 기반 Q&A 품질 평가기.

    Clova X HCX-007 모델을 사용하며, with_structured_output 패턴을 따릅니다.

    사용 예시:
        evaluator = QualityEvaluator()
        result = await evaluator.evaluate(evaluation_input)
    """

    DEFAULT_FALLBACK = QualityEvaluation(
        completeness=DimensionScore(score=1, reasoning="eval failed"),
        context_independence=DimensionScore(score=1, reasoning="eval failed"),
        technical_accuracy=DimensionScore(score=1, reasoning="eval failed"),
        overall_quality="remove",
        improvement_suggestion="LLM evaluation error",
    )

    def __init__(
        self,
        llm: Runnable | None = None,
        temperature: float = 0.0,
    ):
        """
        평가기 초기화.

        Args:
            llm: 사용할 LLM 인스턴스 (None이면 Clova X HCX-007 사용)
            temperature: 생성 온도 (일관성을 위해 0 권장)
        """
        if llm is None:
            # max_tokens를 충분히 설정하여 JSON 응답이 잘리지 않도록 함
            # 한국어는 영어보다 더 많은 토큰을 사용하므로 2000으로 넉넉히 설정
            self.llm = get_chat_model(temperature=temperature, max_tokens=4000)
        else:
            self.llm = llm

        self._chain = None
        logger.info("QualityEvaluator initialized")

    @property
    def chain(self) -> Runnable:
        """평가 Chain (lazy initialization)."""
        if self._chain is None:
            self._chain = _create_evaluation_chain(self.llm)
        return self._chain

    async def evaluate(
        self,
        input_data: EvaluationInput,
        fallback: QualityEvaluation | None = None,
        max_retries: int = 3,
    ) -> QualityEvaluation:
        """
        단일 Q&A 쌍 평가 (빈 응답 시 자동 재시도).

        Args:
            input_data: 평가할 Q&A 입력
            fallback: 평가 실패 시 반환할 기본값 (None이면 DEFAULT_FALLBACK)
            max_retries: 빈 응답 시 최대 재시도 횟수 (기본 3)

        Returns:
            QualityEvaluation: 평가 결과
        """
        if fallback is None:
            fallback = self.DEFAULT_FALLBACK

        qa_content = input_data.to_prompt_format()

        for attempt in range(max_retries):
            try:
                response = await self.chain.ainvoke({"qa_content": qa_content})

                # 문자열 응답에서 JSON 파싱
                parsed = _parse_json_response(response)

                if parsed is not None:
                    # QualityEvaluation 모델로 변환
                    result = QualityEvaluation.model_validate(parsed)
                    logger.debug(f"Evaluated Q&A {input_data.original_id}: {result.overall_quality}")
                    return result

                # 파싱 실패 - 재시도 또는 fallback
                if attempt < max_retries - 1:
                    wait_time = 2.0 * (attempt + 1)
                    logger.warning(
                        f"Empty/invalid response for {input_data.original_id}, "
                        f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(
                        f"Failed to parse JSON after {max_retries} attempts: {response if response else '(empty)'}"
                    )

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2.0 * (attempt + 1)
                    logger.warning(
                        f"Error for {input_data.original_id}: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Evaluation failed for {input_data.original_id} after {max_retries} attempts: {e}")

        return fallback

    def evaluate_sync(
        self,
        input_data: EvaluationInput,
        fallback: QualityEvaluation | None = None,
    ) -> QualityEvaluation:
        """
        동기 방식 평가 (테스트/디버깅용).

        Args:
            input_data: 평가할 Q&A 입력
            fallback: 평가 실패 시 반환할 기본값

        Returns:
            QualityEvaluation: 평가 결과
        """
        if fallback is None:
            fallback = self.DEFAULT_FALLBACK

        try:
            qa_content = input_data.to_prompt_format()

            response = self.chain.invoke({"qa_content": qa_content})

            # 문자열 응답에서 JSON 파싱
            parsed = _parse_json_response(response)
            if parsed is None:
                logger.warning(f"Failed to parse JSON from response: {response}")
                return fallback

            return QualityEvaluation.model_validate(parsed)

        except Exception as e:
            logger.error(f"Sync evaluation failed: {e}")
            return fallback
