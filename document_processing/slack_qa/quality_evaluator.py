"""
LLM 기반 Q&A 품질 평가기.

Clova X HCX-007 모델과 Tool 기반 structured output을 사용합니다.
프로젝트의 표준 패턴(create_agent, parse_agent_response)을 따릅니다.
"""

from typing import Literal

from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from naver_connect_chatbot.config import get_chat_model, logger
from naver_connect_chatbot.service.agents.response_parser import parse_agent_response

from .quality_schemas import EvaluationInput, QualityEvaluation, DimensionScore

SYSTEM_PROMPT = '''# 역할
당신은 교육용 Q&A 데이터의 품질을 평가하는 전문가입니다.
네이버 부스트캠프 AI 교육 과정의 Slack Q&A 데이터를 평가합니다.

# 평가 목적
RAG 시스템에서 사용할 고품질 Q&A를 선별합니다.
좋은 Q&A: 비슷한 질문을 하는 학생에게 유용한 답변을 제공할 수 있어야 합니다.

# 평가 기준 (3가지)

## 1. 답변 완전성 (completeness)
| 점수 | 기준 |
|-----|------|
| 5 | 모든 부분에 완벽히 답변 + 추가 유용한 정보 |
| 4 | 모든 부분에 충분히 답변 |
| 3 | 핵심에는 답변, 일부 세부사항 누락 |
| 2 | 부분적으로만 답변, 중요 내용 누락 |
| 1 | 거의 답변 없거나 질문과 무관 |

## 2. 맥락 독립성 (context_independence)
| 점수 | 기준 |
|-----|------|
| 5 | 완전히 독립적, 배경 설명 포함 |
| 4 | 대부분 이해 가능 |
| 3 | 일부 외부 맥락 필요 |
| 2 | 상당한 외부 맥락 필요 ("그거", "아까") |
| 1 | 맥락 없이 이해 불가 |

## 3. 기술적 정확성 (technical_accuracy)
| 점수 | 기준 |
|-----|------|
| 5 | 완벽히 정확 + 모범 사례 |
| 4 | 정확함, 작동하는 솔루션 |
| 3 | 대체로 정확, 사소한 오류 가능 |
| 2 | 부분적으로 부정확 |
| 1 | 심각하게 부정확 |

# 종합 등급 (overall_quality)
- high: 평균 ≥ 4.0 AND 최저점 ≥ 3
- medium: 평균 ≥ 3.0 AND 최저점 ≥ 2
- low: 평균 ≥ 2.0 OR 최저점=1이지만 가치 있음
- remove: 평균 < 2.0 OR 2개 이상 차원이 1점

# 중요
1. 엄격하게 평가하세요
2. reasoning은 구체적으로 (1-2문장)
3. 한국어 비격식체는 감점 아님

IMPORTANT: 평가 완료 후 반드시 emit_quality_evaluation 도구를 호출하세요.
'''


def emit_quality_evaluation(
    completeness_score: int,
    completeness_reasoning: str,
    context_independence_score: int,
    context_independence_reasoning: str,
    technical_accuracy_score: int,
    technical_accuracy_reasoning: str,
    overall_quality: Literal["high", "medium", "low", "remove"],
    improvement_suggestion: str | None = None,
) -> QualityEvaluation:
    """
    Q&A 품질 평가 결과를 구조화된 형식으로 반환합니다.

    이 도구는 Q&A 평가 완료 후 반드시 호출해야 합니다.
    모든 파라미터를 정확히 채워서 호출하세요.

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
        overall_quality=overall_quality,
        improvement_suggestion=improvement_suggestion,
    )


def _create_evaluation_agent(llm: Runnable) -> Runnable:
    """품질 평가 Agent 생성."""
    emit_tool = StructuredTool.from_function(
        func=emit_quality_evaluation,
        name="emit_quality_evaluation",
        description="Q&A 품질 평가 결과를 구조화된 형식으로 반환",
    )

    agent = create_agent(
        model=llm,
        tools=[emit_tool],
        system_prompt=SYSTEM_PROMPT,
        name="qa_quality_evaluator",
    )

    return agent


class QualityEvaluator:
    """
    LLM 기반 Q&A 품질 평가기.

    Clova X HCX-007 모델을 사용하며, Tool 기반 structured output 패턴을 따릅니다.

    사용 예시:
        evaluator = QualityEvaluator()
        result = await evaluator.evaluate(evaluation_input)
    """

    DEFAULT_FALLBACK = QualityEvaluation(
        completeness=DimensionScore(score=1, reasoning="평가 실패"),
        context_independence=DimensionScore(score=1, reasoning="평가 실패"),
        technical_accuracy=DimensionScore(score=1, reasoning="평가 실패"),
        overall_quality="remove",
        improvement_suggestion="LLM 평가 중 오류 발생",
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
            self.llm = get_chat_model(temperature=temperature)
        else:
            self.llm = llm

        self._agent = None
        logger.info("QualityEvaluator initialized")

    @property
    def agent(self) -> Runnable:
        """평가 Agent (lazy initialization)."""
        if self._agent is None:
            self._agent = _create_evaluation_agent(self.llm)
        return self._agent

    async def evaluate(
        self,
        input_data: EvaluationInput,
        fallback: QualityEvaluation | None = None,
    ) -> QualityEvaluation:
        """
        단일 Q&A 쌍 평가.

        Args:
            input_data: 평가할 Q&A 입력
            fallback: 평가 실패 시 반환할 기본값 (None이면 DEFAULT_FALLBACK)

        Returns:
            QualityEvaluation: 평가 결과
        """
        if fallback is None:
            fallback = self.DEFAULT_FALLBACK

        try:
            qa_content = input_data.to_prompt_format()
            user_message = f"다음 Q&A를 평가해주세요.\n\n{qa_content}"

            response = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            result = parse_agent_response(
                response,
                QualityEvaluation,
                fallback=fallback,
            )

            logger.debug(
                f"Evaluated Q&A {input_data.original_id}: {result.overall_quality}"
            )
            return result

        except Exception as e:
            logger.error(f"Evaluation failed for {input_data.original_id}: {e}")
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
            user_message = f"다음 Q&A를 평가해주세요.\n\n{qa_content}"

            response = self.agent.invoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            return parse_agent_response(
                response,
                QualityEvaluation,
                fallback=fallback,
            )

        except Exception as e:
            logger.error(f"Sync evaluation failed: {e}")
            return fallback
