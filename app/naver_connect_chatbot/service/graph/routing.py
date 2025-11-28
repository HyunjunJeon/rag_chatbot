"""
Adaptive RAG 워크플로의 라우팅 로직.

상태 조건에 따라 다음 노드를 결정하는 조건부 라우팅 함수를 제공합니다.
"""

from typing import Literal
from naver_connect_chatbot.service.graph.state import AdaptiveRAGState
from naver_connect_chatbot.config import logger


def check_document_sufficiency(
    state: AdaptiveRAGState,
) -> Literal["generate_answer", "refine_query", "generate_best_effort"]:
    """
    검색된 문서가 충분한지 확인합니다.

    다음 중 어떤 경로로 진행할지 결정합니다.
    - 현재 문서로 답변 생성
    - 질의를 정제 후 재검색
    - 재시도 한계를 넘긴 경우 최선의 답변 생성

    매개변수:
        state: 현재 워크플로 상태

    반환값:
        다음 노드 이름
    """
    sufficient = state.get("sufficient_context", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    logger.debug(
        f"Checking document sufficiency: sufficient={sufficient}, "
        f"retry_count={retry_count}, max_retries={max_retries}"
    )

    if sufficient:
        logger.info("Documents are sufficient, proceeding to answer generation")
        return "generate_answer"
    elif retry_count >= max_retries:
        logger.warning(
            f"Max retrieval retries ({max_retries}) exceeded, generating best-effort answer"
        )
        return "generate_best_effort"
    else:
        logger.info(
            f"Documents insufficient (retry {retry_count + 1}/{max_retries}), refining query"
        )
        return "refine_query"
