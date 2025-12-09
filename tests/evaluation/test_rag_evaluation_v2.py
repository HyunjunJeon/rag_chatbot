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
sys.path.insert(0, str(PROJECT_ROOT))

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
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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
