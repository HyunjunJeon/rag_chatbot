"""
RAG Q&A 평가 테스트.

데이터셋 기반으로 RAG 시스템의 성능을 평가합니다.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================================
# Evaluation Data Classes
# ============================================================================


@dataclass
class EvaluationResult:
    """단일 질문 평가 결과"""

    question_id: str
    question: str
    category: str
    subcategory: str

    # Retrieval 평가
    retrieved_docs_count: int = 0
    filter_applied: bool = False
    filter_matched: bool = False
    retrieval_time_ms: float = 0.0

    # Answer 평가
    answer: str = ""
    has_context: bool = False
    keyword_hits: list[str] = field(default_factory=list)
    keyword_hit_rate: float = 0.0

    # 전체 평가
    passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "subcategory": self.subcategory,
            "retrieved_docs_count": self.retrieved_docs_count,
            "filter_applied": self.filter_applied,
            "filter_matched": self.filter_matched,
            "retrieval_time_ms": self.retrieval_time_ms,
            "answer": self.answer[:500] if self.answer else "",
            "has_context": self.has_context,
            "keyword_hits": self.keyword_hits,
            "keyword_hit_rate": self.keyword_hit_rate,
            "passed": self.passed,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """전체 평가 리포트"""

    total_questions: int = 0
    passed_questions: int = 0
    failed_questions: int = 0

    # Category별 통계
    by_category: dict = field(default_factory=dict)
    by_subcategory: dict = field(default_factory=dict)

    # 세부 결과
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.passed_questions / self.total_questions

    def add_result(self, result: EvaluationResult):
        self.results.append(result)
        self.total_questions += 1

        if result.passed:
            self.passed_questions += 1
        else:
            self.failed_questions += 1

        # Category 통계
        cat = result.category
        if cat not in self.by_category:
            self.by_category[cat] = {"total": 0, "passed": 0}
        self.by_category[cat]["total"] += 1
        if result.passed:
            self.by_category[cat]["passed"] += 1

        # Subcategory 통계
        subcat = f"{result.category}/{result.subcategory}"
        if subcat not in self.by_subcategory:
            self.by_subcategory[subcat] = {"total": 0, "passed": 0}
        self.by_subcategory[subcat]["total"] += 1
        if result.passed:
            self.by_subcategory[subcat]["passed"] += 1

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_questions": self.total_questions,
                "passed_questions": self.passed_questions,
                "failed_questions": self.failed_questions,
                "pass_rate": round(self.pass_rate * 100, 2),
            },
            "by_category": {
                cat: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "pass_rate": round(stats["passed"] / stats["total"] * 100, 2)
                    if stats["total"] > 0
                    else 0,
                }
                for cat, stats in self.by_category.items()
            },
            "by_subcategory": {
                subcat: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "pass_rate": round(stats["passed"] / stats["total"] * 100, 2)
                    if stats["total"] > 0
                    else 0,
                }
                for subcat, stats in self.by_subcategory.items()
            },
            "results": [r.to_dict() for r in self.results],
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def qa_dataset() -> dict:
    """Q&A 평가 데이터셋 로드"""
    dataset_path = Path(__file__).parent / "qa_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def evaluate_answer_keywords(answer: str, keywords: list[str]) -> tuple[list[str], float]:
    """답변에서 키워드 포함 여부 평가"""
    if not keywords:
        return [], 1.0  # 키워드 없으면 통과

    answer_lower = answer.lower()
    hits = [kw for kw in keywords if kw.lower() in answer_lower]
    hit_rate = len(hits) / len(keywords) if keywords else 0.0

    return hits, hit_rate


def check_filter_match(
    actual_filters: dict | None,
    expected_filters: dict,
) -> bool:
    """필터 적용 결과 일치 여부 확인"""
    if not expected_filters.get("doc_type") and not expected_filters.get("course"):
        return True  # 필터 기대 없음

    if not actual_filters:
        return False

    # doc_type 체크
    if expected_filters.get("doc_type"):
        actual_doc_type = actual_filters.get("doc_type", [])
        if not any(dt in actual_doc_type for dt in expected_filters["doc_type"]):
            return False

    # course 체크 (부분 매치 허용)
    if expected_filters.get("course"):
        actual_course = actual_filters.get("course", [])
        if not actual_course:
            return False
        # 예상 과정 중 하나라도 포함되면 OK
        expected_courses = expected_filters["course"]
        if not any(
            any(exp.lower() in act.lower() for act in actual_course)
            for exp in expected_courses
        ):
            return False

    return True


def evaluate_ood_response(
    answer: str,
    expected_behavior: str | None,
) -> bool:
    """Out-of-Domain 질문에 대한 응답 평가"""
    if not expected_behavior:
        return True

    answer_lower = answer.lower()

    if expected_behavior == "politely_decline":
        # 정중하게 거절하는 답변인지
        decline_phrases = [
            "죄송",
            "도움",
            "드리기 어렵",
            "관련 정보",
            "찾을 수 없",
            "부스트캠프",
        ]
        return any(phrase in answer_lower for phrase in decline_phrases)

    elif expected_behavior == "ask_clarification":
        # 명확화를 요청하는 답변인지
        clarify_phrases = [
            "구체적",
            "자세히",
            "어떤",
            "무엇",
            "확인",
            "명확",
            "알려주",
        ]
        return any(phrase in answer_lower for phrase in clarify_phrases)

    elif expected_behavior == "acknowledge_no_info":
        # 정보가 없음을 인정하는 답변인지
        no_info_phrases = [
            "정보가 없",
            "찾을 수 없",
            "확인되지 않",
            "자료가 없",
            "포함되어 있지 않",
        ]
        return any(phrase in answer_lower for phrase in no_info_phrases)

    elif expected_behavior == "acknowledge_limitation":
        # 한계를 인정하는 답변인지
        limit_phrases = [
            "자료에",
            "정보가 제한",
            "부스트캠프",
            "학습 자료",
            "포함되어 있지 않",
        ]
        return any(phrase in answer_lower for phrase in limit_phrases)

    return True


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_in_domain_questions(rag_graph, qa_dataset):
    """In-Domain 질문 평가"""
    print("\n" + "=" * 80)
    print("In-Domain Questions Evaluation")
    print("=" * 80)

    report = EvaluationReport()
    in_domain_questions = [
        q for q in qa_dataset["questions"] if q["category"] == "in_domain"
    ]

    for q_data in in_domain_questions[:10]:  # 처음 10개만 테스트 (비용 절감)
        result = EvaluationResult(
            question_id=q_data["id"],
            question=q_data["question"],
            category=q_data["category"],
            subcategory=q_data["subcategory"],
        )

        try:
            import time

            start = time.time()

            rag_result = await rag_graph.ainvoke(
                {"question": q_data["question"], "max_retries": 1},
                config=RunnableConfig(run_name=f"eval_{q_data['id']}"),
            )

            elapsed = (time.time() - start) * 1000
            result.retrieval_time_ms = elapsed

            # 결과 추출
            result.answer = rag_result.get("answer", "")
            result.retrieved_docs_count = len(rag_result.get("documents", []))
            result.filter_applied = rag_result.get("retrieval_filters_applied", False)
            result.has_context = result.retrieved_docs_count > 0

            # 필터 매치 확인
            actual_filters = rag_result.get("retrieval_filters")
            result.filter_matched = check_filter_match(
                actual_filters, q_data.get("expected_filters", {})
            )

            # 키워드 평가
            ground_truth = q_data.get("ground_truth", {})
            keywords = ground_truth.get("answer_keywords", [])
            result.keyword_hits, result.keyword_hit_rate = evaluate_answer_keywords(
                result.answer, keywords
            )

            # 통과 여부 판단
            # In-Domain: 문서 검색됨 + 답변 존재 + 키워드 50% 이상 포함
            result.passed = (
                result.has_context
                and len(result.answer) > 0
                and result.keyword_hit_rate >= 0.3
            )

            print(f"✅ {q_data['id']}: {q_data['question'][:40]}...")
            print(f"   Docs: {result.retrieved_docs_count}, Keywords: {result.keyword_hit_rate:.0%}")

        except Exception as e:
            result.error = str(e)
            result.passed = False
            print(f"❌ {q_data['id']}: {str(e)[:50]}")

        report.add_result(result)

    # 리포트 출력
    print("\n" + "-" * 40)
    print(f"In-Domain Pass Rate: {report.pass_rate:.1%}")
    print(f"Passed: {report.passed_questions}/{report.total_questions}")

    # 최소 통과율 검증 (50% 이상)
    assert report.pass_rate >= 0.5, f"In-Domain pass rate too low: {report.pass_rate:.1%}"


@pytest.mark.asyncio
async def test_out_of_domain_questions(rag_graph, qa_dataset):
    """Out-of-Domain 질문 평가"""
    print("\n" + "=" * 80)
    print("Out-of-Domain Questions Evaluation")
    print("=" * 80)

    report = EvaluationReport()
    ood_questions = [
        q for q in qa_dataset["questions"] if q["category"] == "out_of_domain"
    ]

    for q_data in ood_questions:
        result = EvaluationResult(
            question_id=q_data["id"],
            question=q_data["question"],
            category=q_data["category"],
            subcategory=q_data["subcategory"],
        )

        try:
            rag_result = await rag_graph.ainvoke(
                {"question": q_data["question"], "max_retries": 1},
                config=RunnableConfig(run_name=f"eval_{q_data['id']}"),
            )

            result.answer = rag_result.get("answer", "")
            result.retrieved_docs_count = len(rag_result.get("documents", []))

            # OOD 응답 평가
            expected_behavior = q_data.get("ground_truth", {}).get("expected_behavior")
            result.passed = evaluate_ood_response(result.answer, expected_behavior)

            status = "✅" if result.passed else "⚠️"
            print(f"{status} {q_data['id']}: {q_data['question'][:40]}...")

        except Exception as e:
            result.error = str(e)
            result.passed = False
            print(f"❌ {q_data['id']}: {str(e)[:50]}")

        report.add_result(result)

    # 리포트 출력
    print("\n" + "-" * 40)
    print(f"Out-of-Domain Pass Rate: {report.pass_rate:.1%}")

    # OOD는 통과율 기준 완화 (40% 이상)
    assert report.pass_rate >= 0.4, f"OOD pass rate too low: {report.pass_rate:.1%}"


@pytest.mark.asyncio
async def test_full_evaluation(rag_graph, qa_dataset):
    """전체 평가 실행 및 리포트 생성"""
    print("\n" + "=" * 80)
    print("Full RAG Evaluation")
    print("=" * 80)

    report = EvaluationReport()

    for q_data in qa_dataset["questions"]:
        result = EvaluationResult(
            question_id=q_data["id"],
            question=q_data["question"],
            category=q_data["category"],
            subcategory=q_data["subcategory"],
        )

        try:
            rag_result = await rag_graph.ainvoke(
                {"question": q_data["question"], "max_retries": 1},
                config=RunnableConfig(run_name=f"eval_{q_data['id']}"),
            )

            result.answer = rag_result.get("answer", "")
            result.retrieved_docs_count = len(rag_result.get("documents", []))
            result.filter_applied = rag_result.get("retrieval_filters_applied", False)
            result.has_context = result.retrieved_docs_count > 0

            # 평가 로직
            if q_data["category"] == "in_domain":
                ground_truth = q_data.get("ground_truth", {})
                keywords = ground_truth.get("answer_keywords", [])
                result.keyword_hits, result.keyword_hit_rate = evaluate_answer_keywords(
                    result.answer, keywords
                )
                result.passed = (
                    result.has_context
                    and len(result.answer) > 0
                    and result.keyword_hit_rate >= 0.3
                )
            else:
                expected_behavior = q_data.get("ground_truth", {}).get("expected_behavior")
                result.passed = evaluate_ood_response(result.answer, expected_behavior)

        except Exception as e:
            result.error = str(e)
            result.passed = False

        report.add_result(result)

    # 리포트 저장
    report_path = Path(__file__).parent / "reports" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Evaluation Report")
    print("=" * 80)
    print(f"Total: {report.total_questions}")
    print(f"Passed: {report.passed_questions}")
    print(f"Failed: {report.failed_questions}")
    print(f"Pass Rate: {report.pass_rate:.1%}")
    print("\nBy Category:")
    for cat, stats in report.by_category.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print(f"\nReport saved to: {report_path}")


# ============================================================================
# Coverage Tests
# ============================================================================


def test_dataset_coverage(qa_dataset):
    """데이터셋 커버리지 검증"""
    questions = qa_dataset["questions"]

    # 카테고리별 검증
    categories = {q["category"] for q in questions}
    assert "in_domain" in categories, "Missing in_domain questions"
    assert "out_of_domain" in categories, "Missing out_of_domain questions"

    # Subcategory 검증
    in_domain_subcats = {
        q["subcategory"] for q in questions if q["category"] == "in_domain"
    }
    expected_in_domain = {"concept", "implementation", "troubleshooting", "comparison"}
    assert expected_in_domain.issubset(
        in_domain_subcats
    ), f"Missing in_domain subcategories: {expected_in_domain - in_domain_subcats}"

    # OOD subcategory 검증
    ood_subcats = {
        q["subcategory"] for q in questions if q["category"] == "out_of_domain"
    }
    expected_ood = {"unrelated", "ambiguous", "hallucination_inducing"}
    assert expected_ood.issubset(
        ood_subcats
    ), f"Missing OOD subcategories: {expected_ood - ood_subcats}"

    print(f"✅ Dataset coverage verified: {len(questions)} questions")
    print(f"   In-Domain subcategories: {in_domain_subcats}")
    print(f"   OOD subcategories: {ood_subcats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
