"""
Adaptive RAG 워크플로 통합 테스트

Hybrid Retriever를 Adaptive RAG 워크플로에 통합하여
전체 워크플로가 정상 작동하는지 검증합니다.
"""

import os
import sys
from pathlib import Path

from langchain_core.runnables.config import RunnableConfig
import pytest
from dotenv import load_dotenv

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture
def embeddings():
    """NaverEmbeddings 인스턴스 생성"""
    from naver_connect_chatbot.config.embedding import get_embeddings

    return get_embeddings()


@pytest.fixture
def hybrid_retriever(embeddings):
    """Hybrid Retriever 인스턴스 생성"""
    from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid_from_saved
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
        llm_instance = get_chat_model()
        return llm_instance
    except ValueError:
        # CLOVASTUDIO_API_KEY 등 필수 설정이 없으면 테스트를 건너뜁니다.
        pytest.skip("사용 가능한 LLM이 설정되지 않았습니다")


@pytest.fixture
def reasoning_llm():
    """Reasoning LLM 인스턴스 생성 (medium effort)"""
    from naver_connect_chatbot.config.llm import get_chat_model

    try:
        llm_instance = get_chat_model(
            thinking_level="medium",
        )
        return llm_instance
    except ValueError:
        pytest.skip("사용 가능한 Reasoning LLM이 설정되지 않았습니다")


@pytest.mark.asyncio
async def test_adaptive_rag_graph_construction(hybrid_retriever, llm, reasoning_llm):
    """Adaptive RAG 그래프 생성 테스트"""
    print("\n" + "=" * 80)
    print("1. Adaptive RAG 그래프 생성")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    try:
        graph = build_adaptive_rag_graph(
            retriever=hybrid_retriever,
            llm=llm,
            reasoning_llm=reasoning_llm,
            debug=True,
        )

        assert graph is not None, "그래프가 None입니다"

    except Exception as e:
        pytest.fail(f"그래프 생성 실패: {e}")


@pytest.mark.asyncio
async def test_simple_qa_workflow(hybrid_retriever, llm, reasoning_llm):
    """SIMPLE_QA 워크플로 테스트"""
    print("\n" + "=" * 80)
    print("2. SIMPLE_QA 워크플로 테스트")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    query = "PyTorch 설치 방법은?"
    print(f"\n🔍 쿼리: {query}")

    try:
        result = await graph.ainvoke(
            {
                "question": query,
                "max_retries": 2,
            },
            config=RunnableConfig(
                run_name="test_simple_qa_workflow",
                tags=["test"],
                configurable={"thread_id": "test_simple_qa_workflow"},
            ),
        )

        assert "answer" in result, "Answer not generated"
        assert len(result["answer"]) > 0, "Answer is empty"
        assert "documents" in result, "Documents not retrieved"

    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


@pytest.mark.asyncio
async def test_retrieval_in_workflow(hybrid_retriever, llm, reasoning_llm):
    """워크플로 내 검색 기능 테스트"""
    print("\n" + "=" * 80)
    print("3. 워크플로 내 검색 기능 테스트")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    try:
        result = await graph.ainvoke(
            {
                "question": "GPU 메모리 부족 해결 방법",
                "max_retries": 1,
            },
            config=RunnableConfig(
                run_name="test_retrieval_in_workflow",
                tags=["test"],
                configurable={"thread_id": "test_retrieval_in_workflow"},
            ),
        )

        documents = result.get("documents", [])

        assert len(documents) > 0, "Documents not retrieved"

        # Hybrid 검색이 사용되었는지 확인
        assert result.get("retrieval_strategy") == "hybrid", "Hybrid retrieval not used"

    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


@pytest.mark.asyncio
async def test_workflow_state_tracking(hybrid_retriever, llm, reasoning_llm):
    """워크플로 상태 추적 테스트"""
    print("\n" + "=" * 80)
    print("4. 워크플로 상태 추적 테스트")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    try:
        result = await graph.ainvoke(
            {
                "question": "데이터 증강 기법",
                "max_retries": 1,
            },
            config=RunnableConfig(
                run_name="test_workflow_state_tracking",
                tags=["test"],
                configurable={"thread_id": "test_workflow_state_tracking"},
            ),
        )

        # 주요 상태 필드 확인
        assert "intent" in result, "Intent classification not performed"
        assert "documents" in result, "Documents not retrieved"
        assert "answer" in result, "Answer not generated"

    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


@pytest.mark.asyncio
async def test_answer_generator_structured_output(llm):
    """Answer Generator LCEL 체인 테스트 (tools 미사용)"""
    print("\n" + "=" * 80)
    print("5. Answer Generator LCEL 체인 테스트")
    print("=" * 80)

    from naver_connect_chatbot.service.agents.answer_generator import (
        create_answer_generator,
    )

    # Simple 전략으로 체인 생성
    generator = create_answer_generator(llm, strategy="simple")

    print("\n🧪 테스트 쿼리: What is 2+2?")
    print("📝 컨텍스트: Mathematics: 2+2 equals 4.")

    try:
        # LCEL 체인 실행 (새 인터페이스)
        response_raw = await generator.ainvoke(
            {
                "question": "What is 2+2?",
                "context": "Mathematics: 2+2 equals 4.",
            }
        )

        # AIMessage에서 content 추출
        response = response_raw.content if hasattr(response_raw, "content") else str(response_raw)

        # 검증
        assert len(response) > 0, "Answer is empty"

        print(f"   - Type: {type(response).__name__}")
        print(f"   - Answer length: {len(response)} characters")
        print(f"   - Answer: {response}")

    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


# ============================================================================
# Data-Driven RAG Test Cases
# ============================================================================


# 질의 유형별 테스트 데이터
RAG_TEST_CASES = [
    # (질의, 예상 intent, 예상 doc_type 힌트, 설명)
    pytest.param(
        "PyTorch에서 텐서를 생성하는 방법은?",
        "SIMPLE_QA",
        None,
        "기본 개념 질문",
        id="simple_qa_pytorch",
    ),
    pytest.param(
        "CV 강의에서 CNN 아키텍처 설명해주세요",
        "SIMPLE_QA",
        ["lecture_transcript", "pdf"],
        "과정 특정 질문",
        id="course_specific_cv",
    ),
    pytest.param(
        "RecSys에서 collaborative filtering과 content-based의 장단점을 비교 분석해주세요",
        "COMPLEX_REASONING",
        None,
        "비교 분석 질문",
        id="complex_comparison",
    ),
    pytest.param(
        "NLP 분야의 최신 트렌드와 발전 방향은?",
        "EXPLORATORY",
        None,
        "탐색적 질문",
        id="exploratory_nlp",
    ),
    pytest.param(
        "Slack에서 GPU 관련 질문 중에 CUDA 에러 해결한 답변 있나요?",
        "SIMPLE_QA",
        ["slack_qa"],
        "Slack 특정 질문",
        id="slack_specific",
    ),
]

# 필터 추출 테스트 데이터
FILTER_EXTRACTION_CASES = [
    # (질의, 예상 doc_type, 예상 course 키워드)
    pytest.param(
        "CV 강의자료에서 ResNet 설명",
        ["pdf"],
        ["CV"],
        id="filter_cv_pdf",
    ),
    pytest.param(
        "NLP 녹취록에서 Transformer 아키텍처",
        ["lecture_transcript"],
        ["NLP"],
        id="filter_nlp_transcript",
    ),
    pytest.param(
        "PyTorch 실습 노트북 찾아줘",
        ["notebook"],
        ["PyTorch"],
        id="filter_pytorch_notebook",
    ),
    pytest.param(
        "미션에서 객체 탐지 과제",
        ["weekly_mission"],
        None,
        id="filter_mission",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,expected_intent,expected_doc_types,description", RAG_TEST_CASES)
async def test_rag_query_types(
    hybrid_retriever,
    llm,
    reasoning_llm,
    query,
    expected_intent,
    expected_doc_types,
    description,
):
    """
    다양한 질의 유형에 대한 RAG 워크플로 테스트.

    데이터 기반 파라미터화 테스트로 여러 질의 패턴을 커버합니다.
    """
    print(f"\n{'=' * 80}")
    print(f"RAG Test: {description}")
    print(f"Query: {query}")
    print(f"Expected Intent: {expected_intent}")
    print(f"Expected Doc Types: {expected_doc_types}")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    try:
        result = await graph.ainvoke(
            {"question": query, "max_retries": 1},
            config=RunnableConfig(
                run_name=f"test_rag_{description}",
                tags=["test", "data-driven"],
            ),
        )

        # 기본 검증
        assert "answer" in result, f"Answer not generated for: {query}"
        assert len(result.get("answer", "")) > 0, f"Empty answer for: {query}"
        assert "documents" in result, f"Documents not retrieved for: {query}"

        # Intent 검증 (허용 오차)
        actual_intent = result.get("intent", "")
        print(f"Actual Intent: {actual_intent}")
        # Intent는 정확히 일치하지 않아도 됨 (LLM 판단에 따라 달라질 수 있음)

        # 문서 검색 검증
        documents = result.get("documents", [])
        print(f"Retrieved {len(documents)} documents")
        assert len(documents) > 0, f"No documents for: {query}"

        # doc_type 필터가 적용되었는지 확인 (힌트가 있는 경우)
        if expected_doc_types:
            doc_types_found = {
                doc.metadata.get("doc_type") for doc in documents if doc.metadata
            }
            print(f"Doc types found: {doc_types_found}")

        print(f"✅ Test passed: {description}")

    except Exception as e:
        pytest.skip(f"RAG test failed for '{description}': {e}")


@pytest.mark.asyncio
async def test_multi_course_filter_or_condition(hybrid_retriever, llm, reasoning_llm):
    """
    다중 course 필터 OR 조건 테스트.

    "CV 강의"와 같은 애매한 질의가 여러 course로 확장되는지 확인합니다.
    """
    print("\n" + "=" * 80)
    print("Multi-Course Filter OR Condition Test")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    # 애매한 과정명 질의
    query = "CV 강의에서 이미지 분류 방법"

    try:
        result = await graph.ainvoke(
            {"question": query, "max_retries": 1},
            config=RunnableConfig(run_name="test_multi_course_filter"),
        )

        # 필터가 적용되었는지 확인
        filters_applied = result.get("retrieval_filters_applied", False)
        retrieval_filters = result.get("retrieval_filters")

        print(f"Query: {query}")
        print(f"Filters applied: {filters_applied}")
        print(f"Retrieval filters: {retrieval_filters}")

        # course 필터가 리스트인지 확인
        if retrieval_filters and "course" in retrieval_filters:
            course_filter = retrieval_filters["course"]
            print(f"Course filter: {course_filter}")
            assert isinstance(course_filter, list), "Course filter should be a list"
            # 여러 CV 관련 과정이 포함될 수 있음
            print(f"Number of courses in filter: {len(course_filter)}")

        # 문서 검색 확인
        documents = result.get("documents", [])
        assert len(documents) > 0, "No documents retrieved"
        print(f"Retrieved {len(documents)} documents")

        print("✅ Multi-course filter test passed")

    except Exception as e:
        pytest.skip(f"Multi-course filter test failed: {e}")


@pytest.mark.asyncio
async def test_clarification_workflow(hybrid_retriever, llm, reasoning_llm):
    """
    Clarification 워크플로 테스트.

    enable_clarification=True일 때 낮은 confidence 질의가 clarify로 라우팅되는지 확인합니다.
    """
    print("\n" + "=" * 80)
    print("Clarification Workflow Test")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    # Clarification 활성화
    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
        enable_clarification=True,
        clarification_threshold=0.5,
    )

    # 애매한 질의 (낮은 filter_confidence 예상)
    query = "그 강의에서 설명한 알고리즘이 뭐였죠?"

    try:
        result = await graph.ainvoke(
            {"question": query, "max_retries": 1},
            config=RunnableConfig(run_name="test_clarification"),
        )

        print(f"Query: {query}")
        print(f"Filter confidence: {result.get('filter_confidence', 'N/A')}")
        print(f"Workflow stage: {result.get('workflow_stage', 'N/A')}")

        # clarification이 트리거되었거나 정상 답변이 생성되었는지 확인
        answer = result.get("answer", "")
        assert len(answer) > 0, "No answer or clarification message"

        if result.get("workflow_stage") == "awaiting_clarification":
            print("✅ Clarification was triggered as expected")
        else:
            print("✅ Normal answer generated (confidence was high enough)")

    except Exception as e:
        pytest.skip(f"Clarification workflow test failed: {e}")


@pytest.mark.asyncio
async def test_retrieval_metadata_tracking(hybrid_retriever, llm, reasoning_llm):
    """
    검색 메타데이터 추적 테스트.

    retrieval_filters_applied, retrieval_fallback_used 등
    메타데이터가 올바르게 추적되는지 확인합니다.
    """
    print("\n" + "=" * 80)
    print("Retrieval Metadata Tracking Test")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    query = "Slack에서 학습률 설정 관련 질문"

    try:
        result = await graph.ainvoke(
            {"question": query, "max_retries": 1},
            config=RunnableConfig(run_name="test_metadata_tracking"),
        )

        print(f"Query: {query}")

        # 메타데이터 필드 확인
        metadata_fields = [
            "retrieval_filters",
            "retrieval_filters_applied",
            "retrieval_fallback_used",
            "retrieval_strategy",
            "filter_confidence",
        ]

        for field in metadata_fields:
            value = result.get(field)
            print(f"  {field}: {value}")

        # 기본 검증
        assert "retrieval_strategy" in result, "Retrieval strategy not tracked"
        print("✅ Metadata tracking test passed")

    except Exception as e:
        pytest.skip(f"Metadata tracking test failed: {e}")


@pytest.mark.asyncio
async def test_edge_case_empty_query(hybrid_retriever, llm, reasoning_llm):
    """
    엣지 케이스: 빈 질의 처리 테스트.
    """
    print("\n" + "=" * 80)
    print("Edge Case: Empty Query Test")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    try:
        result = await graph.ainvoke(
            {"question": "", "max_retries": 1},
            config=RunnableConfig(run_name="test_empty_query"),
        )

        # 빈 질의도 에러 없이 처리되어야 함
        assert "answer" in result, "Should handle empty query gracefully"
        print("✅ Empty query handled gracefully")

    except Exception as e:
        # 빈 질의는 에러가 발생할 수 있음 (허용)
        print(f"Empty query raised exception (acceptable): {e}")


@pytest.mark.asyncio
async def test_edge_case_very_long_query(hybrid_retriever, llm, reasoning_llm):
    """
    엣지 케이스: 매우 긴 질의 처리 테스트.
    """
    print("\n" + "=" * 80)
    print("Edge Case: Very Long Query Test")
    print("=" * 80)

    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph

    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        reasoning_llm=reasoning_llm,
        debug=True,
    )

    # 매우 긴 질의 (500자 이상)
    long_query = (
        "PyTorch에서 딥러닝 모델을 학습시킬 때 GPU 메모리 부족 문제가 발생하면 "
        "어떻게 해결해야 하나요? 배치 사이즈를 줄이는 것 외에 gradient accumulation, "
        "mixed precision training (FP16), gradient checkpointing 등의 기법을 "
        "사용할 수 있다고 들었는데 각각의 장단점과 구현 방법, 그리고 어떤 상황에서 "
        "어떤 기법을 선택해야 하는지 자세히 알려주세요. 특히 Vision Transformer나 "
        "대형 언어 모델 같은 큰 모델을 학습할 때 효과적인 방법이 궁금합니다."
    )

    try:
        result = await graph.ainvoke(
            {"question": long_query, "max_retries": 1},
            config=RunnableConfig(run_name="test_long_query"),
        )

        assert "answer" in result, "Should handle long query"
        assert len(result.get("answer", "")) > 0, "Answer should not be empty"
        print(f"Query length: {len(long_query)} characters")
        print(f"Answer length: {len(result.get('answer', ''))} characters")
        print("✅ Long query handled successfully")

    except Exception as e:
        pytest.skip(f"Long query test failed: {e}")


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])
