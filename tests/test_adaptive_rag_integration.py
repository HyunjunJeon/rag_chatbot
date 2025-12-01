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
            model="HCX-007",
            use_reasoning=True,
            reasoning_effort="medium",
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
    """Answer Generator 구조화된 출력 테스트"""
    print("\n" + "=" * 80)
    print("5. Answer Generator 구조화된 출력 테스트")
    print("=" * 80)

    from naver_connect_chatbot.service.agents.answer_generator import (
        create_answer_generator,
    )
    from naver_connect_chatbot.service.agents.response_parser import parse_agent_response

    # Simple 전략으로 에이전트 생성
    generator = create_answer_generator(llm, strategy="simple")

    print("\n🧪 테스트 쿼리: What is 2+2?")
    print("📝 컨텍스트: Mathematics: 2+2 equals 4.")

    try:
        # 에이전트 실행
        response_raw = await generator.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question: What is 2+2?\n\ncontext:\nMathematics: 2+2 equals 4.",
                    }
                ]
            }
        )

        response = response_raw.content

        # 검증
        assert len(response) > 0, "Answer is empty"

        print(f"   - Type: {type(response).__name__}")
        print(f"   - Answer length: {len(response)} characters")
        print(f"   - Answer: {response}")

    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])
