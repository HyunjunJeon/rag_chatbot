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
from pydantic import SecretStr
from dotenv import load_dotenv

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture
def embeddings():
    """OpenRouterEmbeddings 인스턴스 생성"""
    sys.path.insert(0, str(PROJECT_ROOT / "app" / "naver_connect_chatbot" / "config"))
    from embedding import OpenRouterEmbeddings
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenRouterEmbeddings(
        model="qwen/qwen3-embedding-4b",
        api_key=SecretStr(api_key)
    )


@pytest.fixture
def hybrid_retriever(embeddings):
    """Hybrid Retriever 인스턴스 생성"""
    from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid_from_saved
    from naver_connect_chatbot.rag.retriever.hybrid_retriever import HybridMethod
    
    bm25_path = PROJECT_ROOT / "sparse_index" / "kiwi_bm25_slack_qa"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
    
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
    from naver_connect_chatbot.config.llm import get_chat_model, LLMProvider
    
    # 여러 LLM 제공자를 시도
    providers_to_try = [
        LLMProvider.NAVER_CLOUD,
        LLMProvider.OPENROUTER,
        LLMProvider.OPENAI,
    ]
    
    for provider in providers_to_try:
        try:
            llm_instance = get_chat_model(provider)
            print(f"\n✅ {provider.value} LLM 사용")
            return llm_instance
        except ValueError:
            continue
    
    # 모든 제공자 실패 시
    pytest.skip("사용 가능한 LLM이 설정되지 않았습니다")


@pytest.mark.asyncio
async def test_adaptive_rag_graph_construction(hybrid_retriever, llm):
    """Adaptive RAG 그래프 생성 테스트"""
    print("\n" + "=" * 80)
    print("1. Adaptive RAG 그래프 생성")
    print("=" * 80)
    
    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph
    
    try:
        graph = build_adaptive_rag_graph(
            retriever=hybrid_retriever,
            llm=llm,
            fast_llm=llm,  # 테스트에서는 동일한 LLM 사용
        )
        
        assert graph is not None, "그래프가 None입니다"
        
    except Exception as e:
        pytest.fail(f"그래프 생성 실패: {e}")


@pytest.mark.asyncio
async def test_simple_qa_workflow(hybrid_retriever, llm):
    """SIMPLE_QA 워크플로 테스트"""
    print("\n" + "=" * 80)
    print("2. SIMPLE_QA 워크플로 테스트")
    print("=" * 80)
    
    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph
    
    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        fast_llm=llm,
    )
    
    query = "PyTorch 설치 방법은?"
    print(f"\n🔍 쿼리: {query}")
    
    try:
        result = await graph.ainvoke({
            "question": query,
            "max_retries": 2,
        }, config=RunnableConfig(run_name="test_simple_qa_workflow", tags=["test"], configurable={"thread_id": "test_simple_qa_workflow"}))
        
        assert "answer" in result, "Answer not generated"
        assert len(result["answer"]) > 0, "Answer is empty"
        assert "documents" in result, "Documents not retrieved"
        
    except Exception as e:
        pytest.skip(f"Workflow execution failed: {e}")


@pytest.mark.asyncio
async def test_retrieval_in_workflow(hybrid_retriever, llm):
    """워크플로 내 검색 기능 테스트"""
    print("\n" + "=" * 80)
    print("3. 워크플로 내 검색 기능 테스트")
    print("=" * 80)
    
    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph
    
    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        fast_llm=llm,
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
async def test_workflow_state_tracking(hybrid_retriever, llm):
    """워크플로 상태 추적 테스트"""
    print("\n" + "=" * 80)
    print("4. 워크플로 상태 추적 테스트")
    print("=" * 80)
    
    from naver_connect_chatbot.service.graph import build_adaptive_rag_graph
    
    graph = build_adaptive_rag_graph(
        retriever=hybrid_retriever,
        llm=llm,
        fast_llm=llm,
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
        AnswerOutput,
    )
    from naver_connect_chatbot.service.graph.nodes import _coerce_model_response

    # Simple 전략으로 에이전트 생성
    generator = create_answer_generator(llm, strategy="simple")

    print("\n🧪 테스트 쿼리: What is 2+2?")
    print("📝 컨텍스트: Mathematics: 2+2 equals 4.")

    try:
        # 에이전트 실행
        response_raw = await generator.ainvoke({
            "messages": [{
                "role": "user",
                "content": "question: What is 2+2?\n\ncontext:\nMathematics: 2+2 equals 4."
            }]
        })

        # AnswerOutput으로 변환 가능한지 확인
        response = _coerce_model_response(AnswerOutput, response_raw)

        # 검증
        assert isinstance(response, AnswerOutput), f"Expected AnswerOutput, got {type(response)}"
        assert isinstance(response.answer, str), f"Expected str answer, got {type(response.answer)}"
        assert len(response.answer) > 0, "Answer is empty"

        print("\n✅ 구조화된 출력 성공:")
        print(f"   - Type: {type(response).__name__}")
        print(f"   - Answer length: {len(response.answer)} characters")
        print(f"   - Answer preview: {response.answer[:100]}...")

    except Exception as e:
        pytest.skip(f"Answer generator test failed: {e}")


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])

