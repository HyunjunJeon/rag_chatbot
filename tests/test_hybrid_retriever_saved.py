"""
저장된 인덱스 기반 Hybrid Retriever 테스트

BM25(Sparse) + Qdrant(Dense) Hybrid Retriever의 동작을 검증합니다.
- 개별 Retriever 테스트 (BM25, Qdrant)
- RRF 융합 테스트
- 가중치 조정 테스트
- 검색 품질 확인
"""

import os
import sys
import time
from pathlib import Path

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
    # 직접 import하여 Settings 초기화 우회
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
        weights=[0.5, 0.5],  # Sparse 50%, Dense 50%
        k=10,
        method=HybridMethod.RRF,
        rrf_c=60,
    )


def test_bm25_retriever_only(hybrid_retriever):
    """BM25 Retriever 단독 테스트"""
    print("\n" + "=" * 80)
    print("1. BM25 Retriever 단독 테스트")
    print("=" * 80)
    
    # BM25 Retriever 접근
    sparse_retriever = hybrid_retriever.retrievers[0]
    
    query = "GPU 메모리 부족 해결 방법"
    
    print(f"\n🔍 쿼리: {query}")
    
    start_time = time.time()
    results = sparse_retriever.invoke(query)
    elapsed = time.time() - start_time
    
    print(f"✅ BM25 검색 완료 (소요 시간: {elapsed:.3f}초)")
    print(f"   - 검색 결과 수: {len(results)}")
    
    # 상위 3개 결과 출력
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get("score", 0.0)
        question = doc.metadata.get("question_text", "")
        course = doc.metadata.get("course", "")
        
        print(f"\n   [{i}] 점수: {score:.4f}")
        print(f"       과정: {course}")
        print(f"       질문: {question[:80]}...")
    
    assert len(results) > 0, "BM25 검색 결과가 없습니다"
    assert len(results) <= 10, "검색 결과가 k=10을 초과합니다"


def test_qdrant_retriever_only(hybrid_retriever):
    """Qdrant Retriever 단독 테스트"""
    print("\n" + "=" * 80)
    print("2. Qdrant Retriever 단독 테스트")
    print("=" * 80)
    
    # Qdrant Retriever 접근
    dense_retriever = hybrid_retriever.retrievers[1]
    
    query = "GPU 메모리 부족 해결 방법"
    
    print(f"\n🔍 쿼리: {query}")
    
    start_time = time.time()
    results = dense_retriever.invoke(query)
    elapsed = time.time() - start_time
    
    print(f"✅ Qdrant 검색 완료 (소요 시간: {elapsed:.3f}초)")
    print(f"   - 검색 결과 수: {len(results)}")
    
    # 상위 3개 결과 출력
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get("score", 0.0)
        question = doc.metadata.get("question_text", "")
        course = doc.metadata.get("course", "")
        
        print(f"\n   [{i}] 점수: {score:.4f}")
        print(f"       과정: {course}")
        print(f"       질문: {question[:80]}...")
    
    assert len(results) > 0, "Qdrant 검색 결과가 없습니다"
    assert len(results) <= 10, "검색 결과가 k=10을 초과합니다"


def test_hybrid_search(hybrid_retriever):
    """Hybrid 검색 테스트 (RRF 융합)"""
    print("\n" + "=" * 80)
    print("3. Hybrid 검색 테스트 (RRF 융합)")
    print("=" * 80)
    
    query = "GPU 메모리 부족 해결 방법"
    
    print(f"\n🔍 쿼리: {query}")
    
    start_time = time.time()
    results = hybrid_retriever.invoke(query)
    elapsed = time.time() - start_time
    
    print(f"✅ Hybrid 검색 완료 (소요 시간: {elapsed:.3f}초)")
    print(f"   - 검색 결과 수: {len(results)}")
    
    # 상위 5개 결과 출력
    print(f"\n📊 상위 5개 결과:")
    for i, doc in enumerate(results[:5], 1):
        # 메타데이터에서 정보 추출
        question = doc.metadata.get("question_text", "")
        answer = doc.metadata.get("answer_text", "")
        course = doc.metadata.get("course", "")
        generation = doc.metadata.get("generation", "")
        
        print(f"\n   [{i}] 과정: {course} ({generation}기)")
        print(f"       질문: {question[:100]}...")
        print(f"       답변: {answer[:100]}...")
    
    assert len(results) > 0, "Hybrid 검색 결과가 없습니다"
    # Hybrid 검색은 두 리트리버의 결과를 RRF로 결합하므로
    # 중복이 적을 경우 k보다 약간 많을 수 있음 (최대 2*k)
    assert len(results) <= 20, f"검색 결과가 너무 많습니다: {len(results)}"


def test_multiple_queries(hybrid_retriever):
    """여러 쿼리로 검색 테스트"""
    print("\n" + "=" * 80)
    print("4. 다양한 쿼리 테스트")
    print("=" * 80)
    
    test_queries = [
        "데이터 증강 기법",
        "optimizer 선택 기준",
        "학습률 설정 방법",
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[{idx}] 쿼리: {query}")
        
        results = hybrid_retriever.invoke(query)
        
        print(f"    ✅ 결과: {len(results)}개")
        if results:
            top_result = results[0]
            question = top_result.metadata.get("question_text", "")
            print(f"    상위 결과: {question[:60]}...")
        
        assert len(results) > 0, f"쿼리 '{query}'의 검색 결과가 없습니다"


def test_weight_variations(embeddings):
    """가중치 변화에 따른 검색 결과 비교"""
    print("\n" + "=" * 80)
    print("5. 가중치 변화 테스트")
    print("=" * 80)
    
    from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid_from_saved
    from naver_connect_chatbot.rag.retriever.hybrid_retriever import HybridMethod
    
    bm25_path = PROJECT_ROOT / "sparse_index" / "kiwi_bm25_slack_qa"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
    
    query = "GPU 메모리 부족 해결 방법"
    
    weight_configs = [
        ([0.3, 0.7], "Sparse 30%, Dense 70%"),
        ([0.5, 0.5], "Sparse 50%, Dense 50%"),
        ([0.7, 0.3], "Sparse 70%, Dense 30%"),
    ]
    
    for weights, description in weight_configs:
        print(f"\n📊 {description}")
        
        retriever = build_dense_sparse_hybrid_from_saved(
            bm25_index_path=str(bm25_path),
            embedding_model=embeddings,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            weights=weights,
            k=5,
            method=HybridMethod.RRF,
        )
        
        results = retriever.invoke(query)
        
        print(f"   결과 수: {len(results)}")
        if results:
            top_result = results[0]
            question = top_result.metadata.get("question_text", "")
            print(f"   상위 결과: {question[:60]}...")


def test_performance_benchmark(hybrid_retriever):
    """검색 성능 벤치마크"""
    print("\n" + "=" * 80)
    print("6. 검색 성능 벤치마크")
    print("=" * 80)
    
    query = "GPU 메모리 부족 해결 방법"
    num_iterations = 5
    
    print(f"\n쿼리: {query}")
    print(f"반복 횟수: {num_iterations}")
    
    times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        results = hybrid_retriever.invoke(query)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        print(f"  [{i+1}] {elapsed:.3f}초 ({len(results)}개 결과)")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 성능 통계:")
    print(f"   평균: {avg_time:.3f}초")
    print(f"   최소: {min_time:.3f}초")
    print(f"   최대: {max_time:.3f}초")
    
    # 성능 목표: 2초 이내
    assert avg_time < 2.0, f"평균 검색 시간이 목표(2초)를 초과: {avg_time:.3f}초"
    print(f"\n✅ 성능 목표 달성 (평균 {avg_time:.3f}초 < 2초)")


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])

