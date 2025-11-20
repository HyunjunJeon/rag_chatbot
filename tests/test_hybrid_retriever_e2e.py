"""
Hybrid Retriever End-to-End 테스트

이 모듈은 BM25(Sparse) + Qdrant(Dense) Hybrid Retriever의 end-to-end 동작을 검증합니다.
"""

import os
import pickle
import sys
from pathlib import Path

from pydantic import SecretStr

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# 설정 우회를 위한 직접 import
import importlib.util


def load_module_directly(name: str, path: Path):
    """모듈을 직접 로드합니다 (설정 파일 로드 우회)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# OpenRouterEmbeddings 직접 로드
embedding_module = load_module_directly(
    "embedding_module",
    PROJECT_ROOT / "app" / "naver_connect_chatbot" / "config" / "embedding.py"
)
OpenRouterEmbeddings = embedding_module.OpenRouterEmbeddings


def test_hybrid_retriever_e2e():
    """
    Hybrid Retriever End-to-End 테스트
    
    단계:
    1. BM25 인덱스 로드
    2. Qdrant 연결
    3. OpenRouter Embeddings 초기화
    4. Hybrid 검색 수행
    5. 결과 분석
    """
    print("=" * 80)
    print("🚀 Hybrid Retriever End-to-End 테스트")
    print("=" * 80)
    
    # ========================================================================
    # 1. BM25 인덱스 로드
    # ========================================================================
    print("\n[1] BM25 인덱스 로드")
    bm25_index_path = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
    
    if not bm25_index_path.exists():
        print(f"   ❌ BM25 인덱스가 없습니다: {bm25_index_path}")
        return False
    
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
    
    bm25_question = bm25_data["bm25_question"]
    documents = bm25_data["documents"]
    
    print(f"   ✅ BM25 인덱스 로드 완료")
    print(f"      - 문서 수: {len(documents):,}개")
    
    # ========================================================================
    # 2. Qdrant 연결
    # ========================================================================
    print("\n[2] Qdrant 연결")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=qdrant_url)
        info = client.get_collection(collection_name)
        
        print(f"   ✅ Qdrant 연결 완료")
        print(f"      - URL: {qdrant_url}")
        print(f"      - Collection: {collection_name}")
        print(f"      - 벡터 수: {info.points_count:,}")
        print(f"      - 벡터 차원: {info.config.params.vectors.size}")
        
    except Exception as e:
        print(f"   ❌ Qdrant 연결 실패: {e}")
        print(f"   ℹ️  Qdrant가 실행 중인지 확인하세요")
        return False
    
    # ========================================================================
    # 3. OpenRouter Embeddings 초기화
    # ========================================================================
    print("\n[3] OpenRouter Embeddings 초기화")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("   ⚠️  OPENROUTER_API_KEY 환경변수가 없습니다")
        print("   ℹ️  벡터 검색 없이 BM25만 테스트합니다")
        embeddings = None
    else:
        try:
            embeddings = OpenRouterEmbeddings(
                model="qwen/qwen3-embedding-4b",
                api_key=SecretStr(api_key),
            )
            print(f"   ✅ OpenRouter Embeddings 초기화 완료")
            print(f"      - 모델: qwen/qwen3-embedding-4b")
        except Exception as e:
            print(f"   ❌ Embeddings 초기화 실패: {e}")
            embeddings = None
    
    # ========================================================================
    # 4. Hybrid 검색 수행
    # ========================================================================
    print("\n[4] Hybrid 검색 수행")
    
    test_queries = [
        "GPU 메모리 부족 해결 방법",
        "데이터 증강 기법",
        "optimizer 선택",
    ]
    
    for query in test_queries:
        print(f"\n   🔍 쿼리: '{query}'")
        print("   " + "-" * 70)
        
        # ------------------------------------------------------------------
        # BM25 검색
        # ------------------------------------------------------------------
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        # 토큰화
        tokens = kiwi.tokenize(query)
        query_tokens = []
        for token in tokens:
            if token.tag in ["NNG", "NNP", "VV", "VA", "SL", "SN"]:
                query_tokens.append(token.form.lower())
        
        # BM25 점수 계산
        import numpy as np
        bm25_scores = bm25_question.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
        
        bm25_results = []
        for rank, idx in enumerate(top_bm25_indices, 1):
            if bm25_scores[idx] > 0:
                doc = documents[idx]
                bm25_results.append({
                    "rank": rank,
                    "score": float(bm25_scores[idx]),
                    "doc": doc
                })
        
        print(f"\n      📊 BM25 검색 결과: {len(bm25_results)}개")
        for i, result in enumerate(bm25_results[:3], 1):
            doc = result["doc"]
            print(f"         [{i}] 점수: {result['score']:.3f}")
            print(f"             과정: {doc['course']} ({doc['generation']}기)")
            print(f"             질문: {doc['question_text'][:50]}...")
        
        # ------------------------------------------------------------------
        # Qdrant 벡터 검색
        # ------------------------------------------------------------------
        if embeddings:
            try:
                # 쿼리 임베딩
                query_vector = embeddings.embed_query(query)
                
                # 벡터 검색
                qdrant_results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=10,
                ).points
                
                print(f"\n      📊 Qdrant 검색 결과: {len(qdrant_results)}개")
                for i, result in enumerate(qdrant_results[:3], 1):
                    print(f"         [{i}] 점수: {result.score:.4f}")
                    if result.payload:
                        print(f"             과정: {result.payload.get('course', 'N/A')}")
                        question = result.payload.get('question_text', '')
                        print(f"             질문: {question[:50]}...")
                
                # --------------------------------------------------------------
                # Reciprocal Rank Fusion (RRF)
                # --------------------------------------------------------------
                print(f"\n      🔀 Reciprocal Rank Fusion (RRF)")
                
                # RRF 점수 계산
                rrf_scores = {}
                k = 60  # RRF 상수
                
                # BM25 점수 추가
                for result in bm25_results:
                    doc_id = result["doc"]["doc_id"]
                    rank = result["rank"]
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
                
                # Qdrant 점수 추가
                for rank, result in enumerate(qdrant_results, 1):
                    if result.payload:
                        doc_id = str(result.id)
                        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
                
                # 정렬
                sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                
                print(f"\n      📊 RRF 통합 결과: {len(sorted_results)}개")
                for i, (doc_id, score) in enumerate(sorted_results[:3], 1):
                    print(f"         [{i}] RRF 점수: {score:.4f}")
                    print(f"             문서 ID: {doc_id[:50]}...")
                
            except Exception as e:
                print(f"\n      ❌ 벡터 검색 실패: {e}")
    
    # ========================================================================
    # 5. 결과 요약
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ Hybrid Retriever End-to-End 테스트 완료!")
    print("=" * 80)
    
    print("\n📊 테스트 요약:")
    print(f"   - BM25 인덱스: {len(documents):,}개 문서")
    print(f"   - Qdrant 컬렉션: {info.points_count:,}개 벡터")
    print(f"   - 테스트 쿼리: {len(test_queries)}개")
    print(f"   - Embeddings: {'✅ 사용' if embeddings else '❌ 미사용'}")
    
    if embeddings:
        print("\n💡 다음 단계:")
        print("   1. retriever_factory를 사용한 정식 Hybrid Retriever 구성")
        print("   2. LangChain BaseRetriever 인터페이스 통합")
        print("   3. Agent/Chain에 Retriever 연결")
    else:
        print("\n💡 OPENROUTER_API_KEY를 설정하면 전체 Hybrid 검색을 테스트할 수 있습니다")
    
    return True


if __name__ == "__main__":
    try:
        success = test_hybrid_retriever_e2e()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

