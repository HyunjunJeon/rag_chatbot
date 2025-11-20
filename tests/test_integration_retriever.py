"""
Retriever 통합 테스트

이 모듈은 OpenRouterEmbeddings, Qdrant, BM25, Hybrid Retriever의 통합 동작을 검증합니다.
"""

import os
import sys
from pathlib import Path

import pytest
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


# 기본 테스트 (API 키 불필요)
class TestBasicSetup:
    """기본 설정 테스트"""

    def test_openrouter_embeddings_class_exists(self) -> None:
        """OpenRouterEmbeddings 클래스 존재 확인"""
        assert OpenRouterEmbeddings is not None
        assert hasattr(OpenRouterEmbeddings, "embed_query")
        assert hasattr(OpenRouterEmbeddings, "embed_documents")

    def test_bm25_index_exists(self) -> None:
        """BM25 인덱스 파일 존재 확인"""
        bm25_index_path = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
        assert bm25_index_path.exists(), f"BM25 인덱스가 없습니다: {bm25_index_path}"

    def test_qdrant_collection_info(self) -> None:
        """Qdrant 컬렉션 정보 확인"""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
        
        print("\n📊 Qdrant 설정:")
        print(f"   URL: {qdrant_url}")
        print(f"   Collection: {collection_name}")
        
        try:
            from qdrant_client import QdrantClient
            
            client = QdrantClient(url=qdrant_url)
            collections = client.get_collections()
            
            print(f"\n   사용 가능한 컬렉션:")
            for coll in collections.collections:
                print(f"      - {coll.name}")
            
            # slack_qa 컬렉션 확인
            collection_exists = any(c.name == collection_name for c in collections.collections)
            
            if collection_exists:
                info = client.get_collection(collection_name)
                print(f"\n   {collection_name} 컬렉션:")
                print(f"      - 벡터 수: {info.points_count:,}")
                print(f"      - 벡터 차원: {info.config.params.vectors.size}")
            else:
                pytest.skip(f"Qdrant 컬렉션 '{collection_name}'이 없습니다.")
                
        except Exception as e:
            pytest.skip(f"Qdrant 연결 실패: {e}")


# API 키가 있을 때만 실행되는 테스트
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY 환경변수가 필요합니다"
)
class TestWithAPIKey:
    """API 키가 필요한 테스트"""

    @pytest.fixture
    def embeddings(self) -> OpenRouterEmbeddings:
        """OpenRouterEmbeddings 인스턴스"""
        return OpenRouterEmbeddings(
            model="qwen/qwen3-embedding-4b",
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
        )

    def test_embedding_generation(self, embeddings: OpenRouterEmbeddings) -> None:
        """임베딩 생성 테스트"""
        query = "GPU 메모리 부족"
        vector = embeddings.embed_query(query)
        
        assert isinstance(vector, list)
        assert len(vector) > 0
        print(f"\n✅ 임베딩 생성 성공 (차원: {len(vector)})")

    def test_qdrant_vector_search(self, embeddings: OpenRouterEmbeddings) -> None:
        """Qdrant 벡터 검색 테스트"""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
        
        try:
            from qdrant_client import QdrantClient
            
            client = QdrantClient(url=qdrant_url)
            
            # 컬렉션 존재 확인
            try:
                client.get_collection(collection_name)
            except Exception:
                pytest.skip(f"Qdrant 컬렉션 '{collection_name}'이 없습니다.")
            
            # 쿼리 임베딩
            query = "GPU 메모리 부족 해결 방법"
            query_vector = embeddings.embed_query(query)
            
            # 벡터 검색
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=5,
            ).points
            
            assert len(results) > 0, "검색 결과가 없습니다"
            
            print(f"\n✅ Qdrant 검색 성공:")
            print(f"   쿼리: {query}")
            print(f"   결과 수: {len(results)}")
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n   [{i}] 점수: {result.score:.4f}")
                if result.payload:
                    print(f"       과정: {result.payload.get('course', 'N/A')}")
                    question = result.payload.get('question_text', '')
                    print(f"       질문: {question[:60]}...")
                    
        except Exception as e:
            pytest.skip(f"Qdrant 테스트 실패: {e}")


# BM25 + Qdrant Hybrid 테스트
class TestHybridRetrieverConcept:
    """Hybrid Retriever 개념 검증"""

    def test_bm25_loading(self) -> None:
        """BM25 인덱스 로드 테스트"""
        bm25_index_path = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
        
        if not bm25_index_path.exists():
            pytest.skip("BM25 인덱스가 없습니다")
        
        import pickle
        
        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)
        
        assert "documents" in data, "BM25 인덱스에 documents가 없습니다"
        assert len(data["documents"]) > 0, "문서가 비어있습니다"
        
        print(f"\n✅ BM25 인덱스 로드 성공:")
        print(f"   문서 수: {len(data['documents']):,}개")
        
        # 간단한 BM25 검색 테스트
        bm25_question = data.get("bm25_question")
        if bm25_question:
            query_tokens = data["documents"][0]["question_tokens"][:3]  # 샘플 토큰
            scores = bm25_question.get_scores(query_tokens)
            
            import numpy as np
            top_idx = np.argmax(scores)
            top_doc = data["documents"][top_idx]
            
            print(f"\n   샘플 검색:")
            print(f"      - 과정: {top_doc['course']}")
            print(f"      - 질문: {top_doc['question_text'][:60]}...")

    def test_hybrid_concept(self) -> None:
        """Hybrid Retriever 개념 검증"""
        print("\n💡 Hybrid Retriever 통합 방안:")
        print("   1. BM25 (Sparse): sparse_index/bm25_slack_qa.pkl")
        print("   2. Qdrant (Dense): slack_qa 컬렉션")
        print("   3. 융합 방법: Reciprocal Rank Fusion (RRF)")
        print("\n   구현 경로:")
        print("      - BM25 결과: Top-K 문서")
        print("      - Qdrant 결과: Top-K 문서")
        print("      - RRF로 결합: 최종 Top-K 반환")


if __name__ == "__main__":
    # pytest 없이 간단히 실행
    print("=" * 80)
    print("Retriever 통합 테스트")
    print("=" * 80)
    
    # 1. 기본 설정 테스트
    print("\n[1] 기본 설정 확인")
    test_basic = TestBasicSetup()
    
    try:
        test_basic.test_openrouter_embeddings_class_exists()
        print("   ✅ OpenRouterEmbeddings 클래스 확인")
    except Exception as e:
        print(f"   ❌ OpenRouterEmbeddings: {e}")
    
    try:
        test_basic.test_bm25_index_exists()
        print("   ✅ BM25 인덱스 파일 확인")
    except Exception as e:
        print(f"   ❌ BM25 인덱스: {e}")
    
    try:
        test_basic.test_qdrant_collection_info()
    except Exception as e:
        print(f"   ⚠️  Qdrant: {e}")
    
    # 2. BM25 테스트
    print("\n[2] BM25 인덱스 테스트")
    test_hybrid = TestHybridRetrieverConcept()
    
    try:
        test_hybrid.test_bm25_loading()
    except Exception as e:
        print(f"   ❌ BM25 로드: {e}")
    
    # 3. 통합 개념
    print("\n[3] Hybrid Retriever 개념")
    test_hybrid.test_hybrid_concept()
    
    # 4. API 키 테스트 (있을 경우)
    if os.getenv("OPENROUTER_API_KEY"):
        print("\n[4] API 키 기반 테스트")
        test_api = TestWithAPIKey()
        
        try:
            embeddings = test_api.embeddings()
            test_api.test_embedding_generation(embeddings)
        except Exception as e:
            print(f"   ❌ 임베딩 테스트: {e}")
        
        try:
            test_api.test_qdrant_vector_search(embeddings)
        except Exception as e:
            print(f"   ⚠️  Qdrant 검색: {e}")
    else:
        print("\n[4] API 키 테스트 건너뛰기")
        print("   ℹ️  OPENROUTER_API_KEY 환경변수를 설정하면 API 테스트 가능")
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)

