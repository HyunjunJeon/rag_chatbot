"""
환경 검증 테스트

Hybrid Retriever와 Adaptive RAG 통합을 위한 환경 설정을 검증합니다.
- OPENROUTER_API_KEY 확인
- Qdrant 연결 확인
- BM25 인덱스 확인
- 임베딩 차원 호환성 확인
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")


def test_openrouter_api_key():
    """OPENROUTER_API_KEY 환경변수 확인"""
    print("\n" + "=" * 80)
    print("1. OPENROUTER_API_KEY 확인")
    print("=" * 80)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, (
        "OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다. "
        ".env 파일에 OPENROUTER_API_KEY를 설정하세요."
    )
    
    assert len(api_key) > 0, "OPENROUTER_API_KEY가 빈 문자열입니다."
    
    print(f"✅ OPENROUTER_API_KEY 확인 완료 (길이: {len(api_key)}자)")
    print(f"   키 프리픽스: {api_key[:10]}...")


def test_qdrant_connection():
    """Qdrant 서버 연결 확인"""
    print("\n" + "=" * 80)
    print("2. Qdrant 연결 확인")
    print("=" * 80)
    
    from qdrant_client import QdrantClient
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "slack_qa")
    
    try:
        client = QdrantClient(url=qdrant_url)
        print(f"✅ Qdrant 연결 성공: {qdrant_url}")
    except Exception as e:
        pytest.fail(
            f"Qdrant 연결 실패: {e}\n"
            f"확인 사항:\n"
            f"  1. Qdrant가 실행 중인가요? (docker ps | grep qdrant)\n"
            f"  2. URL이 올바른가요? ({qdrant_url})"
        )
    
    # 컬렉션 확인
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ 컬렉션 확인: {collection_name}")
        print(f"   - 벡터 수: {collection_info.points_count:,}")
        print(f"   - 벡터 차원: {collection_info.config.params.vectors.size}")
        
        # 차원 정보 저장 (다음 테스트에서 사용)
        global QDRANT_VECTOR_DIM
        QDRANT_VECTOR_DIM = collection_info.config.params.vectors.size
        
    except Exception as e:
        pytest.fail(
            f"컬렉션 조회 실패: {e}\n"
            f"확인 사항:\n"
            f"  1. 컬렉션이 생성되어 있나요? ({collection_name})\n"
            f"  2. 벡터 데이터가 인덱싱되어 있나요?"
        )


def test_bm25_index():
    """BM25 인덱스 파일 확인"""
    print("\n" + "=" * 80)
    print("3. BM25 인덱스 확인")
    print("=" * 80)
    
    bm25_index_path = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
    
    assert bm25_index_path.exists(), (
        f"BM25 인덱스 파일이 없습니다: {bm25_index_path}\n"
        f"확인 사항:\n"
        f"  1. sparse_index/ 디렉토리가 존재하나요?\n"
        f"  2. bm25_slack_qa.pkl 파일이 생성되어 있나요?\n"
        f"  3. document_processing/rebuild_bm25_for_chatbot.py를 실행했나요?"
    )
    
    # 인덱스 로드 테스트
    import pickle
    
    try:
        with open(bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
        
        assert "bm25_question" in bm25_data, "BM25 인덱스에 'bm25_question' 키가 없습니다."
        assert "documents" in bm25_data, "BM25 인덱스에 'documents' 키가 없습니다."
        
        num_docs = len(bm25_data["documents"])
        
        print(f"✅ BM25 인덱스 로드 성공")
        print(f"   - 경로: {bm25_index_path}")
        print(f"   - 문서 수: {num_docs:,}")
        print(f"   - 파일 크기: {bm25_index_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        pytest.fail(f"BM25 인덱스 로드 실패: {e}")


def test_openrouter_embeddings_dimension():
    """OpenRouter 임베딩 차원 확인 및 Qdrant 호환성 검증"""
    print("\n" + "=" * 80)
    print("4. OpenRouter 임베딩 차원 확인")
    print("=" * 80)
    
    # Settings 초기화 우회하여 직접 OpenRouterEmbeddings 사용
    from pydantic import SecretStr
    
    # 직접 import
    sys.path.insert(0, str(PROJECT_ROOT / "app" / "naver_connect_chatbot" / "config"))
    from embedding import OpenRouterEmbeddings
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.fail("OPENROUTER_API_KEY가 설정되지 않았습니다.")
    
    try:
        embeddings = OpenRouterEmbeddings(
            model="qwen/qwen3-embedding-4b",
            api_key=SecretStr(api_key)
        )
        print(f"✅ OpenRouterEmbeddings 초기화 성공")
        print(f"   - 모델: qwen/qwen3-embedding-4b")
        
    except Exception as e:
        pytest.fail(
            f"OpenRouterEmbeddings 초기화 실패: {e}\n"
            f"확인 사항:\n"
            f"  1. OPENROUTER_API_KEY가 올바른가요?\n"
            f"  2. OpenRouter API에 접근 가능한가요?"
        )
    
    # 테스트 쿼리로 차원 확인
    try:
        test_query = "테스트 쿼리"
        vector = embeddings.embed_query(test_query)
        
        embedding_dim = len(vector)
        print(f"✅ 임베딩 생성 성공")
        print(f"   - 차원: {embedding_dim}")
        print(f"   - 벡터 샘플: [{vector[0]:.4f}, {vector[1]:.4f}, ...]")
        
        # Qdrant 차원과 비교
        if 'QDRANT_VECTOR_DIM' in globals():
            qdrant_dim = globals()['QDRANT_VECTOR_DIM']
            print(f"\n📊 차원 호환성 검증:")
            print(f"   - Qdrant 컬렉션 차원: {qdrant_dim}")
            print(f"   - OpenRouter 임베딩 차원: {embedding_dim}")
            
            if embedding_dim == qdrant_dim:
                print(f"   ✅ 차원 일치! Hybrid Retriever 사용 가능")
            else:
                print(f"   ⚠️  차원 불일치!")
                print(f"\n💡 해결 방안:")
                print(f"   Option A: Naver Cloud Embeddings 사용 (1024차원)")
                print(f"   Option B: qwen3-embedding-4b로 Qdrant 재인덱싱")
                
                pytest.fail(
                    f"임베딩 차원 불일치: Qdrant={qdrant_dim}, OpenRouter={embedding_dim}\n"
                    f"위의 해결 방안 중 하나를 선택하여 진행하세요."
                )
        
    except Exception as e:
        pytest.fail(f"임베딩 생성 실패: {e}")


def test_naver_cloud_embeddings_alternative():
    """대안: Naver Cloud Embeddings 사용 가능 여부 확인"""
    print("\n" + "=" * 80)
    print("5. Naver Cloud Embeddings 확인 (대안)")
    print("=" * 80)
    
    # 환경변수 확인
    naver_url = os.getenv("NAVER_CLOUD_EMBEDDINGS_MODEL_URL")
    naver_key = os.getenv("NAVER_CLOUD_EMBEDDINGS_API_KEY")
    
    if not naver_url or not naver_key:
        print("⚠️  Naver Cloud Embeddings 미설정")
        print("   - 필요 시 .env에 다음 변수를 설정하세요:")
        print("   - NAVER_CLOUD_EMBEDDINGS_MODEL_URL")
        print("   - NAVER_CLOUD_EMBEDDINGS_API_KEY")
        return  # 선택사항이므로 실패하지 않음
    
    try:
        from pydantic import SecretStr
        
        # 직접 import
        sys.path.insert(0, str(PROJECT_ROOT / "app" / "naver_connect_chatbot" / "config"))
        from embedding import NaverCloudEmbeddings
        
        embeddings = NaverCloudEmbeddings(
            api_url=naver_url,
            api_key=SecretStr(naver_key)
        )
        
        # 테스트 쿼리
        test_query = "테스트 쿼리"
        vector = embeddings.embed_query(test_query)
        
        print(f"✅ Naver Cloud Embeddings 사용 가능")
        print(f"   - 차원: {len(vector)} (BGE-M3)")
        
        # Qdrant 차원과 비교
        if 'QDRANT_VECTOR_DIM' in globals():
            qdrant_dim = globals()['QDRANT_VECTOR_DIM']
            if len(vector) == qdrant_dim:
                print(f"   ✅ Qdrant 컬렉션과 호환됨 (차원: {qdrant_dim})")
            else:
                print(f"   ⚠️  Qdrant 차원({qdrant_dim})과 불일치({len(vector)})")
        
    except Exception as e:
        print(f"⚠️  Naver Cloud Embeddings 테스트 실패: {e}")


def test_summary():
    """환경 검증 요약"""
    print("\n" + "=" * 80)
    print("✅ 환경 검증 완료")
    print("=" * 80)
    print("\n다음 단계:")
    print("  1. retriever_factory에 build_dense_sparse_hybrid_from_saved() 추가")
    print("  2. Hybrid Retriever 단독 테스트")
    print("  3. Adaptive RAG 통합")
    print("  4. End-to-End 챗봇 테스트")


if __name__ == "__main__":
    # 직접 실행 시 pytest 대신 순차 실행
    import traceback
    
    tests = [
        ("OPENROUTER_API_KEY 확인", test_openrouter_api_key),
        ("Qdrant 연결 확인", test_qdrant_connection),
        ("BM25 인덱스 확인", test_bm25_index),
        ("OpenRouter 임베딩 차원 확인", test_openrouter_embeddings_dimension),
        ("Naver Cloud Embeddings 확인", test_naver_cloud_embeddings_alternative),
        ("요약", test_summary),
    ]
    
    failed = []
    
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {name} 실패:")
            traceback.print_exc()
            failed.append(name)
    
    if failed:
        print(f"\n\n❌ 실패한 테스트: {len(failed)}개")
        for name in failed:
            print(f"   - {name}")
        sys.exit(1)
    else:
        print("\n\n✅ 모든 환경 검증 통과!")
        sys.exit(0)

