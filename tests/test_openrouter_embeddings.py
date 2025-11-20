"""
OpenRouterEmbeddings 단위 테스트

이 모듈은 OpenRouterEmbeddings 클래스의 동작을 검증합니다.
"""

import os
from typing import List

import pytest
from pydantic import SecretStr

from naver_connect_chatbot.config.embedding import OpenRouterEmbeddings


@pytest.fixture
def openrouter_api_key() -> str:
    """OpenRouter API 키를 환경변수에서 가져옵니다."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
    return api_key


@pytest.fixture
def embeddings(openrouter_api_key: str) -> OpenRouterEmbeddings:
    """OpenRouterEmbeddings 인스턴스를 생성합니다."""
    return OpenRouterEmbeddings(
        model="qwen/qwen3-embedding-4b",
        api_key=SecretStr(openrouter_api_key),
        timeout=30.0,
        max_retries=3,
    )


class TestOpenRouterEmbeddings:
    """OpenRouterEmbeddings 클래스 테스트"""

    def test_initialization(self, openrouter_api_key: str) -> None:
        """초기화 테스트"""
        # 명시적 API 키
        embeddings = OpenRouterEmbeddings(
            api_key=SecretStr(openrouter_api_key)
        )
        assert embeddings.model == "qwen/qwen3-embedding-4b"
        assert embeddings.base_url == "https://openrouter.ai/api/v1"
        assert embeddings.timeout == 60.0
        assert embeddings.max_retries == 3
        assert embeddings.batch_size == 32

    def test_initialization_from_env(self) -> None:
        """환경변수에서 API 키 로드 테스트"""
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
        
        # API 키 없이 초기화 (환경변수에서 자동 로드)
        embeddings = OpenRouterEmbeddings()
        assert embeddings.api_key is not None

    def test_initialization_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API 키 없이 초기화 시 에러 발생 테스트"""
        # 환경변수 제거
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OpenRouter API 키가 필요합니다"):
            OpenRouterEmbeddings()

    def test_embed_query(self, embeddings: OpenRouterEmbeddings) -> None:
        """단일 쿼리 임베딩 테스트"""
        query = "GPU 메모리 부족 해결 방법"
        
        vector = embeddings.embed_query(query)
        
        # 검증
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)
        
        # 차원 확인 (일반적으로 512, 1024, 2048 등)
        print(f"임베딩 차원: {len(vector)}")

    def test_embed_documents(self, embeddings: OpenRouterEmbeddings) -> None:
        """다중 문서 임베딩 테스트"""
        documents = [
            "GPU 메모리 부족 해결 방법",
            "데이터 증강 기법",
            "optimizer 선택 기준",
        ]
        
        vectors = embeddings.embed_documents(documents)
        
        # 검증
        assert isinstance(vectors, list)
        assert len(vectors) == len(documents)
        
        for vector in vectors:
            assert isinstance(vector, list)
            assert len(vector) > 0
            assert all(isinstance(v, float) for v in vector)
        
        # 모든 벡터의 차원이 동일한지 확인
        dimensions = [len(v) for v in vectors]
        assert len(set(dimensions)) == 1, "모든 벡터의 차원이 동일해야 합니다"
        
        print(f"문서 수: {len(vectors)}, 임베딩 차원: {dimensions[0]}")

    def test_embed_documents_batch_processing(self, embeddings: OpenRouterEmbeddings) -> None:
        """배치 처리 테스트"""
        # 배치 크기보다 많은 문서
        num_docs = 50
        documents = [f"테스트 문서 {i+1}" for i in range(num_docs)]
        
        vectors = embeddings.embed_documents(documents)
        
        # 검증
        assert len(vectors) == num_docs
        print(f"배치 처리 완료: {num_docs}개 문서")

    @pytest.mark.asyncio
    async def test_aembed_query(self, embeddings: OpenRouterEmbeddings) -> None:
        """비동기 단일 쿼리 임베딩 테스트"""
        query = "학습률 스케줄러"
        
        vector = await embeddings.aembed_query(query)
        
        # 검증
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_aembed_documents(self, embeddings: OpenRouterEmbeddings) -> None:
        """비동기 다중 문서 임베딩 테스트"""
        documents = [
            "Transformer 아키텍처",
            "Attention 메커니즘",
            "BERT 모델",
        ]
        
        vectors = await embeddings.aembed_documents(documents)
        
        # 검증
        assert isinstance(vectors, list)
        assert len(vectors) == len(documents)
        
        for vector in vectors:
            assert isinstance(vector, list)
            assert len(vector) > 0

    def test_vector_similarity(self, embeddings: OpenRouterEmbeddings) -> None:
        """벡터 유사도 테스트"""
        # 유사한 문서
        doc1 = "GPU 메모리 부족 문제"
        doc2 = "GPU 메모리 부족 해결"
        doc3 = "데이터 증강 기법"
        
        v1 = embeddings.embed_query(doc1)
        v2 = embeddings.embed_query(doc2)
        v3 = embeddings.embed_query(doc3)
        
        # 코사인 유사도 계산
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            import math
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            return dot_product / (norm_a * norm_b)
        
        sim_12 = cosine_similarity(v1, v2)
        sim_13 = cosine_similarity(v1, v3)
        
        # 유사한 문서끼리 더 높은 유사도를 가져야 함
        print(f"유사도 (doc1-doc2): {sim_12:.4f}")
        print(f"유사도 (doc1-doc3): {sim_13:.4f}")
        
        assert sim_12 > sim_13, "유사한 문서끼리 더 높은 유사도를 가져야 합니다"

    def test_empty_text_handling(self, embeddings: OpenRouterEmbeddings) -> None:
        """빈 텍스트 처리 테스트"""
        # 빈 문자열
        with pytest.raises(Exception):
            embeddings.embed_query("")

    def test_langchain_compatibility(self, embeddings: OpenRouterEmbeddings) -> None:
        """LangChain Embeddings 인터페이스 호환성 테스트"""
        from langchain_core.embeddings import Embeddings
        
        # OpenRouterEmbeddings가 Embeddings를 상속하는지 확인
        assert isinstance(embeddings, Embeddings)
        
        # 필수 메서드 존재 확인
        assert hasattr(embeddings, "embed_query")
        assert hasattr(embeddings, "embed_documents")
        assert hasattr(embeddings, "aembed_query")
        assert hasattr(embeddings, "aembed_documents")


if __name__ == "__main__":
    # 간단한 동작 확인
    print("=" * 80)
    print("OpenRouterEmbeddings 간단 테스트")
    print("=" * 80)
    
    try:
        embeddings = OpenRouterEmbeddings()
        
        # 단일 쿼리
        print("\n[1] 단일 쿼리 임베딩")
        query = "GPU 메모리 부족 해결 방법"
        vector = embeddings.embed_query(query)
        print(f"   쿼리: {query}")
        print(f"   벡터 차원: {len(vector)}")
        print(f"   벡터 샘플: {vector[:5]}...")
        
        # 다중 문서
        print("\n[2] 다중 문서 임베딩")
        documents = ["문서1", "문서2", "문서3"]
        vectors = embeddings.embed_documents(documents)
        print(f"   문서 수: {len(vectors)}")
        print(f"   벡터 차원: {len(vectors[0])}")
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

