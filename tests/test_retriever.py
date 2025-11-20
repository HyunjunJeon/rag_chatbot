"""
Retriever 통합 테스트 모듈

다양한 검색 전략의 구성과 동작을 검증합니다:
1. Base Hybrid (Dense + Sparse)
2. MultiQuery Retriever
3. Advanced Hybrid (모든 전략 결합)
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from naver_connect_chatbot.rag.retriever_factory import (
    build_dense_sparse_hybrid,
    build_multi_query_retriever,
    build_advanced_hybrid_retriever,
)
from naver_connect_chatbot.rag.retriever.hybrid_retriever import HybridMethod


@pytest.fixture
def sample_documents() -> list[Document]:
    """테스트용 샘플 문서 생성"""
    return [
        Document(
            page_content="Python은 인기 있는 프로그래밍 언어입니다.",
            metadata={"source": "doc1", "topic": "programming"},
        ),
        Document(
            page_content="머신러닝은 인공지능의 한 분야입니다.",
            metadata={"source": "doc2", "topic": "ai"},
        ),
        Document(
            page_content="자연어 처리는 텍스트 분석에 사용됩니다.",
            metadata={"source": "doc3", "topic": "nlp"},
        ),
        Document(
            page_content="벡터 데이터베이스는 임베딩을 저장합니다.",
            metadata={"source": "doc4", "topic": "database"},
        ),
        Document(
            page_content="LangChain은 LLM 애플리케이션 개발 프레임워크입니다.",
            metadata={"source": "doc5", "topic": "framework"},
        ),
    ]


@pytest.fixture
def mock_embedding_model() -> Embeddings:
    """모킹된 임베딩 모델"""
    mock_embeddings = MagicMock(spec=Embeddings)
    # 간단한 더미 벡터 반환 (384차원)
    mock_embeddings.embed_query.return_value = [0.1] * 384
    mock_embeddings.embed_documents.return_value = [[0.1] * 384] * 5
    return mock_embeddings


@pytest.fixture
def mock_llm() -> MagicMock:
    """모킹된 LLM"""
    mock = MagicMock()
    # MultiQuery가 기대하는 형태로 invoke 결과 반환
    mock.invoke.return_value = "관련 쿼리 1\n관련 쿼리 2\n관련 쿼리 3"
    return mock


class TestBaseDenseSparseHybrid:
    """Base Hybrid (Dense + Sparse) 테스트"""

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_dense_sparse_hybrid_default_weights(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """기본 가중치로 Hybrid 생성 테스트"""
        # Qdrant 클라이언트 모킹
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
        )

        assert retriever is not None
        assert len(retriever.retrievers) == 2
        assert retriever.weights == [0.5, 0.5]
        assert retriever.method == HybridMethod.RRF

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_dense_sparse_hybrid_custom_weights(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """커스텀 가중치로 Hybrid 생성 테스트"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            weights=[0.3, 0.7],  # Sparse 30%, Dense 70%
            method=HybridMethod.CC,
        )

        assert retriever.weights == [0.3, 0.7]
        assert retriever.method == HybridMethod.CC

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_sparse_retriever_returns_scores(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """Sparse Retriever가 점수를 메타데이터에 포함하는지 검증"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
        )

        # Sparse retriever만 추출
        sparse_retriever = retriever.retrievers[0]

        # 검색 수행
        results = sparse_retriever.invoke("Python 프로그래밍")

        # 점수가 메타데이터에 포함되어 있는지 확인
        assert len(results) > 0
        for doc in results:
            assert "score" in doc.metadata
            assert isinstance(doc.metadata["score"], float)


class TestMultiQueryRetriever:
    """MultiQuery Retriever 테스트"""

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_multi_query_retriever(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
        mock_llm,
    ):
        """MultiQuery Retriever 생성 테스트"""
        mock_qdrant_client.return_value = MagicMock()

        # Base Hybrid 생성
        base_retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
        )

        # MultiQuery로 래핑
        multi_query = build_multi_query_retriever(
            base_retriever=base_retriever,
            llm=mock_llm,
            num_queries=3,
        )

        assert multi_query is not None
        assert multi_query.num_queries == 3
        assert multi_query.base_retriever == base_retriever


class TestAdvancedHybridRetriever:
    """Advanced Hybrid (모든 전략 결합) 테스트"""

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_advanced_without_multi_query(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """MultiQuery 비활성화 시 Base Hybrid만 반환"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_advanced_hybrid_retriever(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            enable_multi_query=False,
        )

        # HybridRetriever 타입이어야 함
        from naver_connect_chatbot.rag.retriever.hybrid_retriever import (
            HybridRetriever,
        )

        assert isinstance(retriever, HybridRetriever)
        assert len(retriever.retrievers) == 2  # Sparse + Dense

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_advanced_with_multi_query_no_final_hybrid(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
        mock_llm,
    ):
        """MultiQuery 활성화, Final Hybrid 없음"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_advanced_hybrid_retriever(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            llm=mock_llm,
            enable_multi_query=True,
            final_weights=None,  # Final Hybrid 비활성화
        )

        # MultiQueryRetriever 타입이어야 함
        from naver_connect_chatbot.rag.retriever.multi_query_retriever import (
            MultiQueryRetriever,
        )

        assert isinstance(retriever, MultiQueryRetriever)

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_advanced_with_multi_query_and_final_hybrid(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
        mock_llm,
    ):
        """MultiQuery + Final Hybrid 활성화 (이중 계층)"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_advanced_hybrid_retriever(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            llm=mock_llm,
            enable_multi_query=True,
            final_weights=[0.4, 0.6],  # Base 40%, MultiQuery 60%
        )

        # 최상위는 HybridRetriever
        from naver_connect_chatbot.rag.retriever.hybrid_retriever import (
            HybridRetriever,
        )

        assert isinstance(retriever, HybridRetriever)
        # Base Hybrid + MultiQuery = 2개 retriever
        assert len(retriever.retrievers) == 2
        assert retriever.weights == [0.4, 0.6]

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_build_advanced_llm_failure_fallback(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """LLM 초기화 실패 시 Base Hybrid로 폴백"""
        mock_qdrant_client.return_value = MagicMock()

        # LLM을 None으로 전달
        retriever = build_advanced_hybrid_retriever(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            llm=None,  # LLM 없음
            enable_multi_query=True,
        )

        # Base Hybrid만 반환되어야 함
        from naver_connect_chatbot.rag.retriever.hybrid_retriever import (
            HybridRetriever,
        )

        assert isinstance(retriever, HybridRetriever)
        assert len(retriever.retrievers) == 2  # Sparse + Dense


class TestRetrieverIntegration:
    """통합 시나리오 테스트"""

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_end_to_end_retrieval(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """종단간 검색 테스트 (중복 제거 확인)"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            k=3,
        )

        # 검색 수행
        results = retriever.invoke("Python과 머신러닝")

        # 결과 검증
        assert len(results) <= 3  # k=3이므로 최대 3개
        assert len(results) > 0  # 최소 1개는 반환되어야 함

        # 중복 제거 확인 (page_content 기준)
        contents = [doc.page_content for doc in results]
        assert len(contents) == len(set(contents))

    @patch("naver_connect_chatbot.rag.retriever_factory.QdrantClient")
    def test_retrieval_with_scores(
        self,
        mock_qdrant_client,
        sample_documents,
        mock_embedding_model,
    ):
        """점수가 포함된 검색 결과 검증"""
        mock_qdrant_client.return_value = MagicMock()

        retriever = build_dense_sparse_hybrid(
            documents=sample_documents,
            embedding_model=mock_embedding_model,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            method=HybridMethod.CC,  # CC는 점수가 필수
        )

        results = retriever.invoke("데이터베이스")

        # 모든 문서에 점수가 있는지 확인
        for doc in results:
            assert "score" in doc.metadata
            assert 0 <= doc.metadata["score"] <= 1  # 정규화된 점수

