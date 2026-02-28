"""
Dual Retriever (Qdrant + Google WebSearch) + ClovaStudio Reranker 통합 테스트.

테스트 환경:
- Qdrant VectorDB: 불필요 (MockRetriever 사용)
- GOOGLE_API_KEY: integration 테스트에 필요
- CLOVASTUDIO_API_KEY: reranker 테스트에 필요
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from naver_connect_chatbot.rag.web_search import extract_grounding_documents
from naver_connect_chatbot.service.tool.retrieval_tool import (
    DualRetrievalResult,
    ValidatedRetrievalResult,
    _deduplicate_documents,
    _tag_source_type,
    rerank_and_filter,
    retrieve_and_validate,
    retrieve_dual_source_async,
)


# ============================================================================
# T-1: Google Search → Document 반환 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_google_search_returns_documents(gemini_llm_settings):
    """Google Search grounding 호출 → Document 리스트 반환 검증."""
    from naver_connect_chatbot.rag.web_search import google_search_retrieve

    docs = await google_search_retrieve(
        "PyTorch autograd 자동 미분 원리",
        gemini_llm_settings,
        max_results=5,
    )

    # grounding이 반환되어야 함 (빈 결과도 API 자체는 성공)
    assert isinstance(docs, list)
    if docs:
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata.get("source_type") == "web_search" for d in docs)
        assert all(d.page_content.strip() for d in docs)
        # 최소 하나는 URL을 가져야 함
        urls = [d.metadata.get("url", "") for d in docs]
        assert any(url for url in urls), "At least one document should have a URL"


# ============================================================================
# T-2: Grounding Document 추출 (Unit, Mock)
# ============================================================================


def test_extract_grounding_documents():
    """AIMessage에서 grounding_chunks/supports 추출 → Document 변환."""
    # grounding 메타데이터가 포함된 mock AIMessage
    response = AIMessage(
        content="PyTorch는 자동 미분을 지원합니다.",
        additional_kwargs={
            "grounding_metadata": {
                "web_search_queries": ["PyTorch autograd"],
                "grounding_chunks": [
                    {"web": {"uri": "https://pytorch.org/docs", "title": "PyTorch Docs"}},
                    {"web": {"uri": "https://example.com/ml", "title": "ML Tutorial"}},
                ],
                "grounding_supports": [
                    {
                        "segment": {"text": "PyTorch의 autograd는 자동 미분 엔진입니다."},
                        "grounding_chunk_indices": [0],
                        "confidence_scores": [0.95],
                    },
                    {
                        "segment": {"text": "텐서 연산의 기울기를 자동으로 계산합니다."},
                        "grounding_chunk_indices": [0, 1],
                        "confidence_scores": [0.88, 0.75],
                    },
                ],
            }
        },
    )

    docs = extract_grounding_documents(response)

    assert len(docs) == 2
    # 첫 번째 문서: PyTorch autograd
    assert "autograd" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "web_search"
    assert docs[0].metadata["url"] == "https://pytorch.org/docs"
    assert docs[0].metadata["grounding_confidence"] == 0.95

    # 두 번째 문서: 텐서 연산
    assert "기울기" in docs[1].page_content
    assert docs[1].metadata["source_type"] == "web_search"
    # confidence_scores 중 가장 높은 값의 chunk 사용
    assert docs[1].metadata["grounding_confidence"] == 0.88


def test_extract_grounding_documents_empty():
    """grounding 메타데이터가 없는 경우 빈 리스트 반환."""
    response = AIMessage(content="일반 응답", additional_kwargs={})
    docs = extract_grounding_documents(response)
    assert docs == []


def test_extract_grounding_documents_no_supports():
    """grounding_supports 없이 chunks만 있는 경우."""
    response = AIMessage(
        content="응답",
        additional_kwargs={
            "grounding_metadata": {
                "grounding_chunks": [
                    {"web": {"uri": "https://example.com", "title": "Example Page"}},
                ],
                "grounding_supports": [],
            }
        },
    )
    docs = extract_grounding_documents(response)
    assert len(docs) == 1
    assert docs[0].page_content == "Example Page"
    assert docs[0].metadata["source_type"] == "web_search"


# ============================================================================
# T-3: Dual Retrieval 병합 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_dual_retrieval_merge(mock_retriever_with_source, gemini_llm_settings):
    """MockRetriever(3 docs) + WebSearch → 병합 결과에 양쪽 source_type 존재."""
    result = await retrieve_dual_source_async(
        query="딥러닝 프레임워크 PyTorch란?",
        retriever=mock_retriever_with_source,
        llm_settings=gemini_llm_settings,
        enable_web_search=True,
    )

    assert isinstance(result, DualRetrievalResult)
    assert result.qdrant_count == 3
    assert result.web_search_activated is True
    assert len(result.documents) >= 3  # 최소 qdrant 3개

    # source_type 분포 확인
    source_types = {doc.metadata.get("source_type") for doc in result.documents}
    assert "qdrant" in source_types
    # web_search가 있을 수도 있고 없을 수도 있음 (Gemini grounding 여부에 따라)


# ============================================================================
# T-4: 중복 제거 (Unit, Mock)
# ============================================================================


def test_dual_retrieval_deduplication():
    """동일 content의 web/qdrant 문서 → 중복 제거 확인."""
    same_content = "PyTorch는 자동 미분을 지원하는 프레임워크입니다."

    docs = [
        Document(
            page_content=same_content,
            metadata={"source_type": "qdrant"},
        ),
        Document(
            page_content=same_content,
            metadata={"source_type": "web_search"},
        ),
        Document(
            page_content="CNN은 이미지 처리에 사용됩니다.",
            metadata={"source_type": "qdrant"},
        ),
    ]

    unique, removed = _deduplicate_documents(docs)

    assert len(unique) == 2
    assert removed == 1
    # 첫 번째 (qdrant)가 유지되어야 함
    assert unique[0].metadata["source_type"] == "qdrant"


def test_tag_source_type():
    """source_type 태깅 함수 동작 검증."""
    docs = [
        Document(page_content="test1", metadata={}),
        Document(page_content="test2", metadata={"source_type": "existing"}),
    ]

    tagged = _tag_source_type(docs, "qdrant")

    assert tagged[0].metadata["source_type"] == "qdrant"
    # 이미 태깅된 것은 유지 (setdefault 동작)
    assert tagged[1].metadata["source_type"] == "existing"


# ============================================================================
# T-5: Reranker 병합 문서 처리 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_rerank_combined_documents(reranker):
    """병합된 문서(web+qdrant) → Reranker → rank/score 부여."""
    documents = [
        Document(
            page_content="PyTorch는 Facebook이 개발한 딥러닝 프레임워크로 자동 미분을 지원합니다.",
            metadata={"source_type": "qdrant"},
        ),
        Document(
            page_content="TensorFlow는 Google이 개발한 머신러닝 프레임워크입니다.",
            metadata={"source_type": "web_search"},
        ),
        Document(
            page_content="PyTorch의 autograd는 자동 미분 엔진으로 gradient를 자동 계산합니다.",
            metadata={"source_type": "web_search"},
        ),
    ]

    reranked = await reranker.arerank(
        query="PyTorch 자동 미분이란?",
        documents=documents,
        top_k=3,
    )

    assert len(reranked) > 0
    # 모든 reranked 문서에 rerank_rank가 있어야 함
    for doc in reranked:
        assert "rerank_rank" in doc.metadata
        assert "rerank_score" in doc.metadata
        assert doc.metadata["rerank_rank"] >= 1
    # rank는 오름차순
    ranks = [doc.metadata["rerank_rank"] for doc in reranked]
    assert ranks == sorted(ranks)


# ============================================================================
# T-6: Rank 기반 필터링 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_rerank_filter_by_rank(reranker):
    """top_k=10, min_rank=2 → rank 3+ 문서 제거 확인."""
    documents = [
        Document(
            page_content="PyTorch autograd는 자동 미분 기능을 제공합니다.",
            metadata={"source_type": "qdrant"},
        ),
        Document(
            page_content="CNN은 이미지 분류에 사용되는 딥러닝 모델입니다.",
            metadata={"source_type": "qdrant"},
        ),
        Document(
            page_content="Transformer는 Self-Attention 기반의 아키텍처입니다.",
            metadata={"source_type": "web_search"},
        ),
        Document(
            page_content="RNN은 시퀀스 데이터 처리에 사용됩니다.",
            metadata={"source_type": "web_search"},
        ),
        Document(
            page_content="GAN은 생성 모델의 한 종류입니다.",
            metadata={"source_type": "qdrant"},
        ),
    ]

    filtered = await rerank_and_filter(
        query="PyTorch 자동 미분 원리",
        documents=documents,
        reranker=reranker,
        top_k=5,
        min_rank=2,
    )

    assert len(filtered) <= 2
    # 통과한 문서들은 rank 2 이하여야 함
    for doc in filtered:
        assert doc.metadata["rerank_rank"] <= 2


# ============================================================================
# T-7: Full Pipeline E2E (Integration)
# ============================================================================


@pytest.mark.integration
async def test_full_pipeline_end_to_end(
    mock_retriever_with_source, reranker, gemini_llm_settings
):
    """질문 → Dual Retrieval → Rerank → Filter → ValidatedRetrievalResult."""
    result = await retrieve_and_validate(
        query="딥러닝에서 PyTorch 텐서(Tensor)란 무엇인가?",
        retriever=mock_retriever_with_source,
        reranker=reranker,
        llm_settings=gemini_llm_settings,
        enable_web_search=True,
        rerank_top_k=10,
        rerank_min_rank=5,
    )

    assert isinstance(result, ValidatedRetrievalResult)
    assert result.qdrant_count == 3
    assert len(result.all_documents) >= 3
    assert len(result.verified_documents) > 0
    assert result.filtered_count == len(result.verified_documents)

    # 검증된 문서에 rerank metadata가 있어야 함
    for doc in result.verified_documents:
        assert "rerank_rank" in doc.metadata
        assert doc.metadata["rerank_rank"] <= 5

    # source_distribution이 비어있지 않아야 함
    assert result.source_distribution
    assert sum(result.source_distribution.values()) == len(result.verified_documents)


# ============================================================================
# T-8: Qdrant 빈 결과 → WebSearch만으로 파이프라인 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_pipeline_web_only_fallback(empty_retriever, reranker, gemini_llm_settings):
    """Qdrant 빈 결과 → WebSearch만으로 파이프라인 완주."""
    result = await retrieve_and_validate(
        query="최신 LLM 모델 비교 GPT vs Claude",
        retriever=empty_retriever,
        reranker=reranker,
        llm_settings=gemini_llm_settings,
        enable_web_search=True,
        min_qdrant_results=2,
        rerank_top_k=10,
        rerank_min_rank=5,
    )

    assert isinstance(result, ValidatedRetrievalResult)
    assert result.qdrant_count == 0
    # WebSearch가 활성화되어 문서가 있을 수 있음
    assert result.all_documents is not None
    # web_count >= 0 (Gemini grounding 여부에 따라)
    assert result.web_count >= 0


# ============================================================================
# T-9: enable_web_search=False → Qdrant만 사용 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_pipeline_qdrant_only(mock_retriever_with_source, reranker, gemini_llm_settings):
    """enable_web_search=False → Qdrant만 사용 + Rerank."""
    result = await retrieve_and_validate(
        query="CNN 아키텍처 설명",
        retriever=mock_retriever_with_source,
        reranker=reranker,
        llm_settings=gemini_llm_settings,
        enable_web_search=False,
        min_qdrant_results=0,  # WebSearch 강제 활성화 방지
        rerank_top_k=5,
        rerank_min_rank=3,
    )

    assert isinstance(result, ValidatedRetrievalResult)
    assert result.qdrant_count == 3
    assert result.web_count == 0

    # 모든 verified_documents는 qdrant에서 온 것이어야 함
    for doc in result.verified_documents:
        assert doc.metadata.get("source_type") == "qdrant"

    # source_distribution에 web_search가 없어야 함
    assert "web_search" not in result.source_distribution


# ============================================================================
# T-10: source_distribution 정확성 (Integration)
# ============================================================================


@pytest.mark.integration
async def test_source_distribution_tracking(
    mock_retriever_with_source, reranker, gemini_llm_settings
):
    """ValidatedRetrievalResult.source_distribution 정확성 검증."""
    result = await retrieve_and_validate(
        query="PyTorch 텐서 연산과 자동 미분",
        retriever=mock_retriever_with_source,
        reranker=reranker,
        llm_settings=gemini_llm_settings,
        enable_web_search=True,
        rerank_top_k=10,
        rerank_min_rank=5,
    )

    # source_distribution 합계 == verified_documents 수
    total_in_dist = sum(result.source_distribution.values())
    assert total_in_dist == len(result.verified_documents)

    # 각 문서의 source_type과 distribution이 일치하는지 검증
    actual_counts: dict[str, int] = {}
    for doc in result.verified_documents:
        src = doc.metadata.get("source_type", "unknown")
        actual_counts[src] = actual_counts.get(src, 0) + 1

    assert actual_counts == result.source_distribution
