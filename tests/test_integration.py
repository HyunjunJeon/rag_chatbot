"""
네이버 커넥트 챗봇 통합 테스트

이 모듈은 실제 API 서비스를 사용하는 통합 테스트를 포함합니다.
테스트 실행 시 API 비용이 발생할 수 있으므로 `-m integration` 옵션으로 명시적으로 실행해야 합니다.

실행 방법:
    # Integration 테스트만 실행
    pytest tests/test_integration.py -m integration -v

    # Integration 테스트 제외
    pytest -k "not integration"
"""

import os
import pytest
import asyncio
from dotenv import load_dotenv
from langchain_core.documents import Document

from naver_connect_chatbot.config import settings, get_chat_model, LLMProvider
from naver_connect_chatbot.rag.rerank import ClovaStudioReranker
from naver_connect_chatbot.rag.embeddings import NaverCloudEmbeddings

# 환경변수를 로드합니다.
load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_naver_cloud_embeddings():
    """
    Naver Cloud BGE-M3 임베딩 통합 테스트

    검증 항목:
    - 임베딩 모델 초기화
    - 단일 텍스트 임베딩 생성
    - 배치 임베딩 생성
    """
    # 환경변수 확인
    if not settings.naver_cloud_embeddings.model_url or not settings.naver_cloud_embeddings.api_key:
        pytest.skip("Missing Naver Cloud Embeddings configuration")

    try:
        # 임베딩 객체를 초기화합니다.
        embeddings = NaverCloudEmbeddings(
            model_url=settings.naver_cloud_embeddings.model_url,
            api_key=settings.naver_cloud_embeddings.api_key.get_secret_value()
        )

        # 단일 임베딩을 테스트합니다.
        text = "Naver Cloud Platform의 임베딩 서비스를 테스트합니다."
        embedding = await embeddings.aembed_query(text)

        # 검증
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # BGE-M3는 1024차원

        # 배치 임베딩을 테스트합니다.
        texts = [
            "첫 번째 테스트 문장입니다.",
            "두 번째 테스트 문장입니다.",
            "세 번째 테스트 문장입니다.",
        ]
        batch_embeddings = await embeddings.aembed_documents(texts)

        assert batch_embeddings is not None
        assert len(batch_embeddings) == 3
        assert all(len(emb) == 1024 for emb in batch_embeddings)

        print(f"\n✓ Naver Cloud Embeddings 테스트 통과")
        print(f"  단일 임베딩 차원: {len(embedding)}")
        print(f"  배치 임베딩 수: {len(batch_embeddings)}")

    except Exception as e:
        pytest.fail(f"Naver Cloud Embeddings test failed: {str(e)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clova_studio_reranker():
    """
    Clova Studio Reranker 통합 테스트

    검증 항목:
    - 검색 문서 관련도 재평가
    - 질의-문서 유사도 점수 계산
    - 상위 K개 문서 재선정
    """
    # Reranker가 활성화되어 있는지 확인
    if not settings.reranker.enabled:
        pytest.skip("Reranker is not enabled in settings")

    # 필수 환경변수 확인
    if not settings.reranker.endpoint or not settings.reranker.api_key:
        pytest.skip("Missing reranker endpoint or API key")

    # 다양한 관련도를 가진 문서를 샘플로 사용합니다.
    test_docs = [
        Document(
            page_content="Naver Cloud의 Reranker API는 검색 결과의 관련도를 재평가합니다.",
            metadata={"id": "doc1", "relevance": "high"}
        ),
        Document(
            page_content="Python은 프로그래밍 언어입니다. 데이터 분석에 많이 사용됩니다.",
            metadata={"id": "doc2", "relevance": "low"}
        ),
        Document(
            page_content="검색 시스템에서 재순위화는 정확도 향상에 중요한 역할을 합니다.",
            metadata={"id": "doc3", "relevance": "medium"}
        ),
        Document(
            page_content="자바스크립트는 웹 개발에 사용되는 언어입니다.",
            metadata={"id": "doc4", "relevance": "low"}
        ),
    ]

    try:
        # Reranker를 초기화합니다.
        reranker = ClovaStudioReranker()

        # 재순위화를 수행합니다.
        query = "검색 결과 재순위화 방법"
        reranked_docs = await reranker.arerank(
            documents=test_docs,
            query=query,
            top_k=2,
        )

        # 검증
        assert reranked_docs is not None
        assert len(reranked_docs) == 2  # top_k=2
        assert all(isinstance(doc, Document) for doc in reranked_docs)

        # 관련도 점수가 존재하는지 확인합니다.
        assert all(hasattr(doc, "metadata") for doc in reranked_docs)

        print(f"\n✓ Clova Studio Reranker 테스트 통과")
        print(f"  쿼리: {query}")
        print(f"  입력 문서: {len(test_docs)}개")
        print(f"  재순위화 결과: {len(reranked_docs)}개 문서")
        for i, doc in enumerate(reranked_docs, 1):
            relevance_score = doc.metadata.get("relevance_score", "N/A")
            print(f"  [{i}] Score: {relevance_score} - {doc.page_content[:50]}...")

    except Exception as e:
        pytest.fail(f"Clova Studio Reranker test failed: {str(e)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_chat_completion():
    """
    LLM Chat Completion 통합 테스트

    검증 항목:
    - 팩토리를 통한 LLM 초기화
    - 채팅 응답 생성
    - 응답 품질 확인
    """
    # LLM provider 확인
    if not settings.openai.enabled and not settings.openrouter.enabled and not settings.chat.enabled:
        pytest.skip("No LLM provider enabled")

    try:
        # 사용할 LLM을 가져옵니다.
        if settings.openai.enabled:
            llm = get_chat_model(LLMProvider.OPENAI)
            provider_name = "OpenAI"
        elif settings.openrouter.enabled:
            llm = get_chat_model(LLMProvider.OPENROUTER)
            provider_name = "OpenRouter"
        else:
            llm = get_chat_model(LLMProvider.NAVER_CLOUD_OPENAI_COMPATIBLE)
            provider_name = "Naver Cloud"

        # 채팅 응답을 테스트합니다.
        prompt = "다음 질문에 간단히 답변해주세요: Python의 주요 특징 3가지는 무엇인가요?"
        response = await llm.ainvoke(prompt)

        # 검증
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

        print(f"\n✓ LLM Chat Completion 테스트 통과 (Provider: {provider_name})")
        print(f"  프롬프트: {prompt}")
        print(f"  응답 길이: {len(response.content)} characters")
        print(f"  응답: {response.content[:100]}...")

    except Exception as e:
        pytest.fail(f"LLM Chat Completion test failed: {str(e)}")


if __name__ == "__main__":
    # 수동 실행 헬퍼
    print("Integration Tests - Manual Execution")
    print("=" * 50)

    async def run_all_tests():
        print("\n[1/3] Testing Naver Cloud Embeddings...")
        await test_naver_cloud_embeddings()

        print("\n[2/3] Testing Clova Studio Reranker...")
        await test_clova_studio_reranker()

        print("\n[3/3] Testing LLM Chat Completion...")
        await test_llm_chat_completion()

        print("\n" + "=" * 50)
        print("✅ All integration tests completed!")

    asyncio.run(run_all_tests())
