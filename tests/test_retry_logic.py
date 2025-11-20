"""
Retry Logic 테스트 모듈

ClovaStudioReranker의 retry 로직이 올바르게 동작하는지 테스트합니다.

테스트 시나리오:
1. 네트워크 오류 후 재시도 성공
2. 타임아웃 후 재시도 성공
3. 5xx 서버 오류 후 재시도 성공
4. 4xx 클라이언트 오류 시 즉시 실패 (재시도 없음)
5. 최대 재시도 횟수 초과 시 최종 실패
6. 지수 백오프 타이밍 검증
"""

import pytest
import respx
from httpx import NetworkError, TimeoutException, Response
from langchain_core.documents import Document

from naver_connect_chatbot.rag.rerank import ClovaStudioReranker


@pytest.fixture
def reranker():
    """ClovaStudioReranker 인스턴스를 생성합니다."""
    return ClovaStudioReranker(
        endpoint="https://test.api.com/reranker/test-id",
        api_key="test-api-key",
        api_gateway_key="test-gateway-key",
        request_timeout=5.0,
        default_top_k=3,
    )


@pytest.fixture
def sample_documents():
    """테스트용 Document 리스트를 생성합니다."""
    return [
        Document(page_content="AI is artificial intelligence", metadata={"source": "doc1"}),
        Document(page_content="ML is machine learning", metadata={"source": "doc2"}),
        Document(page_content="DL is deep learning", metadata={"source": "doc3"}),
    ]


@pytest.fixture
def successful_response():
    """성공적인 API 응답 데이터를 반환합니다."""
    return {
        "result": {
            "topN": 3,
            "documents": [
                {"index": 0, "score": 0.95, "document": {"text": "AI is artificial intelligence"}},
                {"index": 2, "score": 0.85, "document": {"text": "DL is deep learning"}},
                {"index": 1, "score": 0.75, "document": {"text": "ML is machine learning"}},
            ],
        }
    }


class TestRetryOnNetworkError:
    """네트워크 오류 시 재시도 로직을 테스트합니다."""

    @respx.mock
    def test_retry_succeeds_after_network_error(self, reranker, sample_documents, successful_response):
        """
        첫 번째 호출에서 NetworkError 발생, 두 번째 호출에서 성공하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 실패, 두 번째 성공
        route.side_effect = [
            NetworkError("Connection failed"),
            Response(200, json=successful_response),
        ]

        # 재시도 후 성공해야 함
        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert result[0].metadata["rerank_score"] == 0.95
        # NetworkError로 1번 실패 + 1번 성공 = 총 2번 호출
        assert route.call_count == 2

    @respx.mock
    def test_retry_fails_after_max_attempts_network_error(self, reranker, sample_documents):
        """
        NetworkError가 계속 발생하여 최대 재시도 횟수(3회) 초과 시 최종 실패를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(side_effect=NetworkError("Connection failed"))

        # 3번 재시도 후 최종 실패
        with pytest.raises(RuntimeError, match="API 요청 중 오류 발생"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        # 최대 3번 시도
        assert route.call_count == 3


class TestRetryOnTimeout:
    """타임아웃 시 재시도 로직을 테스트합니다."""

    @respx.mock
    def test_retry_succeeds_after_timeout(self, reranker, sample_documents, successful_response):
        """
        첫 번째 호출에서 TimeoutException 발생, 두 번째 호출에서 성공하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 실패, 두 번째 성공
        route.side_effect = [
            TimeoutException("Request timeout"),
            Response(200, json=successful_response),
        ]

        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 2

    @respx.mock
    def test_retry_fails_after_max_attempts_timeout(self, reranker, sample_documents):
        """
        TimeoutException이 계속 발생하여 최대 재시도 횟수 초과 시 최종 실패를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(side_effect=TimeoutException("Request timeout"))

        with pytest.raises(RuntimeError, match="API 요청 중 오류 발생"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 3


class TestRetryOn5xxError:
    """5xx 서버 오류 시 재시도 로직을 테스트합니다."""

    @respx.mock
    def test_retry_succeeds_after_500_error(self, reranker, sample_documents, successful_response):
        """
        첫 번째 호출에서 500 Internal Server Error 발생, 두 번째 호출에서 성공하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 500 오류, 두 번째 성공
        route.side_effect = [
            Response(500, json={"error": "Internal Server Error"}),
            Response(200, json=successful_response),
        ]

        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 2

    @respx.mock
    def test_retry_succeeds_after_503_error(self, reranker, sample_documents, successful_response):
        """
        첫 번째 호출에서 503 Service Unavailable 발생, 두 번째 호출에서 성공하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 503 오류, 두 번째 성공
        route.side_effect = [
            Response(503, json={"error": "Service Unavailable"}),
            Response(200, json=successful_response),
        ]

        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 2

    @respx.mock
    def test_retry_fails_after_max_attempts_5xx(self, reranker, sample_documents):
        """
        500 오류가 계속 발생하여 최대 재시도 횟수 초과 시 최종 실패를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(500, json={"error": "Internal Server Error"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 500\\)"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        # 3번 시도 후 실패
        assert route.call_count == 3


class TestNoRetryOn4xxError:
    """4xx 클라이언트 오류 시 즉시 실패 (재시도 없음)를 테스트합니다."""

    @respx.mock
    def test_no_retry_on_400_bad_request(self, reranker, sample_documents):
        """
        400 Bad Request 발생 시 재시도 없이 즉시 실패하는지 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(400, json={"error": "Bad Request"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 400\\)"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        # 재시도 없이 1번만 호출
        assert route.call_count == 1

    @respx.mock
    def test_no_retry_on_401_unauthorized(self, reranker, sample_documents):
        """
        401 Unauthorized 발생 시 재시도 없이 즉시 실패하는지 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(401, json={"error": "Unauthorized"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 401\\)"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 1

    @respx.mock
    def test_no_retry_on_404_not_found(self, reranker, sample_documents):
        """
        404 Not Found 발생 시 재시도 없이 즉시 실패하는지 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(404, json={"error": "Not Found"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 404\\)"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 1

    @respx.mock
    def test_no_retry_on_429_rate_limit(self, reranker, sample_documents):
        """
        429 Too Many Requests 발생 시 재시도 없이 즉시 실패하는지 테스트합니다.
        (참고: 실제로는 429도 재시도할 수 있지만, 현재 구현은 4xx는 모두 즉시 실패)
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(429, json={"error": "Too Many Requests"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 429\\)"):
            reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 1


class TestAsyncRetryLogic:
    """비동기 메서드의 retry 로직을 테스트합니다."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_retry_succeeds_after_network_error(self, reranker, sample_documents, successful_response):
        """
        비동기 호출에서 NetworkError 후 재시도 성공을 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 실패, 두 번째 성공
        route.side_effect = [
            NetworkError("Connection failed"),
            Response(200, json=successful_response),
        ]

        result = await reranker.arerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert result[0].metadata["rerank_score"] == 0.95
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_retry_succeeds_after_500_error(self, reranker, sample_documents, successful_response):
        """
        비동기 호출에서 500 오류 후 재시도 성공을 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        # 순차적 응답: 첫 번째 500 오류, 두 번째 성공
        route.side_effect = [
            Response(500, json={"error": "Internal Server Error"}),
            Response(200, json=successful_response),
        ]

        result = await reranker.arerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_no_retry_on_400_error(self, reranker, sample_documents):
        """
        비동기 호출에서 400 오류 시 즉시 실패를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(400, json={"error": "Bad Request"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 400\\)"):
            await reranker.arerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_retry_fails_after_max_attempts(self, reranker, sample_documents):
        """
        비동기 호출에서 최대 재시도 횟수 초과 시 최종 실패를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")
        route.mock(return_value=Response(503, json={"error": "Service Unavailable"}))

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 503\\)"):
            await reranker.arerank("What is AI?", sample_documents, top_k=3)

        assert route.call_count == 3


class TestExponentialBackoff:
    """지수 백오프 타이밍을 테스트합니다."""

    @respx.mock
    def test_exponential_backoff_timing(self, reranker, sample_documents, successful_response):
        """
        재시도 간 대기 시간이 지수적으로 증가하는지 테스트합니다.
        """
        import time

        route = respx.post("https://test.api.com/reranker/test-id")

        # 2번 실패 후 성공
        route.side_effect = [
            Response(500, json={"error": "Server Error"}),
            Response(500, json={"error": "Server Error"}),
            Response(200, json=successful_response),
        ]

        start_time = time.time()
        result = reranker.rerank("What is AI?", sample_documents, top_k=3)
        elapsed_time = time.time() - start_time

        assert len(result) == 3
        # 첫 재시도: 최소 2초, 두 번째 재시도: 최소 4초 (지수 백오프)
        # 총 대기 시간은 최소 6초 정도 (실제로는 약간 더 걸릴 수 있음)
        assert elapsed_time >= 4.0, f"Expected at least 4s wait, got {elapsed_time}s"
        assert route.call_count == 3


class TestMixedErrorScenarios:
    """다양한 오류가 혼합된 시나리오를 테스트합니다."""

    @respx.mock
    def test_network_error_then_500_then_success(self, reranker, sample_documents, successful_response):
        """
        NetworkError → 500 오류 → 성공 순서로 발생하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        route.side_effect = [
            NetworkError("Connection failed"),
            Response(500, json={"error": "Server Error"}),
            Response(200, json=successful_response),
        ]

        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 3

    @respx.mock
    def test_timeout_then_503_then_success(self, reranker, sample_documents, successful_response):
        """
        TimeoutException → 503 오류 → 성공 순서로 발생하는 경우를 테스트합니다.
        """
        route = respx.post("https://test.api.com/reranker/test-id")

        route.side_effect = [
            TimeoutException("Request timeout"),
            Response(503, json={"error": "Service Unavailable"}),
            Response(200, json=successful_response),
        ]

        result = reranker.rerank("What is AI?", sample_documents, top_k=3)

        assert len(result) == 3
        assert route.call_count == 3
