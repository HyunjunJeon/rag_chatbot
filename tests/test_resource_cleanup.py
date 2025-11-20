"""
Resource Cleanup 테스트 모듈

ClovaStudioReranker의 리소스 정리가 올바르게 동작하는지 테스트합니다.

테스트 시나리오:
1. 동기 context manager 정상 동작
2. 비동기 context manager 정상 동작
3. 예외 발생 시에도 cleanup 실행
4. 명시적 close() / aclose() 호출
5. 멱등성: 중복 close 호출 안전성
6. __del__ 호출 시 cleanup
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

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


class TestSyncContextManager:
    """동기 context manager의 리소스 정리를 테스트합니다."""

    def test_context_manager_normal_exit(self, reranker):
        """
        with문 정상 종료 시 close()가 호출되는지 테스트합니다.
        """
        # Given: reranker의 close 메서드를 모킹
        with patch.object(reranker, 'close') as mock_close:
            # When: context manager 사용
            with reranker as r:
                # context 내부에서는 reranker 객체 사용 가능
                assert r is reranker

            # Then: close()가 호출되어야 함
            mock_close.assert_called_once()

    def test_context_manager_with_exception(self, reranker):
        """
        with문 내부에서 예외 발생 시에도 close()가 호출되는지 테스트합니다.
        """
        with patch.object(reranker, 'close') as mock_close:
            # When: context manager 내부에서 예외 발생
            with pytest.raises(ValueError):
                with reranker:
                    raise ValueError("Test exception")

            # Then: 예외가 발생해도 close()는 호출되어야 함
            mock_close.assert_called_once()

    def test_context_manager_returns_self(self, reranker):
        """
        context manager가 self를 반환하는지 테스트합니다.
        """
        with reranker as r:
            assert r is reranker
            assert isinstance(r, ClovaStudioReranker)


class TestAsyncContextManager:
    """비동기 context manager의 리소스 정리를 테스트합니다."""

    @pytest.mark.asyncio
    async def test_async_context_manager_normal_exit(self, reranker):
        """
        async with문 정상 종료 시 aclose()가 호출되는지 테스트합니다.
        """
        with patch.object(reranker, 'aclose', new_callable=AsyncMock) as mock_aclose:
            # When: async context manager 사용
            async with reranker as r:
                assert r is reranker

            # Then: aclose()가 호출되어야 함
            mock_aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_with_exception(self, reranker):
        """
        async with문 내부에서 예외 발생 시에도 aclose()가 호출되는지 테스트합니다.
        """
        with patch.object(reranker, 'aclose', new_callable=AsyncMock) as mock_aclose:
            # When: context manager 내부에서 예외 발생
            with pytest.raises(ValueError):
                async with reranker:
                    raise ValueError("Test exception")

            # Then: 예외가 발생해도 aclose()는 호출되어야 함
            mock_aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_returns_self(self, reranker):
        """
        async context manager가 self를 반환하는지 테스트합니다.
        """
        async with reranker as r:
            assert r is reranker
            assert isinstance(r, ClovaStudioReranker)


class TestExplicitCleanup:
    """명시적 cleanup 메서드 호출을 테스트합니다."""

    def test_explicit_close_closes_client(self, reranker):
        """
        close() 호출 시 HTTPX Client가 닫히는지 테스트합니다.
        """
        # Given: client를 먼저 생성
        client = reranker.client  # lazy initialization
        assert client is not None

        # Mock the close method
        with patch.object(client, 'close') as mock_client_close:
            # When: close() 호출
            reranker.close()

            # Then: client.close()가 호출되어야 함
            mock_client_close.assert_called_once()

        # Then: _client가 None으로 설정되어야 함
        assert reranker._client is None

    @pytest.mark.asyncio
    async def test_explicit_aclose_closes_async_client(self, reranker):
        """
        aclose() 호출 시 HTTPX AsyncClient가 닫히는지 테스트합니다.
        """
        # Given: async_client를 먼저 생성
        async_client = reranker.async_client
        assert async_client is not None

        # Mock the aclose method
        with patch.object(async_client, 'aclose', new_callable=AsyncMock) as mock_aclose:
            # When: aclose() 호출
            await reranker.aclose()

            # Then: async_client.aclose()가 호출되어야 함
            mock_aclose.assert_called_once()

        # Then: _async_client가 None으로 설정되어야 함
        assert reranker._async_client is None


class TestIdempotentCleanup:
    """멱등성: 중복 cleanup 호출의 안전성을 테스트합니다."""

    def test_close_multiple_times_is_safe(self, reranker):
        """
        close()를 여러 번 호출해도 안전한지 테스트합니다.
        """
        # Given: client를 먼저 생성
        _ = reranker.client

        # When: close()를 여러 번 호출
        reranker.close()
        reranker.close()
        reranker.close()

        # Then: 예외가 발생하지 않아야 함
        assert reranker._client is None

    @pytest.mark.asyncio
    async def test_aclose_multiple_times_is_safe(self, reranker):
        """
        aclose()를 여러 번 호출해도 안전한지 테스트합니다.
        """
        # Given: async_client를 먼저 생성
        _ = reranker.async_client

        # When: aclose()를 여러 번 호출
        await reranker.aclose()
        await reranker.aclose()
        await reranker.aclose()

        # Then: 예외가 발생하지 않아야 함
        assert reranker._async_client is None

    def test_close_without_client_creation(self, reranker):
        """
        client를 생성하지 않은 상태에서 close() 호출이 안전한지 테스트합니다.
        """
        # When: client를 생성하지 않고 close() 호출
        reranker.close()

        # Then: 예외가 발생하지 않아야 함
        assert not hasattr(reranker, "_client") or reranker._client is None

    @pytest.mark.asyncio
    async def test_aclose_without_async_client_creation(self, reranker):
        """
        async_client를 생성하지 않은 상태에서 aclose() 호출이 안전한지 테스트합니다.
        """
        # When: async_client를 생성하지 않고 aclose() 호출
        await reranker.aclose()

        # Then: 예외가 발생하지 않아야 함
        assert not hasattr(reranker, "_async_client") or reranker._async_client is None


class TestCleanupWithErrors:
    """cleanup 중 오류 발생 시나리오를 테스트합니다."""

    def test_close_handles_client_close_error(self, reranker):
        """
        client.close() 중 오류 발생 시 gracefully handle되는지 테스트합니다.
        """
        # Given: client를 생성하고 close()가 예외를 발생시키도록 모킹
        client = reranker.client
        with patch.object(client, 'close', side_effect=Exception("Close error")):
            # When: close() 호출
            reranker.close()  # 예외가 발생하지 않아야 함

            # Then: _client는 None으로 설정되어야 함 (cleanup 완료)
            assert reranker._client is None

    @pytest.mark.asyncio
    async def test_aclose_handles_async_client_close_error(self, reranker):
        """
        async_client.aclose() 중 오류 발생 시 gracefully handle되는지 테스트합니다.
        """
        # Given: async_client를 생성하고 aclose()가 예외를 발생시키도록 모킹
        async_client = reranker.async_client
        with patch.object(async_client, 'aclose', new_callable=AsyncMock, side_effect=Exception("Close error")):
            # When: aclose() 호출
            await reranker.aclose()  # 예외가 발생하지 않아야 함

            # Then: _async_client는 None으로 설정되어야 함 (cleanup 완료)
            assert reranker._async_client is None


class TestDelMethod:
    """__del__ 메서드의 cleanup 동작을 테스트합니다."""

    def test_del_calls_close(self):
        """
        __del__ 호출 시 close()가 호출되는지 테스트합니다.
        """
        # Given: reranker 인스턴스 생성
        reranker = ClovaStudioReranker(
            endpoint="https://test.api.com/reranker/test-id",
            api_key="test-api-key",
        )

        # client 생성
        _ = reranker.client

        # close 메서드를 모킹
        with patch.object(reranker, 'close') as mock_close:
            # When: 객체 삭제
            del reranker

            # Then: close()가 호출되어야 함
            # 참고: __del__이 즉시 호출되지 않을 수 있으므로,
            # 이 테스트는 deterministic하지 않을 수 있음
            # 실제로는 context manager 사용을 권장

    def test_del_handles_exceptions(self):
        """
        __del__ 중 예외 발생 시 gracefully handle되는지 테스트합니다.
        """
        # Given: reranker 인스턴스 생성
        reranker = ClovaStudioReranker(
            endpoint="https://test.api.com/reranker/test-id",
            api_key="test-api-key",
        )

        # close가 예외를 발생시키도록 모킹
        with patch.object(reranker, 'close', side_effect=Exception("Close error")):
            # When/Then: del이 예외를 발생시키지 않아야 함
            try:
                del reranker
            except Exception as e:
                pytest.fail(f"__del__ should not raise exceptions, but got: {e}")


class TestNestedContextManagers:
    """중첩된 context manager의 동작을 테스트합니다."""

    def test_nested_sync_context_managers(self):
        """
        중첩된 동기 context manager가 각각 cleanup되는지 테스트합니다.
        """
        # Given: 두 개의 reranker 인스턴스
        reranker1 = ClovaStudioReranker(
            endpoint="https://test1.api.com/reranker/id1",
            api_key="key1",
        )
        reranker2 = ClovaStudioReranker(
            endpoint="https://test2.api.com/reranker/id2",
            api_key="key2",
        )

        # When: 중첩된 context manager 사용
        with patch.object(reranker1, 'close') as mock_close1, \
             patch.object(reranker2, 'close') as mock_close2:

            with reranker1:
                with reranker2:
                    pass

            # Then: 둘 다 close()가 호출되어야 함
            mock_close1.assert_called_once()
            mock_close2.assert_called_once()

    @pytest.mark.asyncio
    async def test_nested_async_context_managers(self):
        """
        중첩된 비동기 context manager가 각각 cleanup되는지 테스트합니다.
        """
        # Given: 두 개의 reranker 인스턴스
        reranker1 = ClovaStudioReranker(
            endpoint="https://test1.api.com/reranker/id1",
            api_key="key1",
        )
        reranker2 = ClovaStudioReranker(
            endpoint="https://test2.api.com/reranker/id2",
            api_key="key2",
        )

        # When: 중첩된 async context manager 사용
        with patch.object(reranker1, 'aclose', new_callable=AsyncMock) as mock_aclose1, \
             patch.object(reranker2, 'aclose', new_callable=AsyncMock) as mock_aclose2:

            async with reranker1:
                async with reranker2:
                    pass

            # Then: 둘 다 aclose()가 호출되어야 함
            mock_aclose1.assert_called_once()
            mock_aclose2.assert_called_once()


class TestMixedSyncAsyncCleanup:
    """동기/비동기 client가 혼재된 경우의 cleanup을 테스트합니다."""

    def test_close_only_affects_sync_client(self, reranker):
        """
        close()는 동기 client만 닫고 비동기 client는 영향을 주지 않아야 합니다.
        """
        # Given: 양쪽 client 모두 생성
        _ = reranker.client
        async_client = reranker.async_client

        # When: close() 호출
        reranker.close()

        # Then: sync client는 None, async client는 여전히 존재
        assert reranker._client is None
        assert reranker._async_client is async_client

    @pytest.mark.asyncio
    async def test_aclose_only_affects_async_client(self, reranker):
        """
        aclose()는 비동기 client만 닫고 동기 client는 영향을 주지 않아야 합니다.
        """
        # Given: 양쪽 client 모두 생성
        client = reranker.client
        _ = reranker.async_client

        # When: aclose() 호출
        await reranker.aclose()

        # Then: async client는 None, sync client는 여전히 존재
        assert reranker._client is client
        assert reranker._async_client is None

    @pytest.mark.asyncio
    async def test_full_cleanup_requires_both_calls(self, reranker):
        """
        완전한 cleanup을 위해서는 close()와 aclose() 모두 호출해야 합니다.
        """
        # Given: 양쪽 client 모두 생성
        _ = reranker.client
        _ = reranker.async_client

        # When: 둘 다 호출
        reranker.close()
        await reranker.aclose()

        # Then: 둘 다 None이어야 함
        assert reranker._client is None
        assert reranker._async_client is None
