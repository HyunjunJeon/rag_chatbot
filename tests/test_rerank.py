"""
Reranker 모듈 테스트

BaseReranker 인터페이스와 ClovaStudioReranker 구현체를 검증합니다.
"""

import uuid
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from langchain_core.documents import Document

from naver_connect_chatbot.rag.rerank import (
    BaseReranker,
    ClovaStudioReranker,
    _generate_document_id,
    _merge_rerank_metadata,
    _serialize_documents_for_api,
)


# ============================================================================
# 유틸리티 함수 테스트
# ============================================================================


class TestUtilityFunctions:
    """유틸리티 함수 테스트"""

    def test_generate_document_id_with_metadata_key(self) -> None:
        """메타데이터에 id_key가 있을 때 ID 생성 검증"""
        doc = Document(
            page_content="Hello World",
            metadata={"source_id": "doc123", "other": "data"},
        )
        doc_id = _generate_document_id(doc, id_key="source_id")
        assert doc_id == "doc123"

    def test_generate_document_id_with_missing_key(self) -> None:
        """메타데이터에 id_key가 없을 때 해시 생성 검증"""
        doc = Document(page_content="Hello World", metadata={})
        doc_id = _generate_document_id(doc, id_key="source_id")
        # MD5 해시는 32자 hex 문자열
        assert len(doc_id) == 32
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_generate_document_id_fallback_to_hash(self) -> None:
        """id_key가 None일 때 해시 생성 검증"""
        doc = Document(page_content="Test Content", metadata={"source_id": "doc1"})
        doc_id = _generate_document_id(doc, id_key=None)
        # id_key=None이므로 source_id를 무시하고 해시 생성
        assert len(doc_id) == 32

    def test_generate_document_id_consistency(self) -> None:
        """동일한 내용은 동일한 해시 생성 검증"""
        doc1 = Document(page_content="Same Content", metadata={})
        doc2 = Document(page_content="Same Content", metadata={})
        assert _generate_document_id(doc1, None) == _generate_document_id(doc2, None)

    def test_serialize_documents_for_api(self) -> None:
        """문서를 API용 텍스트 리스트로 변환 검증"""
        docs = [
            Document(page_content="First", metadata={"id": "1"}),
            Document(page_content="Second", metadata={"id": "2"}),
            Document(page_content="Third", metadata={"id": "3"}),
        ]
        serialized = _serialize_documents_for_api(docs)
        assert serialized == ["First", "Second", "Third"]

    def test_serialize_empty_documents(self) -> None:
        """빈 문서 리스트 직렬화 검증"""
        serialized = _serialize_documents_for_api([])
        assert serialized == []

    def test_merge_rerank_metadata(self) -> None:
        """재정렬 메타데이터 병합 검증"""
        doc = Document(
            page_content="Original Content",
            metadata={"source": "file.txt", "page": 1},
        )
        updated = _merge_rerank_metadata(doc, score=0.95, rank=1)

        # 원본 메타데이터 보존
        assert updated.metadata["source"] == "file.txt"
        assert updated.metadata["page"] == 1

        # 새로운 메타데이터 추가
        assert updated.metadata["rerank_score"] == 0.95
        assert updated.metadata["rerank_rank"] == 1
        assert updated.metadata["score"] == 0.95

        # 원본 문서는 변경되지 않음
        assert "rerank_score" not in doc.metadata

    def test_merge_rerank_metadata_overwrites_existing_score(self) -> None:
        """기존 score 필드를 덮어쓰는지 검증"""
        doc = Document(
            page_content="Content",
            metadata={"score": 0.5},
        )
        updated = _merge_rerank_metadata(doc, score=0.9, rank=2)
        assert updated.metadata["score"] == 0.9
        assert updated.metadata["rerank_score"] == 0.9


# ============================================================================
# BaseReranker 인터페이스 테스트
# ============================================================================


class TestBaseReranker:
    """BaseReranker 추상 클래스 계약 검증"""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """추상 클래스를 직접 인스턴스화할 수 없음을 검증"""
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_subclass_must_implement_rerank(self) -> None:
        """서브클래스가 rerank 메서드를 구현해야 함을 검증"""

        class IncompleteReranker(BaseReranker):
            pass

        with pytest.raises(TypeError):
            IncompleteReranker()  # type: ignore[abstract]

    def test_arerank_default_implementation_calls_rerank(self) -> None:
        """arerank 기본 구현이 rerank를 호출하는지 검증"""

        class MockReranker(BaseReranker):
            def rerank(
                self,
                query: str,
                documents: list[Document],
                *,
                top_k: int | None = None,
            ) -> list[Document]:
                return documents[:top_k] if top_k else documents

        reranker = MockReranker()
        docs = [Document(page_content=f"Doc {i}") for i in range(5)]

        # 동기 메서드 호출
        result_sync = reranker.rerank("query", docs, top_k=3)
        assert len(result_sync) == 3

        # 비동기 메서드 호출 (기본 구현은 동기 메서드를 호출)
        import asyncio

        result_async = asyncio.run(reranker.arerank("query", docs, top_k=3))
        assert len(result_async) == 3


# ============================================================================
# ClovaStudioReranker 초기화 테스트
# ============================================================================


class TestClovaStudioRerankerInitialization:
    """ClovaStudioReranker 초기화 및 설정 테스트"""

    def test_initialization_with_required_params(self) -> None:
        """필수 파라미터로 초기화 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com/rerank",
            api_key="test_key",
        )
        assert reranker.endpoint == "https://api.example.com/rerank"
        assert reranker.api_key == "test_key"
        assert reranker.api_gateway_key is None
        assert reranker.request_timeout == 30.0
        assert reranker.default_top_k == 10

    def test_initialization_with_all_params(self) -> None:
        """모든 파라미터로 초기화 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com/rerank",
            api_key="test_key",
            api_gateway_key="gateway_key",
            request_timeout=60.0,
            default_top_k=20,
            id_key="doc_id",
        )
        assert reranker.api_gateway_key == "gateway_key"
        assert reranker.request_timeout == 60.0
        assert reranker.default_top_k == 20
        assert reranker.id_key == "doc_id"

    def test_initialization_fails_without_endpoint(self) -> None:
        """endpoint 없이 초기화 시 실패 검증"""
        with pytest.raises(ValueError, match="endpoint는 빈 문자열일 수 없습니다"):
            ClovaStudioReranker(endpoint="", api_key="test_key")

    def test_initialization_fails_without_api_key(self) -> None:
        """api_key 없이 초기화 시 실패 검증"""
        with pytest.raises(ValueError, match="api_key는 빈 문자열일 수 없습니다"):
            ClovaStudioReranker(
                endpoint="https://api.example.com",
                api_key="",
            )

    def test_from_settings(self) -> None:
        """Settings 객체로부터 생성 검증"""
        # Mock settings 객체
        mock_settings = MagicMock()
        mock_settings.endpoint = "https://settings.api.com"
        mock_settings.api_key = "settings_key"
        mock_settings.api_gateway_key = "gateway"
        mock_settings.request_timeout = 45.0
        mock_settings.default_top_k = 15
        mock_settings.id_key = "custom_id"

        reranker = ClovaStudioReranker.from_settings(mock_settings)

        assert reranker.endpoint == "https://settings.api.com"
        assert reranker.api_key == "settings_key"
        assert reranker.api_gateway_key == "gateway"
        assert reranker.request_timeout == 45.0
        assert reranker.default_top_k == 15
        assert reranker.id_key == "custom_id"

    def test_client_lazy_initialization(self) -> None:
        """Client가 lazy 생성되는지 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )
        # 초기에는 None
        assert reranker._client is None

        # 첫 접근 시 생성
        client = reranker.client
        assert isinstance(client, httpx.Client)

        # 같은 인스턴스 재사용
        assert reranker.client is client


# ============================================================================
# ClovaStudioReranker 헤더 및 검증 테스트
# ============================================================================


class TestClovaStudioRerankerHelpers:
    """ClovaStudioReranker 헬퍼 메서드 테스트"""

    def test_build_headers_minimal(self) -> None:
        """최소 헤더 생성 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )
        headers = reranker._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["X-NCP-CLOVASTUDIO-API-KEY"] == "test_key"
        assert "X-NCP-CLOVASTUDIO-REQUEST-ID" in headers
        assert "X-NCP-APIGW-API-KEY" not in headers

        # REQUEST-ID가 유효한 UUID인지 확인
        request_id = headers["X-NCP-CLOVASTUDIO-REQUEST-ID"]
        uuid.UUID(request_id)  # 유효하지 않으면 예외 발생

    def test_build_headers_with_gateway_key(self) -> None:
        """Gateway 키가 있을 때 헤더 생성 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
            api_gateway_key="gateway_key",
        )
        headers = reranker._build_headers()

        assert headers["X-NCP-APIGW-API-KEY"] == "gateway_key"

    def test_build_headers_with_custom_request_id(self) -> None:
        """사용자 지정 REQUEST-ID 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )
        custom_id = "custom-request-id-123"
        headers = reranker._build_headers(request_id=custom_id)

        assert headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] == custom_id

    def test_validate_inputs_empty_query(self) -> None:
        """빈 query 검증 실패 테스트"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )
        docs = [Document(page_content="Test")]

        with pytest.raises(ValueError, match="query는 빈 문자열일 수 없습니다"):
            reranker._validate_inputs("", docs)

        with pytest.raises(ValueError, match="query는 빈 문자열일 수 없습니다"):
            reranker._validate_inputs("   ", docs)

    def test_validate_inputs_empty_documents(self) -> None:
        """빈 documents 검증 실패 테스트"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )

        with pytest.raises(ValueError, match="documents는 비어있을 수 없습니다"):
            reranker._validate_inputs("query", [])

    def test_validate_inputs_too_many_documents(self) -> None:
        """100개 초과 문서 경고 로그 테스트"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com",
            api_key="test_key",
        )
        docs = [Document(page_content=f"Doc {i}") for i in range(101)]

        # 경고는 발생하지만 예외는 발생하지 않음
        reranker._validate_inputs("query", docs)


# ============================================================================
# ClovaStudioReranker API 호출 테스트 (Mocked)
# ============================================================================


@respx.mock
class TestClovaStudioRerankerAPI:
    """ClovaStudioReranker API 호출 테스트 (respx mock 사용)"""

    def test_rerank_success(self) -> None:
        """정상적인 API 호출 및 응답 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
            default_top_k=3,
        )

        # Mock API 응답
        mock_response = {
            "result": {
                "topN": 3,
                "documents": [
                    {"index": 2, "score": 0.95, "document": {"text": "Third"}},
                    {"index": 0, "score": 0.85, "document": {"text": "First"}},
                    {"index": 1, "score": 0.75, "document": {"text": "Second"}},
                ],
            }
        }

        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        # 테스트 문서
        docs = [
            Document(page_content="First", metadata={"id": "1"}),
            Document(page_content="Second", metadata={"id": "2"}),
            Document(page_content="Third", metadata={"id": "3"}),
        ]

        # 재정렬 수행
        result = reranker.rerank("What is AI?", docs)

        # 검증
        assert len(result) == 3
        assert result[0].page_content == "Third"
        assert result[1].page_content == "First"
        assert result[2].page_content == "Second"

        # 메타데이터 검증
        assert result[0].metadata["rerank_score"] == 0.95
        assert result[0].metadata["rerank_rank"] == 1
        assert result[1].metadata["rerank_rank"] == 2
        assert result[2].metadata["rerank_rank"] == 3

    def test_rerank_with_top_k(self) -> None:
        """top_k 파라미터 동작 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        mock_response = {
            "result": {
                "topN": 5,
                "documents": [
                    {"index": i, "score": 1.0 - i * 0.1, "document": {"text": f"Doc {i}"}}
                    for i in range(5)
                ],
            }
        }

        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        docs = [Document(page_content=f"Doc {i}") for i in range(5)]

        # top_k=2로 제한
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].metadata["rerank_rank"] == 1
        assert result[1].metadata["rerank_rank"] == 2

    def test_rerank_empty_documents_returns_immediately(self) -> None:
        """빈 문서 리스트는 API 호출 없이 즉시 반환 검증"""
        reranker = ClovaStudioReranker(
            endpoint="https://api.example.com/rerank",
            api_key="test_key",
        )

        # API 호출이 발생하지 않아야 함 (respx mock이 호출되지 않음)
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_http_error(self) -> None:
        """HTTP 오류 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        # Mock 400 Bad Request
        respx.post(endpoint).mock(
            return_value=httpx.Response(400, json={"error": "Bad Request"})
        )

        docs = [Document(page_content="Test")]

        with pytest.raises(RuntimeError, match="API 호출 실패 \\(HTTP 400\\)"):
            reranker.rerank("query", docs)

    def test_rerank_network_error(self) -> None:
        """네트워크 오류 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
            request_timeout=1.0,
        )

        # Mock 네트워크 오류
        respx.post(endpoint).mock(side_effect=httpx.ConnectError("Connection failed"))

        docs = [Document(page_content="Test")]

        with pytest.raises(RuntimeError, match="API 요청 중 오류 발생"):
            reranker.rerank("query", docs)

    def test_rerank_invalid_response_schema(self) -> None:
        """잘못된 응답 스키마 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        # Mock 잘못된 스키마
        invalid_response = {"invalid": "schema"}
        respx.post(endpoint).mock(return_value=httpx.Response(200, json=invalid_response))

        docs = [Document(page_content="Test")]

        with pytest.raises(ValueError, match="API 응답 스키마가 올바르지 않습니다"):
            reranker.rerank("query", docs)

    def test_rerank_invalid_json_response(self) -> None:
        """유효하지 않은 JSON 응답 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        # Mock 잘못된 JSON
        respx.post(endpoint).mock(return_value=httpx.Response(200, content=b"Not JSON"))

        docs = [Document(page_content="Test")]

        with pytest.raises(RuntimeError, match="응답 JSON 파싱 실패"):
            reranker.rerank("query", docs)

    def test_rerank_invalid_document_index(self) -> None:
        """유효하지 않은 문서 인덱스 처리 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        # Mock 유효하지 않은 인덱스
        mock_response = {
            "result": {
                "topN": 2,
                "documents": [
                    {"index": 0, "score": 0.9, "document": {"text": "Valid"}},
                    {"index": 10, "score": 0.8, "document": {"text": "Invalid"}},  # 범위 초과
                ],
            }
        }
        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        docs = [Document(page_content="Only one doc")]

        # 유효하지 않은 인덱스는 건너뛰고 유효한 문서만 반환
        result = reranker.rerank("query", docs)
        assert len(result) == 1
        assert result[0].page_content == "Only one doc"


# ============================================================================
# ClovaStudioReranker 비동기 테스트
# ============================================================================


@pytest.mark.asyncio
@respx.mock
class TestClovaStudioRerankerAsync:
    """ClovaStudioReranker 비동기 메서드 테스트"""

    async def test_arerank_success(self) -> None:
        """비동기 재정렬 성공 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
        )

        mock_response = {
            "result": {
                "topN": 2,
                "documents": [
                    {"index": 1, "score": 0.9, "document": {"text": "Second"}},
                    {"index": 0, "score": 0.8, "document": {"text": "First"}},
                ],
            }
        }

        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        docs = [
            Document(page_content="First"),
            Document(page_content="Second"),
        ]

        result = await reranker.arerank("query", docs)

        assert len(result) == 2
        assert result[0].page_content == "Second"
        assert result[1].page_content == "First"

    async def test_arerank_with_top_k(self) -> None:
        """비동기 top_k 파라미터 검증"""
        endpoint = "https://api.example.com/rerank"
        reranker = ClovaStudioReranker(
            endpoint=endpoint,
            api_key="test_key",
            default_top_k=1,
        )

        mock_response = {
            "result": {
                "topN": 3,
                "documents": [
                    {"index": i, "score": 1.0 - i * 0.1, "document": {"text": f"Doc {i}"}}
                    for i in range(3)
                ],
            }
        }

        respx.post(endpoint).mock(return_value=httpx.Response(200, json=mock_response))

        docs = [Document(page_content=f"Doc {i}") for i in range(3)]

        # default_top_k=1 사용
        result = await reranker.arerank("query", docs)
        assert len(result) == 1

