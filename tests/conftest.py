"""테스트 전역 설정."""

# pytest-asyncio 플러그인을 명시적으로 로드하여 asyncio 테스트를 지원합니다.
pytest_plugins = ("pytest_asyncio",)

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from loguru import logger

# ============================================================================
# Test Logging Configuration
# ============================================================================

# 테스트 로그 디렉토리
TEST_LOG_DIR = Path(__file__).parent.parent / "logs" / "tests"

# 세션 레벨에서 공유할 로그 파일 경로 (전역 변수)
_test_log_file: Path | None = None


def get_test_log_file() -> Path | None:
    """현재 테스트 세션의 로그 파일 경로를 반환합니다."""
    return _test_log_file


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """
    세션 스코프 fixture로 테스트 로깅을 설정합니다.

    - evaluation 컨텍스트가 바인딩된 로그만 파일에 저장
    - 앱 로거와 분리되어 테스트 로그만 별도 파일에 기록
    """
    global _test_log_file

    # 로그 디렉토리 생성
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 타임스탬프 기반 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _test_log_file = TEST_LOG_DIR / f"evaluation_{timestamp}.log"

    # 테스트 전용 파일 핸들러 추가
    handler_id = logger.add(
        str(_test_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {message}",
        level="DEBUG",
        filter=lambda record: "evaluation" in record["extra"].get("context", ""),
        encoding="utf-8",
        enqueue=True,
    )

    # 테스트 로거 초기화 로그
    test_logger = logger.bind(context="evaluation")
    test_logger.info(f"Test logging initialized: {_test_log_file}")

    yield

    # 세션 종료 시 정리
    test_logger.info("Test session completed")
    try:
        logger.remove(handler_id)
    except ValueError:
        # 앱의 setup_logger()가 logger.remove()를 호출하여
        # 이미 핸들러가 제거된 경우 안전하게 무시
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """async def 테스트를 asyncio.run으로 강제 실행합니다."""
    testfunction = pyfuncitem.obj
    import inspect

    if inspect.iscoroutinefunction(testfunction):
        sig = inspect.signature(testfunction)
        kwargs = {name: pyfuncitem.funcargs[name] for name in sig.parameters}
        asyncio.run(testfunction(**kwargs))
        return True
    return None


# ============================================================================
# Gemini Integration Test Fixtures
# ============================================================================

import sys  # noqa: E402

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402


class MockRetriever(BaseRetriever):
    """VectorDB 없이 사전 정의된 문서를 반환하는 Mock Retriever."""

    documents: list[Document] = []

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.documents


@pytest.fixture
def mock_retriever():
    """Mock retriever with sample educational documents."""
    return MockRetriever(
        documents=[
            Document(
                page_content="PyTorch는 Facebook AI Research에서 개발한 오픈소스 머신러닝 프레임워크입니다. 텐서 연산과 자동 미분을 지원합니다.",
                metadata={
                    "doc_type": "pdf",
                    "course": "AI 기초",
                    "lecture_num": "3",
                    "topic": "PyTorch",
                },
            ),
            Document(
                page_content="CNN(Convolutional Neural Network)은 이미지 인식에 주로 사용되는 딥러닝 모델입니다.",
                metadata={
                    "doc_type": "lecture_transcript",
                    "course": "CV 이론",
                    "lecture_num": "5",
                    "topic": "CNN",
                },
            ),
            Document(
                page_content="Transformer는 Self-Attention 메커니즘을 기반으로 한 모델 아키텍처입니다.",
                metadata={
                    "doc_type": "notebook",
                    "course": "NLP 기초",
                    "topic": "Transformer",
                },
            ),
        ]
    )


@pytest.fixture
def empty_retriever():
    """Empty mock retriever for testing RAG context insufficient scenarios."""
    return MockRetriever(documents=[])


@pytest.fixture
def gemini_llm():
    """Gemini LLM instance for integration tests. Skips if no API key."""
    import os

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    from naver_connect_chatbot.config import get_chat_model

    return get_chat_model(thinking_level="low")


@pytest.fixture
def gemini_reasoning_llm():
    """Gemini LLM with default thinking (high) for answer generation."""
    import os

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    from naver_connect_chatbot.config import get_chat_model

    return get_chat_model()


# ============================================================================
# Dual Retriever Test Fixtures
# ============================================================================


@pytest.fixture
def mock_retriever_with_source():
    """source_type이 태깅된 MockRetriever (Dual Retriever 테스트용)."""
    return MockRetriever(
        documents=[
            Document(
                page_content="PyTorch는 자동 미분을 지원하는 딥러닝 프레임워크입니다.",
                metadata={
                    "doc_type": "pdf",
                    "course": "AI 기초",
                    "source_type": "qdrant",
                },
            ),
            Document(
                page_content="텐서(Tensor)는 다차원 배열로 PyTorch의 기본 데이터 구조입니다.",
                metadata={
                    "doc_type": "notebook",
                    "course": "AI 기초",
                    "source_type": "qdrant",
                },
            ),
            Document(
                page_content="CNN은 Convolution 연산을 사용하는 신경망 아키텍처입니다.",
                metadata={
                    "doc_type": "lecture_transcript",
                    "course": "CV 이론",
                    "source_type": "qdrant",
                },
            ),
        ]
    )


@pytest.fixture
def gemini_llm_settings():
    """GeminiLLMSettings 인스턴스. GOOGLE_API_KEY 없으면 skip."""
    import os

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    from naver_connect_chatbot.config import settings

    return settings.gemini_llm


@pytest.fixture
def reranker():
    """ClovaStudio Reranker 인스턴스. CLOVASTUDIO_API_KEY 없으면 skip."""
    import os

    if not os.getenv("CLOVASTUDIO_API_KEY"):
        pytest.skip("CLOVASTUDIO_API_KEY not set")
    from naver_connect_chatbot.config import settings
    from naver_connect_chatbot.rag.rerank import ClovaStudioReranker

    return ClovaStudioReranker.from_settings(settings.reranker)
