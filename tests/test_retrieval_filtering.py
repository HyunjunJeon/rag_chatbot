"""
Retrieval filtering 단위 테스트.

filter_documents_by_metadata 함수와 관련 로직을 테스트합니다:
- course: list[str] OR 조건 필터링
- Backward compatibility (str → list 변환)
- 다양한 필터 조합
- Edge cases
"""

import pytest
from unittest.mock import patch
from langchain_core.documents import Document

import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from naver_connect_chatbot.service.tool.retrieval_tool import (
    filter_documents_by_metadata,
    RetrievalResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """테스트용 샘플 문서들"""
    return [
        Document(
            page_content="CV 이론 강의 내용",
            metadata={"doc_type": "lecture_transcript", "course": "CV 이론"},
        ),
        Document(
            page_content="level2_cv 실습 코드",
            metadata={"doc_type": "notebook", "course": "level2_cv"},
        ),
        Document(
            page_content="Computer Vision PDF",
            metadata={"doc_type": "pdf", "course": "Computer Vision"},
        ),
        Document(
            page_content="NLP 강의 내용",
            metadata={"doc_type": "lecture_transcript", "course": "NLP 이론"},
        ),
        Document(
            page_content="PyTorch 슬랙 질문",
            metadata={"doc_type": "slack_qa", "course": "PyTorch"},
        ),
        Document(
            page_content="RecSys 미션",
            metadata={"doc_type": "weekly_mission", "course": "RecSys 이론"},
        ),
    ]


# ============================================================================
# Basic Filtering Tests
# ============================================================================


class TestBasicFiltering:
    """기본 필터링 테스트"""

    def test_no_filter_returns_all(self, sample_documents):
        """필터 없으면 모든 문서 반환"""
        result = filter_documents_by_metadata(sample_documents, {})
        assert len(result) == len(sample_documents)

    def test_none_filter_returns_all(self, sample_documents):
        """None 필터도 모든 문서 반환"""
        result = filter_documents_by_metadata(sample_documents, None)
        assert len(result) == len(sample_documents)

    def test_empty_doc_type_filter(self, sample_documents):
        """빈 doc_type 리스트는 필터링 안 함"""
        result = filter_documents_by_metadata(sample_documents, {"doc_type": []})
        assert len(result) == len(sample_documents)


# ============================================================================
# Course Filter Tests (OR Condition)
# ============================================================================


class TestCourseFilter:
    """course 필터 테스트 (OR 조건)"""

    def test_single_course_list(self, sample_documents):
        """단일 course 리스트 필터"""
        result = filter_documents_by_metadata(
            sample_documents, {"course": ["CV 이론"]}
        )
        assert len(result) == 1
        assert result[0].metadata["course"] == "CV 이론"

    def test_multiple_courses_or_condition(self, sample_documents):
        """다중 course OR 조건 필터"""
        result = filter_documents_by_metadata(
            sample_documents, {"course": ["CV 이론", "level2_cv", "Computer Vision"]}
        )
        # CV 관련 3개 문서
        assert len(result) == 3
        courses = {doc.metadata["course"] for doc in result}
        assert courses == {"CV 이론", "level2_cv", "Computer Vision"}

    def test_empty_course_list(self, sample_documents):
        """빈 course 리스트는 필터링 안 함"""
        result = filter_documents_by_metadata(sample_documents, {"course": []})
        assert len(result) == len(sample_documents)

    def test_no_match_returns_empty(self, sample_documents):
        """매칭 없으면 빈 리스트"""
        result = filter_documents_by_metadata(
            sample_documents, {"course": ["NonExistent"]}
        )
        assert len(result) == 0


class TestBackwardCompatibility:
    """str → list 하위 호환성 테스트"""

    def test_string_course_converted_to_list(self, sample_documents):
        """문자열 course가 리스트로 변환됨"""
        with patch(
            "naver_connect_chatbot.service.tool.retrieval_tool.logger"
        ) as mock_logger:
            result = filter_documents_by_metadata(
                sample_documents, {"course": "CV 이론"}  # str 타입
            )
            assert len(result) == 1
            assert result[0].metadata["course"] == "CV 이론"
            # Deprecation warning 로깅 확인 (각 문서마다 호출됨)
            assert mock_logger.warning.called, "Warning should be logged"
            # 첫 번째 호출의 메시지 확인
            assert "DEPRECATED" in mock_logger.warning.call_args_list[0][0][0]

    def test_string_course_single_match(self, sample_documents):
        """문자열 course도 정상 동작"""
        result = filter_documents_by_metadata(
            sample_documents, {"course": "PyTorch"}
        )
        assert len(result) == 1


# ============================================================================
# doc_type Filter Tests
# ============================================================================


class TestDocTypeFilter:
    """doc_type 필터 테스트"""

    def test_single_doc_type(self, sample_documents):
        """단일 doc_type 필터"""
        result = filter_documents_by_metadata(
            sample_documents, {"doc_type": ["lecture_transcript"]}
        )
        assert len(result) == 2  # CV 이론, NLP 이론

    def test_multiple_doc_types(self, sample_documents):
        """다중 doc_type 필터"""
        result = filter_documents_by_metadata(
            sample_documents, {"doc_type": ["slack_qa", "notebook"]}
        )
        assert len(result) == 2


# ============================================================================
# Combined Filter Tests
# ============================================================================


class TestCombinedFilters:
    """복합 필터 테스트"""

    def test_course_and_doc_type(self, sample_documents):
        """course + doc_type 조합"""
        result = filter_documents_by_metadata(
            sample_documents,
            {
                "course": ["CV 이론", "level2_cv"],
                "doc_type": ["lecture_transcript"],
            },
        )
        # CV 이론이면서 lecture_transcript인 것만
        assert len(result) == 1
        assert result[0].metadata["course"] == "CV 이론"

    def test_all_filters_no_match(self, sample_documents):
        """모든 필터 적용 시 매칭 없음"""
        result = filter_documents_by_metadata(
            sample_documents,
            {
                "course": ["CV 이론"],
                "doc_type": ["slack_qa"],  # CV 이론은 lecture_transcript
            },
        )
        assert len(result) == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_document_without_metadata(self):
        """메타데이터 없는 문서"""
        docs = [Document(page_content="No metadata")]
        result = filter_documents_by_metadata(docs, {"course": ["CV"]})
        assert len(result) == 0

    def test_document_with_partial_metadata(self):
        """부분 메타데이터 문서"""
        docs = [
            Document(page_content="Only course", metadata={"course": "CV"}),
            Document(page_content="Only doc_type", metadata={"doc_type": "pdf"}),
        ]
        result = filter_documents_by_metadata(docs, {"course": ["CV"]})
        assert len(result) == 1

    def test_none_metadata_value(self):
        """None 메타데이터 값"""
        docs = [
            Document(
                page_content="None course", metadata={"course": None, "doc_type": "pdf"}
            )
        ]
        result = filter_documents_by_metadata(docs, {"doc_type": ["pdf"]})
        assert len(result) == 1  # doc_type만 필터링

    def test_empty_documents_list(self):
        """빈 문서 리스트"""
        result = filter_documents_by_metadata([], {"course": ["CV"]})
        assert len(result) == 0


# ============================================================================
# String Filters Tests
# ============================================================================


class TestStringFilters:
    """문자열 필터 테스트 (course_topic, generation 등)"""

    def test_course_topic_filter(self):
        """course_topic 정확 일치 필터"""
        docs = [
            Document(
                page_content="PyTorch CNN",
                metadata={"course": "CV", "course_topic": "CNN"},
            ),
            Document(
                page_content="PyTorch RNN",
                metadata={"course": "NLP", "course_topic": "RNN"},
            ),
        ]
        result = filter_documents_by_metadata(docs, {"course_topic": "CNN"})
        assert len(result) == 1
        assert result[0].metadata["course_topic"] == "CNN"

    def test_generation_filter(self):
        """generation 필터"""
        docs = [
            Document(page_content="1기 자료", metadata={"generation": "1기"}),
            Document(page_content="2기 자료", metadata={"generation": "2기"}),
        ]
        result = filter_documents_by_metadata(docs, {"generation": "1기"})
        assert len(result) == 1
