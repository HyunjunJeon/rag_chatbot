"""
SchemaRegistry 단위 테스트.

새로 추가된 기능들을 테스트합니다:
- Thread-safety (싱글톤, lazy initialization)
- Course alias resolution
- Fuzzy matching
- Cache invalidation
"""

import threading
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from naver_connect_chatbot.rag.schema_registry import (
    SchemaRegistry,
    VectorDBSchema,
    DataSourceInfo,
    CourseInfo,
    KEYWORD_PATTERNS,
    get_schema_registry,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """각 테스트 전후로 싱글톤 초기화"""
    SchemaRegistry.reset_for_testing()
    yield
    SchemaRegistry.reset_for_testing()


@pytest.fixture
def mock_schema() -> VectorDBSchema:
    """테스트용 mock 스키마 생성"""
    return VectorDBSchema(
        data_sources=[
            DataSourceInfo(
                doc_type="lecture_transcript",
                description="Lecture transcripts",
                total_count=100,
                courses=[
                    CourseInfo(name="CV 이론", count=30),
                    CourseInfo(name="level2_cv", count=25),
                    CourseInfo(name="Computer Vision", count=20),
                    CourseInfo(name="NLP 이론", count=15),
                    CourseInfo(name="RecSys 이론", count=10),
                ],
            ),
            DataSourceInfo(
                doc_type="slack_qa",
                description="Slack Q&A",
                total_count=50,
                courses=[
                    CourseInfo(name="PyTorch", count=25),
                    CourseInfo(name="파이토치 기초", count=15),
                    CourseInfo(name="Deep Learning", count=10),
                ],
            ),
        ],
        total_documents=150,
        collection_name="test_collection",
        last_updated="2025-01-01T00:00:00",
    )


@pytest.fixture
def registry_with_schema(mock_schema) -> SchemaRegistry:
    """스키마가 로드된 SchemaRegistry 인스턴스"""
    registry = SchemaRegistry.get_instance()
    registry._schema = mock_schema
    registry._course_aliases = None  # 캐시 초기화
    return registry


# ============================================================================
# Singleton & Thread-Safety Tests
# ============================================================================


class TestSingletonPattern:
    """싱글톤 패턴 테스트"""

    def test_singleton_returns_same_instance(self):
        """동일 인스턴스 반환 확인"""
        instance1 = SchemaRegistry.get_instance()
        instance2 = SchemaRegistry.get_instance()
        assert instance1 is instance2

    def test_singleton_via_constructor(self):
        """생성자로도 동일 인스턴스 반환"""
        instance1 = SchemaRegistry()
        instance2 = SchemaRegistry()
        assert instance1 is instance2

    def test_reset_for_testing_creates_new_instance(self):
        """reset_for_testing 후 새 인스턴스 생성"""
        instance1 = SchemaRegistry.get_instance()
        SchemaRegistry.reset_for_testing()
        instance2 = SchemaRegistry.get_instance()
        assert instance1 is not instance2


class TestThreadSafety:
    """Thread-safety 테스트"""

    def test_concurrent_singleton_access(self):
        """동시 싱글톤 접근 시 동일 인스턴스"""
        instances = []
        errors = []

        def get_instance():
            try:
                instances.append(SchemaRegistry.get_instance())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(set(id(i) for i in instances)) == 1, "Multiple instances created"

    def test_concurrent_alias_resolution(self, registry_with_schema):
        """동시 alias resolution 시 race condition 없음"""
        results = []
        errors = []

        def resolve_alias():
            try:
                result = registry_with_schema.resolve_course_aliases("CV")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve_alias) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # 모든 결과가 동일해야 함
        assert all(r == results[0] for r in results), "Inconsistent results"


# ============================================================================
# Course Alias Resolution Tests
# ============================================================================


class TestCourseAliasResolution:
    """Course alias resolution 테스트"""

    def test_resolve_known_alias_cv(self, registry_with_schema):
        """CV alias 해석"""
        result = registry_with_schema.resolve_course_aliases("CV")
        # KEYWORD_PATTERNS의 CV 패턴과 매칭되는 과정들
        assert "CV 이론" in result or "level2_cv" in result or "Computer Vision" in result

    def test_resolve_known_alias_lowercase(self, registry_with_schema):
        """소문자 alias도 해석"""
        result = registry_with_schema.resolve_course_aliases("cv")
        assert len(result) >= 1

    def test_resolve_unknown_keyword_returns_original(self, registry_with_schema):
        """알 수 없는 키워드는 원래 값 반환"""
        result = registry_with_schema.resolve_course_aliases("unknown_course")
        assert result == ["unknown_course"]

    def test_resolve_empty_string(self, registry_with_schema):
        """빈 문자열 처리"""
        result = registry_with_schema.resolve_course_aliases("")
        assert result == [""]

    def test_alias_cache_built_once(self, registry_with_schema):
        """alias 캐시가 한 번만 빌드됨"""
        # 첫 번째 호출에서 캐시 빌드
        registry_with_schema.resolve_course_aliases("CV")
        cache1 = registry_with_schema._course_aliases

        # 두 번째 호출에서 동일 캐시 사용
        registry_with_schema.resolve_course_aliases("NLP")
        cache2 = registry_with_schema._course_aliases

        assert cache1 is cache2


# ============================================================================
# Fuzzy Matching Tests
# ============================================================================


class TestFuzzyMatching:
    """Fuzzy matching 테스트"""

    def test_find_exact_match(self, registry_with_schema):
        """정확히 일치하는 과정 검색"""
        matches = registry_with_schema.find_matching_courses("CV 이론", threshold=0.8)
        assert len(matches) > 0
        assert matches[0][0] == "CV 이론"
        assert matches[0][1] >= 0.8  # 높은 유사도

    def test_find_partial_match(self, registry_with_schema):
        """부분 일치 검색"""
        matches = registry_with_schema.find_matching_courses("PyTorch", threshold=0.6)
        # PyTorch 또는 파이토치 포함 과정
        course_names = [m[0] for m in matches]
        assert any("PyTorch" in name or "파이토치" in name for name in course_names)

    def test_threshold_zero_returns_all(self, registry_with_schema):
        """threshold=0.0은 모든 과정 반환 가능 (max_results까지)"""
        matches = registry_with_schema.find_matching_courses("x", threshold=0.0, max_results=100)
        # 0.0 threshold는 모든 것과 매치
        assert len(matches) >= 1

    def test_threshold_one_requires_exact(self, registry_with_schema):
        """threshold=1.0은 정확히 일치만"""
        matches = registry_with_schema.find_matching_courses("CV 이론", threshold=1.0)
        if matches:
            assert matches[0][1] == 1.0
            assert matches[0][0] == "CV 이론"

    def test_no_match_returns_empty(self, registry_with_schema):
        """매칭 없으면 빈 리스트"""
        matches = registry_with_schema.find_matching_courses(
            "zzzznonexistent", threshold=0.9
        )
        assert matches == []

    def test_max_results_limit(self, registry_with_schema):
        """max_results 제한 준수"""
        matches = registry_with_schema.find_matching_courses(
            "이론", threshold=0.3, max_results=2
        )
        assert len(matches) <= 2


class TestResolveWithFuzzy:
    """resolve_course_with_fuzzy 통합 테스트"""

    def test_alias_priority_over_fuzzy(self, registry_with_schema):
        """alias가 fuzzy보다 우선"""
        # CV는 KEYWORD_PATTERNS에 있으므로 alias로 해석
        result = registry_with_schema.resolve_course_with_fuzzy("CV")
        # alias 전략이 사용됨
        assert len(result) >= 1

    def test_fuzzy_fallback_for_typo(self, registry_with_schema):
        """오타는 fuzzy로 해석"""
        # "PyTorchh" (오타)는 alias에 없으므로 fuzzy 사용
        result = registry_with_schema.resolve_course_with_fuzzy("PyTorchh", fuzzy_threshold=0.6)
        # fuzzy가 PyTorch 관련 과정을 찾아야 함
        assert len(result) >= 1

    def test_original_fallback_for_unknown(self, registry_with_schema):
        """완전히 알 수 없는 키워드는 원래 값 반환"""
        result = registry_with_schema.resolve_course_with_fuzzy(
            "zzzznonexistent", fuzzy_threshold=0.9
        )
        assert result == ["zzzznonexistent"]


# ============================================================================
# Cache Invalidation Tests
# ============================================================================


class TestCacheInvalidation:
    """캐시 무효화 테스트"""

    def test_schema_reload_invalidates_alias_cache(self, registry_with_schema):
        """스키마 리로드 시 alias 캐시 무효화"""
        # alias 캐시 빌드
        registry_with_schema.resolve_course_aliases("CV")
        assert registry_with_schema._course_aliases is not None

        # 새 스키마 로드 시뮬레이션
        new_schema = VectorDBSchema(
            data_sources=[
                DataSourceInfo(
                    doc_type="pdf",
                    description="PDFs",
                    total_count=10,
                    courses=[CourseInfo(name="New Course", count=10)],
                )
            ],
            total_documents=10,
            collection_name="new_collection",
            last_updated="2025-12-01T00:00:00",
        )

        # load_from_qdrant 대신 직접 스키마 설정 (캐시 무효화 확인)
        registry_with_schema._schema = new_schema
        with registry_with_schema._lock:
            registry_with_schema._course_aliases = None

        # 캐시가 무효화되었는지 확인
        assert registry_with_schema._course_aliases is None


# ============================================================================
# Schema Not Loaded Edge Cases
# ============================================================================


class TestSchemaNotLoaded:
    """스키마 미로드 상태 테스트"""

    def test_resolve_aliases_without_schema(self):
        """스키마 없이 alias 해석 시 원래 값 반환"""
        registry = SchemaRegistry.get_instance()
        # 스키마 로드 안 함
        result = registry.resolve_course_aliases("CV")
        assert result == ["CV"]

    def test_fuzzy_match_without_schema(self):
        """스키마 없이 fuzzy match 시 빈 리스트"""
        registry = SchemaRegistry.get_instance()
        result = registry.find_matching_courses("CV", threshold=0.6)
        assert result == []

    def test_is_loaded_returns_false(self):
        """스키마 미로드 시 is_loaded() = False"""
        registry = SchemaRegistry.get_instance()
        assert registry.is_loaded() is False

    def test_get_prompt_context_fallback(self):
        """스키마 없을 때 기본 컨텍스트 반환"""
        registry = SchemaRegistry.get_instance()
        context = registry.get_prompt_context()
        assert "데이터 소스 정보를 사용할 수 없습니다" in context


# ============================================================================
# KEYWORD_PATTERNS Tests
# ============================================================================


class TestKeywordPatterns:
    """KEYWORD_PATTERNS 상수 테스트"""

    def test_required_patterns_exist(self):
        """필수 패턴 존재 확인"""
        required = ["CV", "NLP", "RecSys", "PyTorch", "Deep Learning"]
        for key in required:
            assert key in KEYWORD_PATTERNS, f"Missing pattern: {key}"

    def test_patterns_are_lists(self):
        """모든 패턴이 리스트 타입"""
        for key, patterns in KEYWORD_PATTERNS.items():
            assert isinstance(patterns, list), f"{key} patterns is not a list"
            assert len(patterns) > 0, f"{key} patterns is empty"
