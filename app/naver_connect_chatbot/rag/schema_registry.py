"""
VectorDB 스키마 정보를 조회하고 캐시하는 레지스트리.

서버 시작 시 Qdrant에서 데이터 분포를 조회하여 캐시하고,
Query Analyzer 프롬프트에 데이터 소스 정보를 제공합니다.

사용 예시:
    ```python
    from naver_connect_chatbot.rag.schema_registry import SchemaRegistry
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    registry = SchemaRegistry.get_instance()
    schema = registry.load_from_qdrant(client, "naver_connect_docs")

    # 프롬프트에 주입할 텍스트 가져오기
    context = registry.get_prompt_context()
    ```
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from naver_connect_chatbot.config import logger


# doc_type별 설명 (한국어/영어 혼합)
DOC_TYPE_DESCRIPTIONS = {
    "slack_qa": "Slack Q&A conversations from Boost Camp students (슬랙 질의응답)",
    "pdf": "Lecture slides and handouts (강의 슬라이드 및 교안)",
    "notebook": "Practice code and Jupyter notebooks (실습 코드 및 노트북)",
    "lecture_transcript": "Lecture speech transcriptions (강의 녹취록)",
    "weekly_mission": "Weekly assignment problems and rubrics (주간 미션 및 루브릭)",
}

# 키워드 패턴 정의 - VectorDB 기반 alias 자동 생성에 사용
# 각 키워드에 대해 패턴을 포함하는 과정들을 그룹핑
KEYWORD_PATTERNS: dict[str, list[str]] = {
    "CV": ["CV", "cv", "Computer Vision", "컴퓨터비전"],
    "NLP": ["NLP", "nlp", "자연어"],
    "RecSys": ["RecSys", "recsys", "추천", "MLforRecSys"],
    "MRC": ["MRC", "mrc", "Machine Reading"],
    "PyTorch": ["PyTorch", "pytorch", "파이토치"],
    "Object Detection": ["Object Det", "객체 탐지", "object_det"],
    "Semantic Segmentation": ["Semantic Seg", "세그멘테이션", "segmentation"],
    "Data Engineering": ["Data Eng", "데이터 엔지니어링", "data_eng"],
    "AI Math": ["AI Math", "AI 수학", "수학"],
    "Deep Learning": ["Deep Learning", "딥러닝", "DL"],
}


@dataclass
class CourseInfo:
    """과정 정보"""

    name: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "count": self.count}


@dataclass
class DataSourceInfo:
    """단일 데이터 소스(doc_type) 정보"""

    doc_type: str
    description: str
    total_count: int
    courses: list[CourseInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "description": self.description,
            "total_count": self.total_count,
            "courses": [c.to_dict() for c in self.courses],
        }

    def to_prompt_text(self, max_courses: int = 10) -> str:
        """프롬프트에 삽입할 텍스트 형태로 변환"""
        lines = [
            f"### {self.doc_type} ({self.total_count:,} documents)",
            f"{self.description}",
            "Available courses:",
        ]

        # 상위 N개 과정만 표시
        for course in self.courses[:max_courses]:
            lines.append(f"- {course.name} ({course.count})")

        if len(self.courses) > max_courses:
            remaining = len(self.courses) - max_courses
            lines.append(f"- ... and {remaining} more courses")

        return "\n".join(lines)


@dataclass
class VectorDBSchema:
    """VectorDB 전체 스키마 정보"""

    data_sources: list[DataSourceInfo] = field(default_factory=list)
    total_documents: int = 0
    collection_name: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "total_documents": self.total_documents,
            "collection_name": self.collection_name,
            "last_updated": self.last_updated,
        }

    def to_prompt_text(self, max_courses_per_type: int = 10) -> str:
        """프롬프트에 삽입할 전체 데이터 소스 정보 텍스트"""
        lines = [
            "## Available Data Sources",
            f"Total: {self.total_documents:,} documents in {len(self.data_sources)} data sources",
            "",
        ]

        for i, ds in enumerate(self.data_sources, 1):
            lines.append(f"{i}. " + ds.to_prompt_text(max_courses_per_type))
            lines.append("")

        return "\n".join(lines)

    def get_valid_doc_types(self) -> list[str]:
        """유효한 doc_type 목록 반환"""
        return [ds.doc_type for ds in self.data_sources]

    def get_courses_for_doc_type(self, doc_type: str) -> list[str]:
        """특정 doc_type의 course 목록 반환"""
        for ds in self.data_sources:
            if ds.doc_type == doc_type:
                return [c.name for c in ds.courses]
        return []


class SchemaRegistry:
    """
    싱글톤 스키마 레지스트리.

    VectorDB 스키마 정보를 캐시하고 제공합니다.
    Thread-safe 싱글톤 패턴을 사용합니다.

    Thread-Safety:
        - 싱글톤 생성: double-checked locking with _lock
        - Lazy cache (_course_aliases): _lock으로 보호
        - 스키마 리로드 시 캐시 자동 무효화
    """

    _instance: SchemaRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._schema = None
                    instance._course_aliases = None  # 명시적 초기화
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """
        테스트를 위해 싱글톤 인스턴스를 초기화합니다.

        주의: 프로덕션 코드에서는 사용하지 마세요.
        """
        with cls._lock:
            cls._instance = None

    @classmethod
    def get_instance(cls) -> SchemaRegistry:
        """싱글톤 인스턴스 반환"""
        return cls()

    def load_from_qdrant(
        self,
        client: Any,  # QdrantClient
        collection_name: str,
        sample_size: int = 10000,
    ) -> VectorDBSchema:
        """
        Qdrant에서 스키마 정보를 로드합니다.

        Args:
            client: QdrantClient 인스턴스
            collection_name: 컬렉션 이름
            sample_size: 샘플링할 문서 수 (정확한 분포를 위해 충분히 크게)

        Returns:
            VectorDBSchema 인스턴스
        """
        logger.info(f"VectorDB 스키마 로드 중: {collection_name}")

        try:
            # 컬렉션 정보 조회
            collection_info = client.get_collection(collection_name)
            total_points = collection_info.points_count

            # 샘플 데이터 조회 (doc_type, course 필드만)
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=min(sample_size, total_points),
                with_payload={"include": ["doc_type", "course"]},
                with_vectors=False,
            )

            points = scroll_result[0]

            # doc_type별 course 분포 집계
            doc_type_courses: dict[str, dict[str, int]] = {}

            for point in points:
                payload = point.payload or {}
                doc_type = payload.get("doc_type", "unknown")
                course = payload.get("course", "unknown")

                if doc_type not in doc_type_courses:
                    doc_type_courses[doc_type] = {}

                if course not in doc_type_courses[doc_type]:
                    doc_type_courses[doc_type][course] = 0

                doc_type_courses[doc_type][course] += 1

            # DataSourceInfo 리스트 생성
            data_sources: list[DataSourceInfo] = []

            for doc_type, courses in sorted(
                doc_type_courses.items(),
                key=lambda x: sum(x[1].values()),
                reverse=True,
            ):
                # 과정별 정보 생성 (문서 수 내림차순 정렬)
                course_infos = [
                    CourseInfo(name=name, count=count)
                    for name, count in sorted(courses.items(), key=lambda x: -x[1])
                ]

                total_count = sum(courses.values())
                description = DOC_TYPE_DESCRIPTIONS.get(
                    doc_type, f"Documents of type {doc_type}"
                )

                data_sources.append(
                    DataSourceInfo(
                        doc_type=doc_type,
                        description=description,
                        total_count=total_count,
                        courses=course_infos,
                    )
                )

            # 스키마 생성 및 캐시
            # Note: total_documents는 실제 컬렉션의 points_count를 사용
            # (sample_size로 인해 스크롤된 데이터 수와 다를 수 있음)
            self._schema = VectorDBSchema(
                data_sources=data_sources,
                total_documents=total_points,  # 실제 컬렉션의 포인트 수 사용
                collection_name=collection_name,
                last_updated=datetime.now().isoformat(),
            )

            # 스키마 변경 시 alias 캐시 무효화 (thread-safe)
            with self._lock:
                self._course_aliases = None

            logger.info(
                f"VectorDB 스키마 로드 완료: "
                f"{self._schema.total_documents:,}개 문서, "
                f"{len(data_sources)}개 데이터 소스 "
                f"(alias 캐시 무효화됨)"
            )

            return self._schema

        except Exception as e:
            logger.error(f"VectorDB 스키마 로드 실패: {e}")
            # 빈 스키마 반환 (fallback)
            self._schema = VectorDBSchema(
                last_updated=datetime.now().isoformat(),
            )
            return self._schema

    def get_schema(self) -> VectorDBSchema | None:
        """캐시된 스키마 반환"""
        return self._schema

    def get_prompt_context(self, max_courses_per_type: int = 10) -> str:
        """
        프롬프트에 삽입할 데이터 소스 컨텍스트 텍스트를 반환합니다.

        스키마가 로드되지 않았으면 기본 텍스트를 반환합니다.

        Args:
            max_courses_per_type: 각 doc_type당 표시할 최대 과정 수

        Returns:
            프롬프트에 삽입할 텍스트
        """
        if self._schema is None or not self._schema.data_sources:
            return self._get_default_context()

        return self._schema.to_prompt_text(max_courses_per_type)

    def _get_default_context(self) -> str:
        """스키마가 없을 때 사용할 기본 컨텍스트"""
        return """## Available Data Sources
데이터 소스 정보를 사용할 수 없습니다.
일반적인 doc_type: slack_qa, pdf, notebook, lecture_transcript, weekly_mission
"""

    def get_valid_doc_types(self) -> list[str]:
        """유효한 doc_type 목록 반환"""
        if self._schema:
            return self._schema.get_valid_doc_types()
        return ["slack_qa", "pdf", "notebook", "lecture_transcript", "weekly_mission"]

    def is_loaded(self) -> bool:
        """스키마가 로드되었는지 확인"""
        return self._schema is not None and bool(self._schema.data_sources)

    def _build_course_aliases(self) -> dict[str, list[str]]:
        """
        VectorDB 스키마에서 과정 목록을 분석하여 자동으로 alias 매핑을 생성합니다.

        알고리즘:
        1. 모든 course 이름을 수집
        2. KEYWORD_PATTERNS의 각 키워드에 대해
        3. 패턴을 포함하는 과정들을 그룹핑

        Returns:
            키워드 → 과정 이름 목록 매핑 딕셔너리
        """
        if not self._schema:
            return {}

        # 모든 과정명 수집
        all_courses = [
            course.name
            for ds in self._schema.data_sources
            for course in ds.courses
        ]

        aliases: dict[str, list[str]] = {}

        for keyword, patterns in KEYWORD_PATTERNS.items():
            matched: set[str] = set()
            for course in all_courses:
                for pattern in patterns:
                    if pattern.lower() in course.lower():
                        matched.add(course)
                        break
            if matched:
                course_list = list(matched)
                aliases[keyword] = course_list
                aliases[keyword.lower()] = course_list

        logger.debug(f"Built {len(aliases)} course aliases from VectorDB schema")
        return aliases

    def resolve_course_aliases(self, keyword: str) -> list[str]:
        """
        별칭을 실제 과정 이름 목록으로 변환합니다 (동적 생성).

        Thread-safe lazy initialization을 사용합니다.

        Args:
            keyword: 사용자가 입력한 과정 키워드 (예: "CV", "nlp")

        Returns:
            매핑된 실제 과정 이름 목록.
            별칭에 없으면 입력값을 단일 리스트로 반환.

        Example:
            >>> registry.resolve_course_aliases("CV")
            ["CV 이론", "level2_cv", "Computer Vision"]
            >>> registry.resolve_course_aliases("unknown")
            ["unknown"]
        """
        # Thread-safe lazy initialization (double-checked locking)
        if self._course_aliases is None:
            with self._lock:
                if self._course_aliases is None:
                    self._course_aliases = self._build_course_aliases()

        if keyword in self._course_aliases:
            return self._course_aliases[keyword]
        if keyword.lower() in self._course_aliases:
            return self._course_aliases[keyword.lower()]
        return [keyword]

    def get_alias_context_for_prompt(self) -> str:
        """
        프롬프트에 주입할 별칭 정보를 반환합니다 (동적 생성).

        Thread-safe lazy initialization을 사용합니다.

        Returns:
            프롬프트에 삽입할 별칭 매핑 텍스트
        """
        # Thread-safe lazy initialization (double-checked locking)
        if self._course_aliases is None:
            with self._lock:
                if self._course_aliases is None:
                    self._course_aliases = self._build_course_aliases()

        if not self._course_aliases:
            return ""

        lines = ["## Course Aliases (VectorDB 기반 자동 생성)"]
        seen: set[str] = set()

        for alias, courses in self._course_aliases.items():
            # 대표 별칭만 표시 (대문자 시작 또는 전체 대문자)
            if alias not in seen and (alias == alias.upper() or alias[0].isupper()):
                # 최대 5개 과정만 표시
                display_courses = courses[:5]
                if len(courses) > 5:
                    display_courses.append(f"...외 {len(courses) - 5}개")
                lines.append(f"- \"{alias}\" → {display_courses}")
                seen.add(alias.lower())

        return "\n".join(lines)

    def find_matching_courses(
        self,
        query: str,
        threshold: float = 0.6,
        max_results: int = 5,
    ) -> list[tuple[str, float, str]]:
        """
        쿼리와 유사한 과정 이름을 퍼지 매칭으로 찾습니다.

        Args:
            query: 검색할 과정 이름 또는 키워드
            threshold: 최소 유사도 임계값 (0.0 ~ 1.0)
            max_results: 반환할 최대 결과 수

        Returns:
            (과정명, 유사도, doc_type) 튜플 목록 (유사도 내림차순)

        Example:
            >>> registry.find_matching_courses("CV 이론", threshold=0.6)
            [("CV 이론", 1.0, "pdf"), ("CV", 0.67, "slack_qa")]
        """
        if not self._schema or not self._schema.data_sources:
            return []

        query_lower = query.lower()
        matches: list[tuple[str, float, str]] = []

        for ds in self._schema.data_sources:
            for course in ds.courses:
                course_lower = course.name.lower()

                # SequenceMatcher로 유사도 계산
                similarity = SequenceMatcher(None, query_lower, course_lower).ratio()

                # 부분 문자열 매치 보너스
                if query_lower in course_lower or course_lower in query_lower:
                    similarity = max(similarity, 0.8)

                if similarity >= threshold:
                    matches.append((course.name, similarity, ds.doc_type))

        # 유사도 내림차순 정렬, 최대 결과 수 제한
        matches.sort(key=lambda m: -m[1])
        return matches[:max_results]

    def resolve_course_with_fuzzy(
        self,
        keyword: str,
        fuzzy_threshold: float = 0.6,
    ) -> list[str]:
        """
        별칭 매핑 + 퍼지 매칭을 결합하여 과정 이름을 해석합니다.

        우선순위:
        1. 별칭 매핑 (KEYWORD_PATTERNS 기반)
        2. 퍼지 매칭 (difflib.SequenceMatcher 기반)
        3. 원래 값 반환

        Args:
            keyword: 사용자 입력 키워드
            fuzzy_threshold: 퍼지 매칭 임계값

        Returns:
            해석된 과정 이름 목록

        Note:
            각 전략의 성공/실패가 로깅됩니다 (observability).
        """
        # 1. 별칭 매핑 먼저 시도
        aliased = self.resolve_course_aliases(keyword)
        if aliased != [keyword]:  # 별칭이 발견됨
            logger.info(
                f"Course resolution [ALIAS]: '{keyword}' → {aliased} "
                f"(strategy=alias, count={len(aliased)})"
            )
            return aliased

        # 2. 퍼지 매칭 시도
        fuzzy_matches = self.find_matching_courses(keyword, threshold=fuzzy_threshold)
        if fuzzy_matches:
            courses = [m[0] for m in fuzzy_matches]
            top_confidence = fuzzy_matches[0][1]
            logger.info(
                f"Course resolution [FUZZY]: '{keyword}' → {courses} "
                f"(strategy=fuzzy, confidence={top_confidence:.2f}, count={len(courses)})"
            )
            return courses

        # 3. 매칭 없으면 원래 값 반환
        logger.debug(
            f"Course resolution [ORIGINAL]: '{keyword}' → ['{keyword}'] "
            f"(strategy=original, no match found)"
        )
        return [keyword]


# 모듈 레벨 헬퍼 함수
def get_schema_registry() -> SchemaRegistry:
    """SchemaRegistry 싱글톤 인스턴스 반환"""
    return SchemaRegistry.get_instance()


def get_data_source_context(max_courses: int = 10) -> str:
    """프롬프트용 데이터 소스 컨텍스트 반환"""
    return get_schema_registry().get_prompt_context(max_courses_per_type=max_courses)
