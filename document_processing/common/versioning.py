"""
버전 관리 유틸리티.

청크 데이터의 버전 추적, 파일 해시 계산, 증분 업데이트 지원을 위한 유틸리티입니다.

사용 예:
    ```python
    from document_processing.common.versioning import (
        compute_file_hash,
        get_current_timestamp,
        create_chunk_version_metadata,
        SCHEMA_VERSION,
        PIPELINE_VERSION,
    )

    # 파일 해시 계산
    file_hash = compute_file_hash(Path("notebook.ipynb"))

    # 청크 메타데이터에 버전 정보 추가
    version_meta = create_chunk_version_metadata(
        source_file=Path("notebook.ipynb"),
    )
    chunk_metadata.update(version_meta)
    ```
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# 버전 상수
# =============================================================================

SCHEMA_VERSION = "2.0.0"
"""
청크 스키마 버전.

변경 시점:
- 메타데이터 필드 추가/제거/변경
- 청크 ID 형식 변경
- 콘텐츠 구조 변경

변경 로그:
- 2.0.0: 라인리지 필드 추가 (corpus_version, pipeline_trace), 요약 필드 추가
- 1.0.0: 초기 버전
"""

PIPELINE_VERSION = "2.0.0"
"""
처리 파이프라인 버전.

변경 시점:
- 청킹 로직 변경
- 필터링 규칙 변경
- 텍스트 정제 로직 변경

변경 로그:
- 2.0.0: 소스 클린업 규칙 추가, 메타 라우팅 지원
- 1.0.0: 초기 버전
"""

CORPUS_VERSION = "2025.11.27"
"""
코퍼스 전체 버전 (날짜 기반).

변경 시점:
- 새로운 데이터 소스 추가
- 대규모 재인제스트
- 필터링/청킹 로직 변경으로 인한 전체 재처리

형식: YYYY.MM.DD (또는 YYYY.MM.DD.N for 같은 날 여러 버전)
"""

# 요약 모델 기본값
DEFAULT_SUMMARY_MODEL = "clova-hcx-003"
DEFAULT_SUMMARY_MAX_LENGTH = 200


# =============================================================================
# 유틸리티 함수
# =============================================================================


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    파일의 해시값을 계산합니다.

    Args:
        file_path: 해시를 계산할 파일 경로
        algorithm: 해시 알고리즘 (기본: sha256)

    Returns:
        "algorithm:hash_value" 형식의 문자열

    예시:
        >>> compute_file_hash(Path("notebook.ipynb"))
        "sha256:abc123def456..."
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # 큰 파일도 처리할 수 있도록 청크 단위로 읽기
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return f"{algorithm}:{hash_func.hexdigest()}"


def get_current_timestamp() -> str:
    """
    현재 시간을 ISO 8601 형식으로 반환합니다.

    Returns:
        ISO 8601 형식의 UTC 타임스탬프

    예시:
        >>> get_current_timestamp()
        "2025-11-27T00:00:00Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_chunk_version_metadata(
    source_file: Path | None = None,
    include_hash: bool = True,
    pipeline_trace: list[str] | None = None,
) -> dict[str, Any]:
    """
    청크에 포함할 버전 메타데이터를 생성합니다.

    Args:
        source_file: 원본 파일 경로 (해시 계산용)
        include_hash: 파일 해시 포함 여부
        pipeline_trace: 파이프라인 추적 정보 (예: ["loaded", "filtered", "chunked"])

    Returns:
        버전 메타데이터 딕셔너리

    예시:
        >>> meta = create_chunk_version_metadata(
        ...     Path("notebook.ipynb"),
        ...     pipeline_trace=["loaded", "filtered_v2", "chunked"]
        ... )
        >>> meta
        {
            "schema_version": "2.0.0",
            "pipeline_version": "2.0.0",
            "corpus_version": "2025.11.27",
            "processed_at": "2025-11-27T00:00:00Z",
            "source_hash": "sha256:abc123...",
            "pipeline_trace": ["loaded", "filtered_v2", "chunked"]
        }
    """
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "corpus_version": CORPUS_VERSION,
        "processed_at": get_current_timestamp(),
    }

    if include_hash and source_file and source_file.exists():
        metadata["source_hash"] = compute_file_hash(source_file)

    if pipeline_trace:
        metadata["pipeline_trace"] = pipeline_trace

    return metadata


# =============================================================================
# 요약 캐시 스키마
# =============================================================================


@dataclass
class SummaryMetadata:
    """
    요약 캐시 메타데이터.

    청크에 요약이 추가될 때 함께 저장되는 메타데이터입니다.

    Attributes:
        summary: 생성된 요약 텍스트
        summary_model: 요약 생성에 사용된 모델
        summary_model_version: 모델 버전
        summary_created_at: 요약 생성 시간
        summary_max_length: 요약 최대 길이 제한
        action_summary: (Slack용) 핵심 조치/해결책 요약
    """

    summary: str = ""
    summary_model: str = DEFAULT_SUMMARY_MODEL
    summary_model_version: str = "1.0"
    summary_created_at: str = field(default_factory=get_current_timestamp)
    summary_max_length: int = DEFAULT_SUMMARY_MAX_LENGTH
    action_summary: str | None = None  # Slack Q&A용

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        result = {
            "summary": self.summary,
            "summary_model": self.summary_model,
            "summary_model_version": self.summary_model_version,
            "summary_created_at": self.summary_created_at,
            "summary_max_length": self.summary_max_length,
        }
        if self.action_summary:
            result["action_summary"] = self.action_summary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryMetadata":
        """딕셔너리에서 생성."""
        return cls(
            summary=data.get("summary", ""),
            summary_model=data.get("summary_model", DEFAULT_SUMMARY_MODEL),
            summary_model_version=data.get("summary_model_version", "1.0"),
            summary_created_at=data.get("summary_created_at", get_current_timestamp()),
            summary_max_length=data.get("summary_max_length", DEFAULT_SUMMARY_MAX_LENGTH),
            action_summary=data.get("action_summary"),
        )


def create_summary_metadata(
    summary: str,
    model: str = DEFAULT_SUMMARY_MODEL,
    action_summary: str | None = None,
) -> dict[str, Any]:
    """
    요약 메타데이터를 생성합니다.

    Args:
        summary: 생성된 요약 텍스트
        model: 사용된 모델
        action_summary: (선택) 핵심 조치 요약 (Slack Q&A용)

    Returns:
        요약 메타데이터 딕셔너리

    예시:
        >>> meta = create_summary_metadata(
        ...     summary="PyTorch 텐서 생성 및 기본 연산 설명",
        ...     model="clova-hcx-003",
        ... )
    """
    metadata = SummaryMetadata(
        summary=summary,
        summary_model=model,
        action_summary=action_summary,
    )
    return metadata.to_dict()


def has_valid_summary(chunk_metadata: dict[str, Any]) -> bool:
    """
    청크에 유효한 요약이 있는지 확인합니다.

    캐시된 요약이 있고, 현재 스키마 버전과 호환되면 True.

    Args:
        chunk_metadata: 청크 메타데이터

    Returns:
        유효한 요약 존재 여부
    """
    if not chunk_metadata.get("summary"):
        return False

    # 스키마 버전 호환성 체크
    chunk_schema = chunk_metadata.get("schema_version", "1.0.0")
    major_version = int(chunk_schema.split(".")[0])

    # 현재 메이저 버전과 같으면 호환
    current_major = int(SCHEMA_VERSION.split(".")[0])
    return major_version == current_major


# =============================================================================
# 버전 파일 관리
# =============================================================================


@dataclass
class SourceFileInfo:
    """원본 파일 정보."""

    hash: str
    chunks: int
    processed_at: str

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)


@dataclass
class VersionInfo:
    """
    버전 정보 (_version.json 구조).

    Attributes:
        schema_version: 청크 스키마 버전
        pipeline_version: 처리 파이프라인 버전
        created_at: 최초 생성 시간
        updated_at: 마지막 업데이트 시간
        total_chunks: 전체 청크 수
        source_files: 원본 파일별 정보
    """

    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    created_at: str = field(default_factory=get_current_timestamp)
    updated_at: str = field(default_factory=get_current_timestamp)
    total_chunks: int = 0
    source_files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_chunks": self.total_chunks,
            "source_files": self.source_files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """딕셔너리에서 생성."""
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            pipeline_version=data.get("pipeline_version", PIPELINE_VERSION),
            created_at=data.get("created_at", get_current_timestamp()),
            updated_at=data.get("updated_at", get_current_timestamp()),
            total_chunks=data.get("total_chunks", 0),
            source_files=data.get("source_files", {}),
        )

    def add_source_file(
        self,
        file_name: str,
        file_path: Path,
        chunk_count: int,
    ) -> None:
        """원본 파일 정보를 추가합니다."""
        self.source_files[file_name] = {
            "hash": compute_file_hash(file_path) if file_path.exists() else "",
            "chunks": chunk_count,
            "processed_at": get_current_timestamp(),
        }
        self.updated_at = get_current_timestamp()

    def get_changed_files(
        self,
        current_files: dict[str, Path],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        변경된 파일 목록을 반환합니다.

        Args:
            current_files: 현재 파일 목록 {파일명: 경로}

        Returns:
            (새 파일, 변경된 파일, 삭제된 파일) 튜플
        """
        new_files: list[str] = []
        changed_files: list[str] = []
        deleted_files: list[str] = []

        # 새 파일 및 변경된 파일 확인
        for file_name, file_path in current_files.items():
            if file_name not in self.source_files:
                new_files.append(file_name)
            else:
                current_hash = compute_file_hash(file_path)
                stored_hash = self.source_files[file_name].get("hash", "")
                if current_hash != stored_hash:
                    changed_files.append(file_name)

        # 삭제된 파일 확인
        for file_name in self.source_files:
            if file_name not in current_files:
                deleted_files.append(file_name)

        return new_files, changed_files, deleted_files


def load_version_file(version_path: Path) -> VersionInfo | None:
    """
    _version.json 파일을 로드합니다.

    Args:
        version_path: _version.json 파일 경로

    Returns:
        VersionInfo 객체 또는 파일이 없으면 None
    """
    if not version_path.exists():
        return None

    try:
        with open(version_path, encoding="utf-8") as f:
            data = json.load(f)
        return VersionInfo.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ 버전 파일 로드 실패: {e}")
        return None


def save_version_file(version_path: Path, version_info: VersionInfo) -> None:
    """
    _version.json 파일을 저장합니다.

    Args:
        version_path: 저장할 경로
        version_info: 버전 정보
    """
    version_info.updated_at = get_current_timestamp()

    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(version_info.to_dict(), f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python versioning.py <file_path>")
        print("\n예시:")
        print("  python versioning.py notebook.ipynb")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)

    print(f"📄 파일: {file_path}")
    print(f"🔒 해시: {compute_file_hash(file_path)}")
    print(f"🕐 시간: {get_current_timestamp()}")
    print(f"📋 스키마 버전: {SCHEMA_VERSION}")
    print(f"🔧 파이프라인 버전: {PIPELINE_VERSION}")
