"""
공통 유틸리티 모듈.

버전 관리, 해시 계산, 콘텐츠 필터링 등 공통 기능을 제공합니다.
"""

from .versioning import (
    SCHEMA_VERSION,
    PIPELINE_VERSION,
    compute_file_hash,
    get_current_timestamp,
    load_version_file,
    save_version_file,
    create_chunk_version_metadata,
    VersionInfo,
)

from .filters import (
    ContentFilter,
    FilterResult,
    contains_copyright,
    remove_copyright_notices,
    is_toc_page,
    remove_headers_footers,
    is_import_only_code,
    estimate_content_quality,
)

__all__ = [
    # versioning
    "SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "compute_file_hash",
    "get_current_timestamp",
    "load_version_file",
    "save_version_file",
    "create_chunk_version_metadata",
    "VersionInfo",
    # filters
    "ContentFilter",
    "FilterResult",
    "contains_copyright",
    "remove_copyright_notices",
    "is_toc_page",
    "remove_headers_footers",
    "is_import_only_code",
    "estimate_content_quality",
]
