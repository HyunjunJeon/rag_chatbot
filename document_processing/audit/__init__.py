"""
RAG 데이터 점검 모듈.

이 모듈은 RAG 시스템의 데이터 무결성과 품질을 점검합니다.

5개 레이어:
1. 원본 데이터 (PDF, Slack JSON, 강의 녹음)
2. 처리된 청크 (메타데이터 일관성, 청크 품질)
3. 인덱스 동기화 (BM25, Qdrant)
4. 품질 메트릭 (통계, 이상치 탐지)
5. 검색 성능 (샘플 쿼리 테스트)

사용법:
    python -m document_processing.audit.run_audit
    python -m document_processing.audit.run_audit --layer sources
    python -m document_processing.audit.run_audit --format json
"""

from document_processing.audit.models.audit_result import (
    AuditReport,
    Issue,
    LayerResult,
    Severity,
)

__all__ = [
    "AuditReport",
    "Issue",
    "LayerResult",
    "Severity",
]
