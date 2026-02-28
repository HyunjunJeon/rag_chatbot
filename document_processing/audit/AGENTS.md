# Audit Module

## PURPOSE

Data quality auditing framework. Runs automated checks on document chunks, BM25 indices, Qdrant collections, and source documents. Generates HTML/JSON reports.

## KEY FILES

| File | Role |
|------|------|
| `run_audit.py` | CLI entry point: runs all auditors and generates reports |
| `auditors/base.py` | `BaseAuditor` abstract class -- defines audit interface |
| `auditors/chunk_auditor.py` | Validates chunk sizes, metadata completeness, content quality |
| `auditors/index_auditor.py` | Checks BM25 index consistency and coverage |
| `auditors/quality_auditor.py` | Evaluates content quality metrics across chunks |
| `auditors/search_auditor.py` | Tests retrieval quality with sample queries |
| `auditors/source_auditor.py` | Validates source document availability and integrity |
| `models/audit_result.py` | `AuditResult` Pydantic model for structured audit output |
| `reporters/html_reporter.py` | Generates HTML audit reports |
| `reporters/json_reporter.py` | Generates JSON audit reports |

## EXTENDING

Add new auditor: create class extending `BaseAuditor` in `auditors/`, implement `run()` method returning `AuditResult`. Register in `run_audit.py`.

## GOTCHAS

- `search_auditor.py` makes real retrieval calls -- needs running Qdrant instance
- Audit reports written to `document_processing/audit/` output directory
- Each auditor is independent; failures in one do not block others
