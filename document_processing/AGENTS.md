# Document Processing Module

## PURPOSE

Data ingestion pipeline: loads raw documents (PDFs, Jupyter notebooks, Slack Q&A, mission docs, lecture transcripts), chunks them, scores quality, and indexes into Qdrant + BM25 sparse index.

## KEY FILES

| File | Role |
|------|------|
| `ingest_all_to_vectordb.py` | Main ingestion orchestrator -- loads all document types, chunks, embeds, upserts to Qdrant |
| `rebuild_unified_bm25.py` | Rebuilds unified KiwiBM25 index from saved chunks (run after ingestion) |

## SUBMODULES

| Directory | Document Type | Key Classes |
|-----------|--------------|-------------|
| `pdf/` | PDF slides (lecture materials) | `pdf_loader.py`, `pdf_chunker.py`, image extraction |
| `notebooks/` | Jupyter notebooks (practice code) | `notebook_loader.py`, `notebook_chunker.py` |
| `mission/` | Weekly mission docs | `mission_loader.py`, `mission_chunker.py` |
| `lecture_transcript/` | Lecture transcripts | `lecture_transcript_chunker.py` |
| `slack_qa/` | Slack Q&A threads | Quality evaluation pipeline (see `slack_qa/AGENTS.md`) |
| `audit/` | Data quality auditing | Auditor framework (see `audit/AGENTS.md`) |
| `common/` | Shared utilities | `filters.py` (document filtering), `versioning.py` |
| `sparse_index/` | BM25 index output | `unified_bm25/` contains saved KiwiBM25 index |

## DATA FLOW

```
original_documents/ -> [loaders] -> [chunkers] -> document_chunks/
  -> ingest_all_to_vectordb.py -> Qdrant (dense vectors)
  -> rebuild_unified_bm25.py -> sparse_index/unified_bm25/ (BM25 index)
```

## DOCUMENT TYPES AND METADATA

All chunks carry metadata: `doc_type`, `course`, `course_topic`, `generation`, `source_file`. These fields enable pre-retriever filtering in the RAG workflow.

## CONVENTIONS

- Each submodule follows loader -> chunker -> `process_all_*.py` pattern
- Chunks saved as JSON in `document_chunks/` for reproducibility
- Loaders produce `langchain_core.documents.Document` objects
- `common/filters.py` provides deduplication and quality filters
