# Slack Q&A Processing

## PURPOSE

Processes raw Slack Q&A threads into scored, filtered document chunks for RAG ingestion. Includes LLM-based quality evaluation and batch processing.

## KEY FILES

| File | Role |
|------|------|
| `slack_qa_loader.py` | Loads raw Slack Q&A JSON from `original_documents/qa_dataset_from_slack/` |
| `quality_evaluator.py` | LLM-based Q&A quality scoring (relevance, completeness, clarity) |
| `quality_schemas.py` | Pydantic schemas for quality evaluation results |
| `batch_processor.py` | Batch processing of Q&A pairs with rate limiting and error handling |
| `evaluate_quality.py` | CLI entry point for quality evaluation |
| `filter_qa_data.py` | Filters Q&A by quality score thresholds |
| `export_scored_qa.py` | Exports scored Q&A to `document_chunks/slack_qa_scored/` |
| `merge_qa_by_course.py` | Merges Q&A data grouped by course |
| `review_scored_qa.py` | Manual review tool for scored Q&A |
| `process_all_slack_data.py` | End-to-end pipeline: load -> evaluate -> filter -> export |
| `prompts/` | Prompt templates for quality evaluation |

## PIPELINE

```
original_documents/qa_dataset_from_slack/{2..8}/ -> slack_qa_loader
  -> quality_evaluator (LLM scoring) -> filter_qa_data (threshold)
  -> export_scored_qa -> document_chunks/slack_qa_scored/
```

## GOTCHAS

- Slack Q&A is organized by generation (folders `2` through `8`)
- Quality evaluation uses LLM calls (costs money); results cached in scored JSON
- `batch_processor.py` handles API rate limits and retries for bulk evaluation
- Filtered output feeds into `ingest_all_to_vectordb.py` alongside other document types
