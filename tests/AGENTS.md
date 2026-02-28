# Tests

## PURPOSE

pytest-asyncio test suite. Unit tests run locally without external services. Integration tests hit real APIs (CLOVA, Qdrant) and are gated behind `@pytest.mark.integration`.

## KEY FILES

| File | Scope |
|------|-------|
| `conftest.py` | Shared fixtures, pytest configuration |
| `test_clova_model.py` | CLOVA HCX-007 model connectivity and response format |
| `test_adaptive_rag_integration.py` | End-to-end RAG workflow integration |
| `test_retrieval_filtering.py` | Metadata-based retrieval filtering logic |
| `test_schema_registry.py` | SchemaRegistry singleton and course resolution |
| `test_checkpointer.py` | AsyncSqliteSaver checkpointing for multi-turn |
| `test_workflow_clarification.py` | Clarification node routing logic |
| `document_processing/` | Unit tests for batch processor, quality evaluator, quality schemas |
| `evaluation/` | LLM-judge RAG evaluation framework |

## EVALUATION FRAMEWORK (`evaluation/`)

| File | Role |
|------|------|
| `evaluators/llm_judge.py` | LLM-as-judge scoring (faithfulness, relevance, completeness) |
| `evaluators/schemas.py` | Pydantic schemas for evaluation metrics |
| `test_rag_evaluation.py` | RAG evaluation test runner |
| `test_rag_evaluation_v2.py` | V2 evaluation with improved metrics |
| `config/` | Evaluation configuration (test questions, expected answers) |
| `prompts/` | LLM judge prompt templates |
| `reports/` | Generated evaluation reports |

## CONVENTIONS

- All integration tests: `@pytest.mark.integration` (skipped without real API keys)
- Async tests: `@pytest.mark.asyncio` with `await`
- Run fast tests: `uv run pytest -k "not integration"`
- Run integration: `uv run pytest -m integration -v` (requires `.env` with API keys + Qdrant running)
- Fixtures should NOT modify `document_chunks/` or `vdb_store/` unless intentional
