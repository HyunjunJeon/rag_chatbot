# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-28

## OVERVIEW

Naver Connect Chatbot: production RAG system for Naver BoostCamp AI Tech Q&A. FastAPI + Slack Bolt frontend, LangGraph adaptive workflow, CLOVA HCX-007 Reasoning model, Qdrant dense + KiwiBM25 sparse hybrid retrieval with Clova reranker.

## STRUCTURE

```
app/naver_connect_chatbot/
  config/         # Pydantic Settings hierarchy, LLM/embedding factories, Langfuse monitoring
  prompts/        # Jinja2/YAML prompt templates for each workflow stage
  rag/            # Retriever stack (KiwiBM25, Qdrant, Hybrid, MultiQuery), reranker, segmentation
  service/        # LangGraph workflow: graph (state/nodes/routing), agents (intent/query/answer), tools
  slack/          # Slack Bolt async handlers, message preprocessing, rate limiting
  server.py       # FastAPI app with lifespan (checkpointer, BM25 auto-rebuild, schema registry)

document_processing/
  slack_qa/       # Slack Q&A ingestion: quality evaluation, batch processing, scoring
  audit/          # Data quality auditing: chunk/index/quality/search/source auditors, HTML/JSON reporters
  pdf/            # PDF loading, chunking, image extraction
  notebooks/      # Jupyter notebook loading and chunking
  mission/        # Mission document loading and chunking
  lecture_transcript/  # Lecture transcript chunking
  common/         # Shared utilities: filters, versioning
  ingest_all_to_vectordb.py   # Main ingestion pipeline
  rebuild_unified_bm25.py     # BM25 index rebuilder

tests/            # pytest-asyncio: unit + integration (marked @pytest.mark.integration)
  evaluation/     # LLM-judge RAG evaluation with custom schemas
  document_processing/  # Document processing unit tests
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change RAG workflow | `service/graph/workflow.py` | `build_adaptive_rag_graph()` assembles StateGraph |
| Add/modify workflow node | `service/graph/nodes.py` | Each node returns TypedDict update |
| Change retrieval strategy | `rag/retriever_factory.py` | Factory functions compose retriever stack |
| Modify Slack bot behavior | `slack/handler.py` | `handle_app_mention`, `handle_message` |
| Add new settings | `config/settings/*.py` | Pydantic Settings with env var prefixes |
| Change prompt templates | `prompts/templates/*.yaml` | YAML prompt files loaded by `prompts/loader.py` |
| Ingest new documents | `document_processing/ingest_all_to_vectordb.py` | Orchestrates all loaders |
| Rebuild BM25 index | `document_processing/rebuild_unified_bm25.py` | Run after document changes |
| Audit data quality | `document_processing/audit/run_audit.py` | Generates HTML/JSON reports |

## CONVENTIONS

- All nodes use `parse_agent_response()` with fallback defaults; errors handled inside nodes, never propagate to workflow level
- State uses `TypedDict(total=False)` -- every field optional
- Node return types are specific TypedDict updates (`IntentUpdate`, `QueryAnalysisUpdate`, etc.) defined in `service/graph/types.py`
- `partial()` binds dependencies (LLM, retriever) to node functions at graph build time
- Korean comments and docstrings throughout (project language context)
- Multi-turn: `messages` field uses `Annotated[Sequence[BaseMessage], operator.add]` for append semantics

## ANTI-PATTERNS

- NEVER raise exceptions from workflow nodes; always return fallback state
- NEVER call `llm.with_structured_output()` in Reasoning mode; use plain `llm.invoke(prompt)` for CLOVA HCX-007 compatibility
- NEVER commit `.env`, `vdb_store/`, `logs/`, `data/checkpoints.db`
- NEVER use `tools`/`function_calling` with CLOVA Reasoning model
- NEVER skip `@pytest.mark.integration` on tests that hit real APIs

## COMMANDS

```bash
# Install
uv sync --group dev

# Run server
python -m naver_connect_chatbot.server

# Docker
docker-compose up -d          # or: make up
make health                   # ping /health

# Test
uv run pytest -k "not integration"        # fast/local
uv run pytest -m integration -v           # real APIs (costs money)

# Quality
uv run ruff check . && uv run ruff format . && uv run mypy app
```
