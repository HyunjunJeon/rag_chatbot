# Repository Guidelines

## Project Structure & Module Organization
- `app/naver_connect_chatbot`: config/logging, Clova LLM+r(eranker) setup (`config/`), FastAPI
  entrypoint (`server.py`), Slack handlers (`slack/`), RAG modules (`rag/`), prompts/services.
- `document_processing/`: ingestion + indexing utilities (Slack Q&A loaders, BM25/Qdrant builders,
  notebooks) feeding `document_chunks/`, `sparse_index/`, and `vdb_store/`.
- `tests/`: unit + integration suites (integration marked with `@pytest.mark.integration`, e.g.
  `tests/test_clova_rag_reasoning.py`).
- Docs live in `docs/`; runtime logs in `logs/`; raw docs/assets in `original_documents/` and
  `images/`; Docker/orchestration: `Dockerfile`, `docker-compose.yml`, `Makefile`; env templates
  `.env.example`, `.env.langfuse.example`.

## Build, Test, and Development Commands
- Install deps: `uv sync` (or `uv sync --group dev` for tooling); relies on `uv.lock`.
- Run API: `uv run uvicorn naver_connect_chatbot.server:api --reload --port 8000` or
  `uv run python -m naver_connect_chatbot.server` with a populated `.env`.
- Docker: `make up` / `make up-logs` start app + Qdrant + LangFuse; `make down` stops; run
  `make health` to ping `/health`.
- Tests: `uv run pytest -k "not integration"` (fast/local),
  `uv run pytest -m integration -v` for real provider/Qdrant/LangFuse calls.
- Quality: `uv run ruff check .`, `uv run ruff format .`, and `uv run mypy app` before merging
  interface-heavy changes.

## Coding Style & Naming Conventions
- Python 3.13, 4-space indent, 100-char lines, double quotes (ruff).
- Prefer type hints; mypy is configured via `mypy.ini` and package is `py.typed`.
- Naming: functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; modules
  stay lowercase.
- Keep handlers small; rely on `get_logger()` for structured logs.

## Testing Guidelines
- Frameworks: pytest + pytest-asyncio; HTTP calls commonly mocked with `respx`.
- Tests live in `tests/` as `test_*.py`; await coroutines and add `pytest.mark.asyncio` when
  required.
- Mark real API invocations with `@pytest.mark.integration`; run only with provider keys and Docker
  services up.
- Keep fixtures lean; touch `document_chunks/` or vector stores only for intentional behavior
  shifts.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix(scope):`, `docs:`, `test:`); imperative mood.
- PRs: short summary, test evidence (`pytest ...`, `ruff check`), env prerequisites, and
  screenshots/log snippets for user-facing flows.
- Link issues or design docs; never commit secrets or bulky generated data (`logs/`, `vdb_store/`,
  backups).

## Security & Configuration Tips
- Copy `env.example` to `.env`; use `.env.langfuse` for the observability stack and rotate keys per
  README steps.
- Logs can include Slack metadata; redact before sharing. Keep API keys out of code, tests, and
  history. `make env` bootstraps local env files.
