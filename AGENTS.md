# Repository Guidelines

## Project Structure & Module Organization
- `app/naver_connect_chatbot`: main package with config/logging, LLM+r(eranker) setup (`config/`), FastAPI entrypoint (`server.py`), Slack handlers (`slack/`), RAG pieces (`rag/`), and prompts/services.
- `tests/`: unit + integration suites (see `tests/INTEGRATION.md`), plus manual scripts such as `compare_retrieval_results.py`.
- Docs live in `docs/`; data/indices in `document_chunks/`, `sparse_index/`, `vdb_store/`; runtime logs in `logs/`.
- Docker/orchestration: `Dockerfile`, `docker-compose.yml`, `Makefile`; env templates `.env.example`, `.env.langfuse.example`.

## Build, Test, and Development Commands
- Install deps: `uv sync` (or `uv sync --group dev` for tooling); relies on `uv.lock`.
- Run API: `uv run uvicorn naver_connect_chatbot.server:api --reload --port 8000` with a populated `.env`.
- Docker: `make up` / `make up-logs` start app + Qdrant + LangFuse, `make down` stops, `make health` pings `/health`.
- Tests: `uv run pytest` (fast), `uv run pytest -m integration -v` for real provider/Qdrant/LangFuse calls.
- Quality: `uv run ruff check .`, `uv run ruff format .`, and `uv run mypy app` before merging interface-heavy changes.

## Coding Style & Naming Conventions
- Python 3.13, 4-space indent, 100-char lines, double quotes (ruff).
- Prefer type hints; mypy is configured via `mypy.ini` and package is `py.typed`.
- Naming: functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; modules stay lowercase.
- Keep handlers small; rely on `get_logger()` for structured logs.

## Testing Guidelines
- Frameworks: pytest + pytest-asyncio; HTTP calls commonly mocked with `respx`.
- Tests live in `tests/` as `test_*.py`; await coroutines and add `pytest.mark.asyncio` when required.
- Mark real API invocations with `@pytest.mark.integration`; run only with provider keys and Docker services up.
- Keep fixtures lean; touch `document_chunks/` or vector stores only for intentional behavior shifts.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix(scope):`, `docs:`, `test:`); imperative mood.
- PRs: short summary, test evidence (`pytest ...`, `ruff check`), env prerequisites, and screenshots/log snippets for user-facing flows.
- Link issues or design docs; never commit secrets or bulky generated data (`logs/`, `vdb_store/`, backups).

## Security & Configuration Tips
- Copy `env.example` to `.env`; use `.env.langfuse` for the observability stack and rotate keys per README steps.
- Logs can include Slack metadata; redact before sharing. Keep API keys out of code, tests, and history. `make env` bootstraps local env files.
