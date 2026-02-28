# Config Module

## PURPOSE

Centralized configuration via hierarchical Pydantic Settings. Provides LLM and embedding factory functions, structured logging, and Langfuse monitoring integration.

## KEY FILES

| File | Role |
|------|------|
| `__init__.py` | Public API: `settings`, `logger`, `get_chat_model()`, `get_embeddings()` |
| `llm.py` | `get_chat_model()` -- creates `ChatClovaX` (langchain_naver), supports `use_reasoning` flag |
| `embedding.py` | `get_embeddings()` -- creates `ClovaXEmbeddings` (BGE-M3, 1024 dims) |
| `log.py` | `get_logger()` -- structlog-based logging with JSON/console output |
| `monitoring.py` | `LangfuseSettings`, `get_langfuse_callback()` -- optional tracing |
| `settings/main.py` | `Settings` -- top-level settings aggregating all sub-settings |
| `settings/base.py` | `PROJECT_ROOT` path resolution |
| `settings/clova.py` | `ClovaXLLMSettings`, `ClovaXEmbeddingsSettings`, `ClovaStudioRerankerSettings` |
| `settings/retriever.py` | `RetrieverSettings`, `MultiQuerySettings`, `AdvancedHybridSettings` |
| `settings/vector_store.py` | `QdrantVectorStoreSettings` |
| `settings/slack.py` | `SlackSettings` -- bot token, signing secret, app token, port |
| `settings/rag_settings.py` | `AdaptiveRAGSettings` -- workflow-level tuning (reranking, clarification) |
| `settings/enums.py` | `RetrieverStrategy`, `HybridMethodType` enums |

## ENV VAR PREFIXES

Settings auto-load from env vars with these prefixes:
- `NAVER_CLOUD_*` -- CLOVA LLM, embeddings, reranker
- `QDRANT_*` -- vector store connection
- `SLACK_*` -- bot tokens and config
- `OPENAI_*` / `OPENROUTER_*` -- alternative LLM providers
- `LANGFUSE_*` -- observability
- `RETRIEVER_*` -- retrieval tuning

## GOTCHAS

- `get_chat_model(use_reasoning=True)` creates a separate LLM instance for answer generation; do NOT use with `with_structured_output()`
- `settings` is a module-level singleton; import via `from naver_connect_chatbot.config import settings`
- SecretStr fields (tokens, API keys) require `.get_secret_value()` to access
- `PROJECT_ROOT` resolves to the repository root, not the `app/` directory
