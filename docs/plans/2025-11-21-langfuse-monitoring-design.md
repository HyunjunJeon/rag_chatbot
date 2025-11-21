# LangFuse Monitoring Integration Design

**Date**: 2025-11-21
**Status**: Approved
**Author**: Claude Code

## Overview

Integration of LangFuse v3 for comprehensive observability of the Adaptive RAG workflow, with node-level tracing, self-hosted deployment, and graceful degradation.

## Requirements

- **Deployment**: Self-hosted LangFuse v3 via Docker Compose
- **Trace Granularity**: Node-level tracing (detailed visibility into each agent)
- **Scope**: Observability only (no prompt management or datasets)
- **Resilience**: Graceful degradation if LangFuse is unavailable
- **Integration Point**: Centralized callback factory pattern

## Architecture

### Component Integration Points

1. **Infrastructure Layer (Docker Compose)**
   - Add 6 LangFuse containers: langfuse-web, langfuse-worker, postgres, clickhouse, redis, minio
   - Integrate with existing `chatbot-network`
   - Persist data to named volumes

2. **Configuration Layer (`config/monitoring.py`)**
   - `LangfuseSettings`: Pydantic settings with `.env.langfuse` support
   - `get_langfuse_callback()`: Factory function with health check
   - `check_langfuse_health()`: Async health verification

3. **Application Layer (`slack/handler.py`)**
   - Initialize callback per request with Slack metadata
   - Pass to `graph.ainvoke()` via RunnableConfig
   - Handle callback completion with `aflush()`

### Data Flow

```
Slack @mention
  → SlackHandler.handle_mention()
  → get_langfuse_callback(user_id, channel_id)
  → graph.ainvoke(config={"callbacks": [handler]})
  → Each node's LLM call auto-traced via callback propagation
  → handler.aflush() ensures trace completion
  → Traces viewable at http://localhost:3000
```

## Docker Compose Changes

### New Services

```yaml
services:
  langfuse-web:
    image: docker.io/langfuse/langfuse:3
    ports:
      - "3000:3000"
    depends_on:
      postgres: {condition: service_healthy}
      clickhouse: {condition: service_healthy}
      redis: {condition: service_healthy}
      minio: {condition: service_healthy}
    env_file: .env.langfuse
    networks:
      - chatbot-network
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/api/public/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  langfuse-worker:
    image: docker.io/langfuse/langfuse-worker:3
    depends_on:
      postgres: {condition: service_healthy}
      clickhouse: {condition: service_healthy}
    env_file: .env.langfuse
    networks:
      - chatbot-network

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_DB_PASSWORD}
    volumes:
      - langfuse_postgres:/var/lib/postgresql/data
    networks:
      - chatbot-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 3s
      retries: 5

  clickhouse:
    image: clickhouse/clickhouse-server:24-alpine
    environment:
      CLICKHOUSE_DB: langfuse
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: ${LANGFUSE_CLICKHOUSE_PASSWORD}
    volumes:
      - langfuse_clickhouse:/var/lib/clickhouse
    networks:
      - chatbot-network
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${LANGFUSE_REDIS_PASSWORD}
    networks:
      - chatbot-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9090"
    environment:
      MINIO_ROOT_USER: ${LANGFUSE_MINIO_USER}
      MINIO_ROOT_PASSWORD: ${LANGFUSE_MINIO_PASSWORD}
    volumes:
      - langfuse_minio:/data
    networks:
      - chatbot-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  langfuse_postgres:
  langfuse_clickhouse:
  langfuse_minio:
```

### App Service Changes

```yaml
app:
  depends_on:
    - qdrant
    - langfuse-web
  environment:
    LANGFUSE_HOST: http://langfuse-web:3000
```

## Configuration Implementation

### config/monitoring.py (New File)

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langfuse.langchain import CallbackHandler
from typing import Optional
import httpx
from loguru import logger


class LangfuseSettings(BaseSettings):
    """LangFuse monitoring configuration"""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=(".env", ".env.langfuse"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    host: str = Field(default="http://langfuse-web:3000")
    public_key: str = Field(default="")
    secret_key: str = Field(default="")
    enabled: bool = Field(default=True)
    health_check_timeout: float = Field(default=2.0)


async def check_langfuse_health(settings: LangfuseSettings) -> bool:
    """Check LangFuse server connectivity"""
    if not settings.enabled:
        return False

    try:
        async with httpx.AsyncClient(timeout=settings.health_check_timeout) as client:
            response = await client.get(f"{settings.host}/api/public/health")
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"LangFuse health check failed: {e}")
        return False


def get_langfuse_callback(
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    **metadata
) -> Optional[CallbackHandler]:
    """
    Create LangFuse callback handler with health check

    Args:
        user_id: Slack user ID for trace tagging
        channel_id: Slack channel ID for trace tagging
        **metadata: Additional metadata to attach to trace

    Returns:
        CallbackHandler if LangFuse is healthy and enabled, else None
    """
    from naver_connect_chatbot.config import settings

    langfuse_settings = settings.langfuse

    if not langfuse_settings.enabled:
        return None

    try:
        handler = CallbackHandler(
            public_key=langfuse_settings.public_key,
            secret_key=langfuse_settings.secret_key,
            host=langfuse_settings.host,
        )

        # Add Slack metadata
        if user_id:
            handler.set_trace_params(user_id=user_id)
        if channel_id or metadata:
            handler.set_trace_params(metadata={"channel_id": channel_id, **metadata})

        return handler
    except Exception as e:
        logger.error(f"Failed to create LangFuse callback: {e}")
        return None
```

### config/__init__.py Changes

```python
from pydantic_settings import BaseSettings
from naver_connect_chatbot.config.monitoring import LangfuseSettings

class Settings(BaseSettings):
    # Existing settings...
    langfuse: LangfuseSettings = LangfuseSettings()
```

## Slack Handler Integration

### slack/handler.py Modifications

```python
from naver_connect_chatbot.config.monitoring import get_langfuse_callback
from langchain_core.runnables import RunnableConfig

async def handle_mention_event(event: dict, say, client):
    """Handle Slack @mention events with LangFuse tracing"""

    # Extract Slack metadata
    user_id = event.get("user")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    question = event.get("text", "").strip()

    # Create LangFuse callback (returns None if unhealthy)
    langfuse_handler = get_langfuse_callback(
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
        event_type="slack_mention"
    )

    # Prepare callbacks (empty list if LangFuse unavailable)
    callbacks = [langfuse_handler] if langfuse_handler else []

    config = RunnableConfig(
        callbacks=callbacks,
        metadata={
            "source": "slack",
            "user_id": user_id,
            "channel_id": channel_id
        }
    )

    try:
        # Execute graph (callbacks auto-propagate to nodes)
        result = await graph.ainvoke(
            {"question": question, "max_retries": 2},
            config=config
        )

        # Send response
        await say(text=result["final_answer"], thread_ts=thread_ts)

        # Ensure trace completion (LangChain 0.3+ async callbacks)
        if langfuse_handler:
            await langfuse_handler.aflush()

    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        await say(text="죄송합니다. 오류가 발생했습니다.", thread_ts=thread_ts)
```

## Security & Secrets Management

### .env.langfuse (New File, Git-Ignored)

```bash
# Database
LANGFUSE_DB_PASSWORD=<generated-via-openssl-rand-hex-32>
DATABASE_URL=postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@postgres:5432/langfuse

# ClickHouse
LANGFUSE_CLICKHOUSE_PASSWORD=<generated-via-openssl-rand-hex-32>
CLICKHOUSE_URL=http://langfuse:${LANGFUSE_CLICKHOUSE_PASSWORD}@clickhouse:8123/langfuse

# Redis
LANGFUSE_REDIS_PASSWORD=<generated-via-openssl-rand-hex-32>
REDIS_CONNECTION_STRING=redis://:${LANGFUSE_REDIS_PASSWORD}@redis:6379

# MinIO
LANGFUSE_MINIO_USER=langfuse
LANGFUSE_MINIO_PASSWORD=<generated-via-openssl-rand-hex-32>
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY_ID=${LANGFUSE_MINIO_USER}
S3_SECRET_ACCESS_KEY=${LANGFUSE_MINIO_PASSWORD}
S3_BUCKET_NAME=langfuse

# Encryption & Auth
SALT=<generated-via-openssl-rand-hex-32>
ENCRYPTION_KEY=<generated-via-openssl-rand-hex-32>
NEXTAUTH_SECRET=<generated-via-openssl-rand-hex-32>
NEXTAUTH_URL=http://localhost:3000

# LangFuse API Keys (generated in UI after first login)
LANGFUSE_PUBLIC_KEY=<from-langfuse-ui>
LANGFUSE_SECRET_KEY=<from-langfuse-ui>
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_ENABLED=true
```

### .env.langfuse.example (Template for Git)

Provide template with `# CHANGEME` comments, actual values in gitignored `.env.langfuse`

### .gitignore Addition

```
.env.langfuse
```

## Testing Strategy

### 1. Unit Tests (LangFuse Disabled)

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_langfuse_disabled(monkeypatch):
    """Disable LangFuse for fast unit tests"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

@pytest.fixture
def mock_langfuse_callback(monkeypatch):
    """Mock LangFuse callback"""
    from naver_connect_chatbot.config import monitoring
    mock = Mock(spec=CallbackHandler)
    monkeypatch.setattr(monitoring, "get_langfuse_callback", lambda **kw: mock)
    return mock
```

### 2. Integration Tests (Real LangFuse)

```python
# tests/test_langfuse_integration.py
import pytest

@pytest.mark.integration
async def test_langfuse_trace_creation():
    """Verify trace creation in LangFuse"""
    from langfuse import Langfuse

    handler = get_langfuse_callback(user_id="test_user")
    result = await graph.ainvoke(
        {"question": "test question"},
        config={"callbacks": [handler]}
    )

    langfuse = Langfuse()
    traces = langfuse.get_traces(user_id="test_user", limit=1)
    assert len(traces) > 0
```

### 3. Health Check Tests

```python
async def test_health_check_timeout():
    """Verify graceful timeout handling"""
    settings = LangfuseSettings(
        host="http://nonexistent:3000",
        health_check_timeout=0.5
    )
    is_healthy = await check_langfuse_health(settings)
    assert is_healthy is False  # No exception raised
```

### 4. Local Development Verification

```bash
# Start full stack
docker-compose up -d

# Access UI
open http://localhost:3000

# Run integration tests
pytest -m integration -v
```

## Error Handling & Edge Cases

### Graceful Degradation

- **LangFuse Down**: `get_langfuse_callback()` returns `None`, graph executes normally
- **Health Check Timeout**: Caught, logged, returns `False`
- **Callback Creation Failure**: Caught, logged, returns `None`
- **Trace Flush Failure**: Logged, does not block response

### LangChain 0.3+ Callback Behavior

- Callbacks run in background by default
- Use `aflush()` to ensure completion before FastAPI response
- Alternative: Set `LANGCHAIN_CALLBACKS_BACKGROUND=false` (not recommended, blocks execution)

## Implementation Checklist

- [ ] Create `docs/plans/` directory
- [ ] Add LangFuse services to `docker-compose.yml`
- [ ] Create `.env.langfuse.example` template
- [ ] Generate secrets with `openssl rand -hex 32`
- [ ] Create `config/monitoring.py`
- [ ] Update `config/__init__.py` to include `LangfuseSettings`
- [ ] Add `langfuse` dependency to `pyproject.toml`
- [ ] Modify `slack/handler.py` to use callback factory
- [ ] Add `.env.langfuse` to `.gitignore`
- [ ] Create unit test fixtures in `tests/conftest.py`
- [ ] Create integration test in `tests/test_langfuse_integration.py`
- [ ] Create health check test in `tests/test_monitoring_config.py`
- [ ] Test Docker Compose startup
- [ ] Verify LangFuse UI access at `http://localhost:3000`
- [ ] Generate API keys in LangFuse UI
- [ ] Update `.env.langfuse` with real API keys
- [ ] Run integration tests
- [ ] Document access credentials in team password manager

## Success Criteria

1. ✅ LangFuse v3 starts successfully via `docker-compose up`
2. ✅ UI accessible at `http://localhost:3000`
3. ✅ Slack messages create traces with node-level spans
4. ✅ Traces include Slack metadata (user_id, channel_id)
5. ✅ Application works normally when LangFuse is disabled
6. ✅ Unit tests pass without LangFuse dependency
7. ✅ Integration tests verify trace creation

## Future Enhancements (Out of Scope)

- Prompt management via LangFuse UI
- Dataset creation from production traces
- User feedback tracking (Slack reactions)
- Cost analysis and dashboards
- A/B testing with LangFuse experiments
