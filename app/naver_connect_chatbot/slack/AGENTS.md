# Slack Module

## PURPOSE

Slack Bolt async integration. Handles app mentions and DMs, preprocesses messages, manages rate limiting, and bridges Slack events to the LangGraph RAG workflow.

## KEY FILES

| File | Role |
|------|------|
| `__init__.py` | Exports `app` (AsyncApp instance) |
| `handler.py` | Event handlers, message preprocessing, agent initialization, rate limiting |

## REQUEST FLOW

```
Slack Event -> preprocess_slack_message() -> is_simple_greeting()?
  -> [greeting] -> random greeting response (no LLM call)
  -> [real question] -> rate limit check -> get_or_create_agent() -> graph.ainvoke() -> say()
```

## KEY BEHAVIORS

- **Socket Mode vs HTTP Mode**: auto-selected based on `SLACK_APP_TOKEN` presence
- **Agent lazy init**: `get_or_create_agent()` uses double-check locking with `asyncio.Lock`
- **Multi-turn threading**: Slack `thread_ts` maps to LangGraph `thread_id` for checkpointed conversations
- **Thread auto-reply**: bot responds without @mention in threads where it previously participated
- **Rate limiting**: 5 requests/minute per user (in-memory, resets on restart)
- **Timeout**: 120s per request via `asyncio.wait_for()`
- **Langfuse tracing**: callback attached per request with Slack metadata (user, channel, thread)

## GOTCHAS

- `_agent_app` is a global singleton; initialization failure sets `_agent_init_failed` permanently until restart
- `preprocess_slack_message()` strips `<@USERID>` mentions before passing to RAG pipeline
- Greeting detection is regex-based and runs BEFORE rate limiting (no cost for greetings)
- `set_checkpointer()` must be called from `server.py` lifespan before any requests
