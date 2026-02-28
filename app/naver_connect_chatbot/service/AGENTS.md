# Service Module (LangGraph Workflow)

## PURPOSE

Adaptive RAG workflow built with LangGraph `StateGraph`. Orchestrates intent classification, query analysis, retrieval, reranking, and answer generation through a directed graph of async nodes.

## KEY FILES

| File | Role |
|------|------|
| `graph/workflow.py` | `build_adaptive_rag_graph()` -- assembles and compiles the StateGraph |
| `graph/state.py` | `AdaptiveRAGState(TypedDict, total=False)` -- all workflow state fields |
| `graph/nodes.py` | Async node functions: `classify_intent_node`, `analyze_query_node`, `retrieve_node`, `rerank_node`, `generate_answer_node`, `generate_ood_response_node`, `clarify_node`, `finalize_node` |
| `graph/types.py` | Node return TypedDicts (`IntentUpdate`, `QueryAnalysisUpdate`, `RetrievalUpdate`, `AnswerUpdate`, `OODResponseUpdate`) and `RetrievalFilters` |
| `graph/routing.py` | Conditional edge functions |
| `agents/intent_classifier.py` | `aclassify_intent()` -- LLM-based intent classification (SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, OUT_OF_DOMAIN) |
| `agents/query_analyzer.py` | `aanalyze_query()` -- query quality scoring + multi-query generation + metadata filter extraction |
| `agents/answer_generator.py` | `get_generation_strategy()` -- strategy selection based on intent |
| `agents/response_parser.py` | `parse_agent_response()` -- type-safe parsing with fallback defaults |
| `tool/retrieval_tool.py` | `retrieve_documents_async()` -- retrieval with metadata filtering and empty-result fallback |

## WORKFLOW GRAPH

```
classify_intent -> [OOD?] -> generate_ood_response -> finalize -> END
                -> [In-domain] -> analyze_query -> [clarify?] -> clarify -> finalize -> END
                                                -> retrieve -> rerank -> generate_answer -> finalize -> END
```

## NODE CONTRACT

Every node:
1. Receives `AdaptiveRAGState` + bound dependencies (via `functools.partial`)
2. Returns a specific TypedDict update (NOT full state)
3. Catches all exceptions internally; returns fallback values on error
4. Uses `logger.info("---NODE_NAME---")` for tracing

## MULTI-TURN SUPPORT

- `messages` field uses `operator.add` annotation for append semantics
- Each node appends `HumanMessage`/`AIMessage` to maintain conversation history
- `_format_chat_history()` extracts last N turns for prompt context
- Thread ID from Slack `thread_ts` used as LangGraph `thread_id` for checkpointing

## GOTCHAS

- `generate_answer_node` uses plain `llm.ainvoke(prompt)`, NOT structured output -- required for CLOVA Reasoning mode
- Thinking effort (`low`/`medium`/`high`) is mapped from intent type
- OOD detection has two layers: pattern matching (fast, no LLM) + LLM classification (fallback)
- `domain_relevance < 0.3` overrides any intent to `OUT_OF_DOMAIN`
