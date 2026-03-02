# Prompts Module

## PURPOSE

YAML-based prompt templates for each workflow stage. Templates are loaded via `loader.py` and rendered through LangChain `ChatPromptTemplate`.

## TEMPLATE FILES

| Template | Used By | Purpose |
|----------|---------|---------|
| `intent_classification.yaml` | `intent_classifier.py` | Intent classification with domain relevance constraints |
| `query_analysis.yaml` | `query_analyzer.py` | Integrated quality scoring + multi-query generation + retrieval filters |
| `query_quality_analysis.yaml` | `query_analyzer.py` (split path) | Quality-only scoring |
| `query_expansion.yaml` | `query_analyzer.py` (split path) | Query expansion + retrieval filter extraction |
| `multi_query_generation.yaml` | `multi_query_retriever.py` | Diverse query generation for hybrid retrieval |
| `answer_generation_simple.yaml` | `answer_generator.py` | Concise grounded answer generation |
| `answer_generation_complex.yaml` | `answer_generator.py` | Stepwise grounded reasoning answer generation |
| `answer_generation_exploratory.yaml` | `answer_generator.py` | Exploratory guidance answer generation |

## CONVENTIONS

- Prompt text is written in English.
- Output language policy for answer prompts is Korean-only.
- Standard multi-turn input variable: `conversation_history`.
- Prefer task-centric structure in system prompts:
  - `## Core Task`
  - `## Critical Rules`
  - `## Grounding & Multi-turn Rules` (or equivalent split)
  - `## Output Contract`
  - `## Final Check`
- Do not use `## Role` or `## Inputs` sections.
- Put variable context blocks in human messages (Conversation History, Question, Context, Intent, etc.).
- For JSON outputs, describe key/type/range contracts in text; avoid raw JSON example blocks with literal braces.

## GOTCHAS

- Templates are not hot-reloaded; restart the server after prompt changes.
- `answer_generation_*.yaml` selection depends on `get_generation_strategy()`.
- `multi_query_generation.yaml` must stay compatible with both structured output and line-parser fallback.
- If template variables change, verify calling code provides compatible values.
