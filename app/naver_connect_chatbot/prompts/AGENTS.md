# Prompts Module

## PURPOSE

YAML-based prompt templates for each workflow stage. Loaded via `loader.py` using Jinja2 rendering.

## TEMPLATE FILES

| Template | Used By | Purpose |
|----------|---------|---------|
| `intent_classification.yaml` | `intent_classifier.py` | Classify user intent (SIMPLE_QA, COMPLEX_REASONING, EXPLORATORY, OUT_OF_DOMAIN) |
| `query_analysis.yaml` | `query_analyzer.py` | Analyze query quality + extract metadata filters |
| `multi_query_generation.yaml` | `multi_query_retriever.py` | Generate diverse search queries from original |
| `query_expansion.yaml` | Query expansion variants | Alternative query reformulation |
| `query_quality_analysis.yaml` | Quality scoring | Evaluate query clarity/specificity/searchability |
| `answer_generation_simple.yaml` | `answer_generator.py` | Simple Q&A answer template |
| `answer_generation_complex.yaml` | `answer_generator.py` | Complex reasoning answer template |
| `answer_generation_exploratory.yaml` | `answer_generator.py` | Exploratory answer template |

## CONVENTIONS

- Templates use YAML format with Jinja2 placeholders (`{{ question }}`, `{{ context }}`)
- Load via `from naver_connect_chatbot.prompts import load_prompt`
- All prompts are in Korean (target audience: BoostCamp students)
- Answer templates instruct the model to think step-by-step and cite sources

## GOTCHAS

- Templates are NOT hot-reloaded; server restart required after changes
- `answer_generation_*.yaml` templates are selected by intent via `get_generation_strategy()`
- Current workflow bypasses templates for `generate_answer_node`, using inline prompt construction instead
