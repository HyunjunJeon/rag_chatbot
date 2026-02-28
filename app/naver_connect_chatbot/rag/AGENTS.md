# RAG Module

## PURPOSE

Hybrid retrieval stack: KiwiBM25 (sparse, Korean morphology via Kiwi tokenizer) + Qdrant (dense, Naver BGE-M3 1024-dim embeddings), fused with RRF or Convex Combination. Post-retrieval reranking via Clova Studio Reranker.

## KEY FILES

| File | Role |
|------|------|
| `retriever_factory.py` | Factory functions: `build_dense_sparse_hybrid_from_saved()` (production), `build_advanced_hybrid_retriever()` (full stack) |
| `retriever/hybrid_retriever.py` | `HybridRetriever(BaseRetriever)` -- RRF/CC fusion of multiple retrievers |
| `retriever/kiwi_bm25_retriever.py` | `KiwiBM25Retriever` -- Korean BM25 with Kiwi tokenizer, save/load index support |
| `retriever/qdrant_sdk_retriever.py` | `QdrantVDBRetriever` -- dense vector search via Qdrant SDK |
| `retriever/multi_query_retriever.py` | `MultiQueryRetriever` -- LLM-based query expansion wrapping base retriever |
| `rerank.py` | `ClovaStudioReranker` -- Clova Studio reranking API integration |
| `segmentation.py` | `ClovaStudioSegmenter` -- document paragraph splitting |
| `summarization.py` | `ClovaStudioSummarizer` -- document summarization |
| `rag_reasoning.py` | `ClovaStudioRAGReasoning` -- function-calling RAG (legacy, not in current workflow) |
| `schema_registry.py` | `SchemaRegistry` -- singleton caching VectorDB schema (data sources, courses, aliases) for pre-retriever filtering |
| `tools.py` | LangChain tool definitions for retrieval |

## RETRIEVAL PIPELINE

```
Query -> [MultiQueryRetriever (LLM expansion)] -> [KiwiBM25 (sparse)] + [Qdrant (dense)]
  -> HybridRetriever (RRF/CC fusion) -> ClovaStudioReranker -> Top-K documents
```

## EXTENSION POINTS

- Add new retriever: extend `BaseRetriever`, implement `_get_relevant_documents()` / `_aget_relevant_documents()`
- Add new fusion method: add enum to `HybridMethod`, implement in `HybridRetriever.hybrid_results()`
- Adjust weights/k: via `RetrieverSettings` in `config/settings/retriever.py`

## GOTCHAS

- `KiwiBM25Retriever.load()` requires `load_user_dict=True` for Korean domain terms
- `HybridRetriever` validates CC weights sum to 1.0; RRF weights are normalized internally
- `SchemaRegistry` is a singleton; loaded once at server startup from Qdrant collection metadata
- Production always uses `build_dense_sparse_hybrid_from_saved()` (pre-built BM25 index), not `build_dense_sparse_hybrid()` (in-memory build from documents)
