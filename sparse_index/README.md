# Sparse Index 디렉토리

이 디렉토리는 Sparse Retriever (BM25)의 인덱스 파일을 저장합니다.

## 파일 구조

```
sparse_index/
├── README.md                    # 이 파일
├── bm25_slack_qa.pkl           # BM25 인덱스 (4,581개 문서)
└── kiwi_bm25_slack_qa/         # KiwiBM25Retriever 인덱스 (향후)
    ├── bm25_index.pkl
    └── user_dict.txt
```

## BM25 인덱스 정보

### bm25_slack_qa.pkl

- **문서 수**: 4,581개
- **데이터 소스**: Slack Q&A (부스트캠프)
- **형태소 분석**: Kiwi
- **생성 도구**: document_processing/bm25_indexer.py

### 데이터 구조

```python
{
    "bm25_question": BM25Okapi,    # 질문 인덱스
    "bm25_answer": BM25Okapi,      # 답변 인덱스
    "bm25_combined": BM25Okapi,    # 통합 인덱스
    "doc_ids": List[str],          # 문서 ID 리스트
    "documents": List[dict],        # 원본 문서 (메타데이터 포함)
}
```

## 인덱스 생성 방법

### 방법 1: 기존 인덱스 복사 (권장)

```bash
# document_chunks에서 복사
cp document_chunks/bm25_index.pkl sparse_index/bm25_slack_qa.pkl
```

### 방법 2: 새로 생성

```bash
# document_processing의 BM25 인덱서 사용
cd document_processing
python bm25_indexer.py \
    --input-dir ../document_chunks/slack_qa_merged \
    --output ../sparse_index/bm25_slack_qa.pkl
```

## 사용 방법

### Python에서 로드

```python
import pickle
from pathlib import Path

# 인덱스 로드
index_path = Path("sparse_index/bm25_slack_qa.pkl")
with open(index_path, "rb") as f:
    bm25_data = pickle.load(f)

# 데이터 확인
print(f"문서 수: {len(bm25_data['documents'])}")
print(f"인덱스 타입: {type(bm25_data['bm25_question'])}")
```

### Hybrid Retriever에서 사용

```python
from naver_connect_chatbot.rag.retriever_factory import build_dense_sparse_hybrid
from naver_connect_chatbot.config.embedding import OpenRouterEmbeddings
from langchain_core.documents import Document
import pickle

# 1. BM25 인덱스 로드
with open("sparse_index/bm25_slack_qa.pkl", "rb") as f:
    bm25_data = pickle.load(f)

# 2. Document 변환
documents = [
    Document(
        page_content=f"질문: {doc['question_text']}\n답변: {doc['answer_text']}",
        metadata=doc
    )
    for doc in bm25_data["documents"]
]

# 3. Hybrid Retriever 구성
embeddings = OpenRouterEmbeddings()
hybrid_retriever = build_dense_sparse_hybrid(
    documents=documents,
    embedding_model=embeddings,
    qdrant_url="http://localhost:6333",
    collection_name="slack_qa",
)

# 4. 검색
results = hybrid_retriever.invoke("GPU 메모리 부족 해결 방법")
```

## 설정

### .env 파일

```bash
# BM25 인덱스 경로 (프로젝트 루트 기준)
RETRIEVER_BM25_INDEX_PATH=sparse_index/bm25_slack_qa.pkl
```

### 코드에서 접근

```python
from naver_connect_chatbot.config import settings

# 설정에서 경로 가져오기
index_path = settings.retriever.bm25_index_path
print(f"BM25 인덱스: {index_path}")
```

## 주의사항

1. **Git 추적 제외**: `.gitignore`에 의해 `.pkl` 파일은 추적되지 않습니다
2. **파일 크기**: 약 11MB (압축 가능)
3. **버전 관리**: 인덱스 버전은 문서 버전과 동기화 필요
4. **재생성**: 원본 데이터가 업데이트되면 인덱스 재생성 필요

## 트러블슈팅

### 파일이 없는 경우

```bash
# document_chunks에서 복사
cp document_chunks/bm25_index.pkl sparse_index/bm25_slack_qa.pkl

# 또는 새로 생성
python document_processing/bm25_indexer.py \
    --input-dir document_chunks/slack_qa_merged \
    --output sparse_index/bm25_slack_qa.pkl
```

### 인덱스 손상

```bash
# 백업이 있다면 복구
cp sparse_index/bm25_slack_qa.pkl.backup sparse_index/bm25_slack_qa.pkl

# 없다면 재생성
python document_processing/bm25_indexer.py --input-dir document_chunks/slack_qa_merged
```

## 향후 개선

- [ ] KiwiBM25Retriever 형식으로 마이그레이션
- [ ] 인덱스 버전 관리 시스템
- [ ] 자동 재생성 스크립트
- [ ] 압축 저장 지원

## 관련 문서

- [RETRIEVER_INTEGRATION_GUIDE.md](../docs/RETRIEVER_INTEGRATION_GUIDE.md)
- [document_processing/README.md](../document_processing/README.md)
- [document_processing/HYBRID_SEARCH_USAGE.md](../document_processing/HYBRID_SEARCH_USAGE.md)

**작성일**: 2025-11-20  
**버전**: 1.0.0

