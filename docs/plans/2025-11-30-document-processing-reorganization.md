# Document Processing 디렉토리 재구성 계획

> **작성일**: 2025-11-30
> **상태**: 설계 완료, 구현 진행

## 1. 개요

### 1.1 목표
- `document_processing/` 디렉토리의 구조 일관성 확보
- Slack Q&A 처리 코드를 다른 모듈(pdf/, notebooks/, mission/)과 동일한 패턴으로 분리
- 레거시 코드 정리 (이미 완료됨)

### 1.2 배경
- 기존: Slack Q&A 관련 파일 4개가 루트에 산재
- 기존: `base.py` (PDF 유틸리티)가 루트에 위치
- 다른 모듈들(pdf/, notebooks/, mission/)은 이미 서브디렉토리로 잘 정리됨

## 2. 최종 디렉토리 구조

```
document_processing/
├── __init__.py
├── ingest_all_to_vectordb.py    # 통합 인덱싱 스크립트
├── rebuild_unified_bm25.py      # BM25 재생성 스크립트
│
├── common/                      # 공통 유틸리티
│   ├── __init__.py
│   ├── versioning.py
│   └── filters.py
│
├── pdf/                         # PDF 처리 모듈
│   ├── __init__.py
│   ├── base.py                  # ← 루트에서 이동
│   ├── pdf_loader.py
│   ├── pdf_chunker.py
│   └── process_all_pdfs.py
│
├── notebooks/                   # 노트북 처리 모듈
│   ├── __init__.py
│   ├── notebook_loader.py
│   ├── notebook_chunker.py
│   ├── process_all_notebooks.py
│   └── README.md
│
├── mission/                     # 미션 처리 모듈
│   ├── __init__.py
│   ├── mission_loader.py
│   ├── mission_chunker.py
│   └── process_all_missions.py
│
├── slack_qa/                    # ← 신규 생성
│   ├── __init__.py
│   ├── slack_qa_loader.py       # ← 루트에서 이동
│   ├── process_all_slack_data.py
│   ├── merge_qa_by_course.py
│   └── filter_qa_data.py
│
└── sparse_index/                # 인덱스 저장소
    └── unified_bm25/
```

## 3. 변경 작업 목록

### 3.1 디렉토리 생성
- [ ] `slack_qa/` 디렉토리 생성
- [ ] `slack_qa/__init__.py` 생성

### 3.2 파일 이동
| 원본 | 대상 |
|------|------|
| `base.py` | `pdf/base.py` |
| `slack_qa_loader.py` | `slack_qa/slack_qa_loader.py` |
| `process_all_slack_data.py` | `slack_qa/process_all_slack_data.py` |
| `merge_qa_by_course.py` | `slack_qa/merge_qa_by_course.py` |
| `filter_qa_data.py` | `slack_qa/filter_qa_data.py` |

### 3.3 Import 경로 업데이트
파일 이동 후 다음 import 경로 업데이트 필요:
- `ingest_all_to_vectordb.py` - Slack Q&A loader import
- `pdf/pdf_loader.py` - base.py import (상대 경로로 변경)
- `pdf/pdf_chunker.py` - base.py import (상대 경로로 변경)
- 기타 base.py를 참조하는 파일

### 3.4 __init__.py 업데이트
- `slack_qa/__init__.py` - 주요 클래스/함수 export
- `pdf/__init__.py` - base.py 함수 export 추가

## 4. 검증 계획

### 4.1 Import 검증
```bash
uv run python -c "from document_processing.slack_qa import SlackQALoader"
uv run python -c "from document_processing.pdf import parse_pdf"
```

### 4.2 기능 검증
```bash
# ingest_all_to_vectordb.py가 정상 동작하는지 확인 (dry-run)
uv run python document_processing/ingest_all_to_vectordb.py --help
```

### 4.3 Lint 검증
```bash
uv run ruff check document_processing/
```

## 5. 롤백 계획

Git으로 관리되므로 문제 발생 시:
```bash
git checkout -- document_processing/
```

## 6. 삭제된 레거시 파일 (이미 완료)

- `hybrid_retriever.py` - app/.../rag/retriever/와 중복
- `ingest_to_vectordb.py` - ingest_all_to_vectordb.py로 대체
- `bm25_indexer.py` - KiwiBM25Retriever로 대체
- `rebuild_bm25_for_chatbot.py` - rebuild_unified_bm25.py로 대체
