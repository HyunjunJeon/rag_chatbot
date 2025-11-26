# 📓 Notebook Processing Module

Jupyter Notebook (.ipynb) 파일에서 RAG 시스템을 위한 의미있는 청크를 추출하는 모듈입니다.

## 개요

이 모듈은 `practice/` 및 `home_work/` 폴더의 노트북 파일을 분석하여:
- ✅ 마크다운 셀에서 개념 설명 추출
- ✅ 코드 셀에서 구현 예시 추출 (정답 파일만)
- ✅ 섹션 단위 청킹으로 컨텍스트 유지
- ✅ 메타데이터 자동 추출 (과목, 주제, 난이도)

## 모듈 구조

```
notebooks/
├── __init__.py              # 패키지 초기화
├── notebook_loader.py       # NotebookLoader 클래스
├── notebook_chunker.py      # NotebookChunker 클래스
├── process_all_notebooks.py # 일괄 처리 스크립트
└── README.md                # 이 문서
```

## 빠른 시작

### 1. 단일 노트북 처리

```python
from document_processing.notebooks import NotebookLoader, NotebookChunker

# 로더와 청커 초기화
loader = NotebookLoader()
chunker = NotebookChunker(max_tokens=500)

# 노트북 로드
notebook = loader.load_from_file("path/to/notebook.ipynb")

print(f"과목: {notebook.course}")
print(f"주제: {notebook.topic}")
print(f"타입: {notebook.file_type.value}")

# 청킹
chunks = chunker.chunk_notebook(notebook)

for chunk in chunks:
    print(f"[{chunk.id}] {chunk.content[:100]}...")
```

### 2. 전체 데이터셋 처리

```bash
cd document_processing/notebooks
python process_all_notebooks.py
```

### 3. 문제 파일 코드도 포함

```bash
python process_all_notebooks.py --include-problems
```

## 클래스 설명

### NotebookLoader

노트북 파일을 로드하고 파싱합니다.

```python
loader = NotebookLoader()

# 단일 파일
notebook = loader.load_from_file("notebook.ipynb")

# 디렉토리 전체 (재귀)
notebooks = loader.load_from_directory("path/to/dir", recursive=True)

# 정답 파일만
notebooks = loader.load_from_directory("path/to/dir", solution_only=True)
```

#### ParsedNotebook 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `file_path` | Path | 파일 경로 |
| `cells` | list[NotebookCell] | 파싱된 셀 리스트 |
| `file_type` | FileType | 문제/정답/알수없음 |
| `difficulty` | Difficulty | 기본/심화/알수없음 |
| `course` | str | 과목명 (예: "PyTorch") |
| `topic` | str | 주제명 (예: "Linear Regression") |

### NotebookChunker

노트북을 RAG용 청크로 분할합니다.

```python
chunker = NotebookChunker(
    max_tokens=500,      # 청크 최대 토큰 수
    min_tokens=50,       # 청크 최소 토큰 수
    include_outputs=True, # 코드 출력 포함
    max_output_lines=30,  # 출력 최대 라인 수
    solution_only=True,   # 정답 파일만 코드 포함
)

chunks = chunker.chunk_notebook(notebook)

# 여러 노트북 일괄 처리
all_chunks = chunker.chunk_notebooks(notebooks)
```

#### NotebookChunk 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `id` | str | 고유 식별자 |
| `content` | str | 청크 내용 |
| `metadata` | dict | 메타데이터 |
| `token_estimate` | int | 추정 토큰 수 |

## 청킹 전략

### 1. 섹션 기반 분할

H1, H2, H3 헤딩을 기준으로 노트북을 섹션으로 분할합니다.

```
## 1. 데이터 로드        ← 새 섹션 시작
설명 마크다운...
코드 셀...

## 2. 모델 정의          ← 새 섹션 시작
설명 마크다운...
코드 셀...
```

### 2. 셀 그룹화

관련된 마크다운, 코드, 출력을 하나의 청크로 묶습니다.

```
청크 1:
├── [Markdown] 섹션 제목 + 설명
├── [Code] 관련 코드
└── [Output] 실행 결과 (선택)
```

### 3. 토큰 제한

`max_tokens`를 초과하면 청크를 분할합니다.

### 4. 코드 필터링

문제 파일의 빈 코드는 제외합니다:
- `# TODO: 코드 작성`
- `pass`
- `...`
- import만 있는 셀

## 출력 형식

### 청크 JSON

```json
{
  "id": "pytorch_linear_regression_s01_c00_abc123",
  "content": "## Linear Regression\n\n선형 회귀 모델을...\n\n```python\nclass Model...",
  "metadata": {
    "source_file": "practice/01. AI Core/01. PyTorch/.../정답.ipynb",
    "course": "PyTorch",
    "topic": "Linear Regression",
    "difficulty": "기본",
    "file_type": "정답",
    "section_idx": 1,
    "chunk_idx": 0,
    "cell_range": [3, 7],
    "cell_types": ["markdown", "code"],
    "title": "PyTorch 기초 실습"
  },
  "token_estimate": 320
}
```

### 출력 디렉토리

```
document_chunks/notebook_chunks/
├── _summary.json              # 전체 통계
├── all_notebook_chunks.json   # 전체 청크
├── PyTorch_chunks.json        # 과목별 청크
├── AI_Math_chunks.json
├── ML_LifeCycle_chunks.json
└── MRC_chunks.json
```

## 메타데이터 추출

### 파일명 패턴

| 패턴 | 의미 |
|------|------|
| `(정답)`, `_정답`, `(해설)` | 정답 파일 |
| `(문제)`, `_문제` | 문제 파일 |
| `(기본-`, `기본_` | 기본 난이도 |
| `(심화-`, `심화_` | 심화 난이도 |

### 경로 기반 추출

```
practice/01. AI Core/01. PyTorch/(기본-2) Linear Regression/(정답).ipynb
         ↓          ↓                    ↓                      ↓
       (무시)     과목명              주제명                 파일타입
```

## 처리 대상

### practice/ (실습 자료)

| 과목 | 파일 수 | 내용 |
|------|--------|------|
| PyTorch | 8개 | Tensor, Linear Regression, Classification |
| AI Math | 6개 | 행렬, einsum, 확률론 |
| ML LifeCycle | 6개 | NumPy, Back Propagation, Self-Attention |

### home_work/ (과제)

| 과목 | 파일 수 | 내용 |
|------|--------|------|
| AI 개발 기초 | 1개 | Shell script (로그 처리) |
| MRC | 10개 | KorQuAD, TF-IDF, Dense Retrieval, FAISS |

## 다음 단계

청크가 생성되면 기존 파이프라인을 활용하여:

1. **BM25 인덱스 생성**
   ```bash
   python rebuild_bm25_for_chatbot.py --input-dir document_chunks/notebook_chunks
   ```

2. **Qdrant 적재**
   ```bash
   python ingest_to_vectordb.py --input-dir document_chunks/notebook_chunks
   ```

## 관련 문서

- [전체 처리 계획](../PROCESSING_PLAN.md)
- [Slack Q&A 처리](../README.md)
- [VectorDB 가이드](../VECTORDB_USAGE.md)
