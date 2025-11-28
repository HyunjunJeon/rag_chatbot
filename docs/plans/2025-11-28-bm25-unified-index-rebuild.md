# BM25 통합 인덱스 재생성 계획

> **작성일**: 2025-11-28
> **상태**: 계획 완료, 실행 대기

## 1. 개요

### 1.1 목표
- Qdrant VectorDB에서 전체 13,089개 문서를 추출하여 **Kiwi 기반 통합 BM25 인덱스** 생성
- VectorDB(Qdrant)의 데이터는 **읽기만** 수행, 절대 수정/삭제하지 않음
- 기존 Slack Q&A 전용 BM25 인덱스를 전체 데이터 통합 인덱스로 대체

### 1.2 배경
- 기존 BM25 인덱스(`sparse_index/bm25_slack_qa.pkl`)는 Slack Q&A 데이터만 포함
- Hybrid Retriever(Dense + Sparse)에서 BM25와 Qdrant가 같은 문서 집합을 가져야 RRF 융합 효과 극대화
- Qdrant에는 PDF, Notebook, Slack Q&A, Weekly Mission 등 전체 데이터가 이미 존재

### 1.3 현재 상태
| 항목 | 상태 |
|------|------|
| Qdrant Collection | `naver_connect_docs` - 13,089 points |
| 기존 BM25 | `sparse_index/bm25_slack_qa.pkl` (11MB, Slack Q&A만) |
| 설정 기본값 | `settings.retriever.bm25_index_path = "sparse_index/unified_bm25"` |

## 2. 데이터 분석

### 2.1 Qdrant Collection 구조
```
Collection: naver_connect_docs
Total Points: 13,089

doc_type 분포 (샘플 1,000개 기준):
  - pdf: 401 (40%)
  - notebook: 382 (38%)
  - slack_qa: 201 (20%)
  - weekly_mission: 16 (2%)
```

### 2.2 Payload 구조 (doc_type별)

#### Slack Q&A
```json
{
  "content": "...",
  "doc_id": "slack_3_level3_product_serving_...",
  "doc_type": "slack_qa",
  "course": "level3_product_serving",
  "generation": "3",
  "date": "2022-05-16",
  "question_text": "...",
  "answer_text": "...",
  "question_user": "...",
  "answer_user": "...",
  "is_bot": false
}
```

#### PDF
```json
{
  "content": "...",
  "doc_id": "pdf_cv_이론_lecture09_...",
  "doc_type": "pdf",
  "course": "CV 이론",
  "instructor": "오태현",
  "lecture_num": 9,
  "page_start": 0,
  "page_end": 0,
  "chunk_idx": 0,
  "total_chunks": 2
}
```

#### Notebook
```json
{
  "content": "...",
  "doc_id": "ai_core_ml_for_recsys_...",
  "doc_type": "notebook",
  "course": "AI Core",
  "topic": "ML for RecSys",
  "difficulty": "기본",
  "file_type": "정답",
  "keywords_list": ["ML for RecSys", ...]
}
```

#### Weekly Mission
```json
{
  "content": "...",
  "doc_id": "mission_nlp_이론_...",
  "doc_type": "weekly_mission",
  "course": "NLP 이론",
  "instructor": "주재걸",
  "week": "",
  "mission_name": "...",
  "chunk_type": "problem"
}
```

## 3. Kiwi 형태소 분석기 설정

### 3.1 Kiwi 개요
- **GitHub**: https://github.com/bab2min/kiwipiepy
- **특징**: 빠른 속도, 웹 텍스트 87% / 문어 텍스트 94% 정확도
- **품사 태그**: 세종 말뭉치 기반 + 확장 태그 (W_URL, W_EMAIL, W_HASHTAG 등)

### 3.2 Kiwi 초기화 파라미터

| 파라미터 | 사용 값 | 설명 |
|---------|---------|------|
| `model_type` | `"knlm"` | KNLM 언어 모델 (기본, 권장) |
| `typos` | `None` | 오타 교정 비활성화 (인덱싱 일관성) |
| `load_default_dict` | `True` | 위키백과 표제어 사전 로드 |
| `load_typo_dict` | `False` | 내장 오타 사전 비활성화 |
| `load_multi_dict` | `False` | WikiData 다어절 사전 비활성화 |
| `num_workers` | `0` | 단일 스레드 (안정성) |

### 3.3 토큰화 설정

| 파라미터 | 사용 값 | 설명 |
|---------|---------|------|
| `important_pos` | `{NNG, NNP, NNB, VV, VA, SL, SH, SN}` | 명사, 동사, 형용사, 영어, 한자, 숫자 |
| `normalize_coda` | `False` | 덧붙은 받침 정규화 비활성화 |
| `z_coda` | `True` | 덧붙은 받침 Z_CODA로 분리 |
| `min_token_len` | `1` | 최소 토큰 길이 |

### 3.4 품사 태그 (important_pos)

```python
{
    "NNG",  # 일반 명사 (예: 컴퓨터, 학습)
    "NNP",  # 고유 명사 (예: PyTorch, BERT)
    "NNB",  # 의존 명사 (예: 것, 수)
    "VV",   # 동사 (예: 하다, 만들다)
    "VA",   # 형용사 (예: 좋다, 크다)
    "SL",   # 알파벳 (예: GPU, API)
    "SH",   # 한자
    "SN",   # 숫자 (예: 2024, 3.14)
}
```

## 4. 실행 계획

### 4.1 Step 1: 기존 BM25 삭제

```bash
rm sparse_index/bm25_slack_qa.pkl
```

**이유**: 구버전 파일이 설정과 충돌하지 않도록 제거

### 4.2 Step 2: Qdrant에서 전체 문서 추출

```python
from qdrant_client import QdrantClient

client = QdrantClient(url='http://localhost:6333')
collection_name = 'naver_connect_docs'

all_points = []
offset = None

while True:
    result = client.scroll(
        collection_name=collection_name,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,  # 벡터 로드 비활성화 (메모리 최적화)
    )
    all_points.extend(result[0])
    offset = result[1]
    if offset is None:
        break

print(f"추출된 문서 수: {len(all_points)}")
assert len(all_points) == 13089, f"문서 수 불일치: {len(all_points)}"
```

**검증**: `len(all_points) == 13,089`

### 4.3 Step 3: LangChain Document 변환

```python
from langchain_core.documents import Document

documents = []
skipped = 0

for point in all_points:
    payload = point.payload
    content = payload.get('content', '')

    # 빈 content 필터링
    if not content or not content.strip():
        skipped += 1
        continue

    # metadata 구성 (content 제외)
    metadata = {k: v for k, v in payload.items() if k != 'content'}
    metadata['qdrant_point_id'] = str(point.id)

    documents.append(Document(page_content=content, metadata=metadata))

print(f"변환된 문서 수: {len(documents)}")
print(f"스킵된 문서 수: {skipped}")
```

**검증**:
- `len(documents) + skipped == 13,089`
- 각 doc_type별 문서 수 확인

### 4.4 Step 4: KiwiBM25Retriever 생성

```python
from naver_connect_chatbot.rag.retriever.kiwi_bm25_retriever import (
    KiwiBM25Retriever,
    get_default_important_pos,
)

retriever = KiwiBM25Retriever.from_documents(
    documents=documents,
    k=10,
    model_type="knlm",
    typos=None,
    important_pos=get_default_important_pos(),
    load_default_dict=True,
    load_typo_dict=False,
    load_multi_dict=False,
    num_workers=0,
    auto_save=True,
    save_path="sparse_index/unified_bm25",
    save_user_dict=True,
)
```

**결과 파일**:
```
sparse_index/unified_bm25/
├── bm25_index.pkl    # BM25 인덱스 + 문서 + 설정
└── user_dict.txt     # 사용자 사전 템플릿
```

### 4.5 Step 5: 검증 및 성능 측정

#### 5.1 파일 크기 확인
```python
import os
index_path = "sparse_index/unified_bm25/bm25_index.pkl"
size_mb = os.path.getsize(index_path) / (1024 * 1024)
print(f"인덱스 크기: {size_mb:.2f} MB")
assert size_mb < 100, f"인덱스 크기 초과: {size_mb:.2f} MB"
```

#### 5.2 로드 시간 측정
```python
import time

load_times = []
for i in range(3):
    start = time.time()
    retriever = KiwiBM25Retriever.load("sparse_index/unified_bm25")
    load_times.append(time.time() - start)

avg_load = sum(load_times) / len(load_times)
print(f"평균 로드 시간: {avg_load:.2f}초")
assert avg_load < 5, f"로드 시간 초과: {avg_load:.2f}초"
```

#### 5.3 검색 시간 측정
```python
test_queries = [
    "GPU 메모리 부족 해결 방법",
    "데이터 증강 기법",
    "transformer 어텐션 메커니즘",
]

for query in test_queries:
    start = time.time()
    results = retriever.invoke(query)
    elapsed = (time.time() - start) * 1000
    print(f"Query: '{query}' - {elapsed:.1f}ms, {len(results)} results")
    assert elapsed < 100, f"검색 시간 초과: {elapsed:.1f}ms"
```

#### 5.4 doc_type별 검색 품질 테스트
```python
test_cases = {
    "slack_qa": "GPU 메모리 부족",
    "pdf": "CNN 합성곱 신경망",
    "notebook": "데이터 전처리 코드",
    "weekly_mission": "미션 과제",
}

for expected_type, query in test_cases.items():
    results = retriever.invoke(query)
    if results:
        actual_type = results[0].metadata.get('doc_type')
        print(f"Query: '{query}' → doc_type: {actual_type}")
```

## 5. 성능 기준

| 항목 | 기준 | Fail 시 대안 |
|------|------|-------------|
| 인덱스 크기 | < 100MB | 메타데이터 필드 제외, doc_type별 분리 |
| 로드 시간 | < 5초 | Lazy loading, 분리 인덱스 |
| 검색 시간 | < 100ms/query | BM25 파라미터 튜닝 (k1, b) |

## 6. 리스크 및 대응

### 6.1 잠재적 위험

| 위험 | 영향 | 대응 |
|------|------|------|
| 테스트 경로 하드코딩 | 테스트 실패 | 테스트 파일 경로 업데이트 |
| 메모리 사용량 | OOM 가능 | batch 단위 scroll, `with_vectors=False` |
| 빈 content 문서 | 검색 품질 저하 | content 필터링 추가 |

### 6.2 롤백 계획

실패 시 기존 인덱스로 롤백 불가 (삭제됨). 대신:
1. Qdrant 데이터는 그대로 유지되므로 언제든 재생성 가능
2. 문제 발생 시 이 문서의 Step 2-4 재실행

## 7. 후속 작업

### 7.1 필수
- [ ] 테스트 파일 경로 업데이트 (`tests/test_adaptive_rag_integration.py`)
- [ ] 통합 테스트 실행

### 7.2 권장
- [ ] 사용자 사전에 도메인 용어 추가 (PyTorch, Transformer 등)
- [ ] BM25 파라미터 튜닝 (k1=1.5, b=0.75 기본값 검증)

## 8. 참고 자료

- [Kiwi GitHub](https://github.com/bab2min/Kiwi)
- [Kiwipiepy GitHub](https://github.com/bab2min/kiwipiepy)
- [rank_bm25 PyPI](https://pypi.org/project/rank-bm25/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

## Appendix A: 전체 실행 스크립트

```python
#!/usr/bin/env python3
"""
BM25 통합 인덱스 재생성 스크립트

사용법:
    python rebuild_unified_bm25.py
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from naver_connect_chatbot.rag.retriever.kiwi_bm25_retriever import (
    KiwiBM25Retriever,
    get_default_important_pos,
)


def main():
    print("=" * 80)
    print("BM25 통합 인덱스 재생성")
    print("=" * 80)

    # Step 1: 기존 BM25 삭제
    old_index = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
    if old_index.exists():
        os.remove(old_index)
        print(f"[1] 기존 인덱스 삭제: {old_index}")
    else:
        print(f"[1] 기존 인덱스 없음 (스킵)")

    # Step 2: Qdrant에서 문서 추출
    print("\n[2] Qdrant에서 문서 추출...")
    client = QdrantClient(url='http://localhost:6333')

    all_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name='naver_connect_docs',
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(result[0])
        offset = result[1]
        if offset is None:
            break
        print(f"    진행: {len(all_points)}개...", end="\r")

    print(f"    추출 완료: {len(all_points)}개 문서")

    # Step 3: Document 변환
    print("\n[3] LangChain Document 변환...")
    documents = []
    skipped = 0
    doc_type_counts = {}

    for point in all_points:
        payload = point.payload
        content = payload.get('content', '')

        if not content or not content.strip():
            skipped += 1
            continue

        metadata = {k: v for k, v in payload.items() if k != 'content'}
        metadata['qdrant_point_id'] = str(point.id)

        doc_type = metadata.get('doc_type', 'unknown')
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"    변환 완료: {len(documents)}개 (스킵: {skipped}개)")
    print("    doc_type별:")
    for dtype, count in sorted(doc_type_counts.items()):
        print(f"      - {dtype}: {count}개")

    # Step 4: BM25 인덱스 생성
    print("\n[4] KiwiBM25Retriever 생성...")
    start = time.time()

    retriever = KiwiBM25Retriever.from_documents(
        documents=documents,
        k=10,
        model_type="knlm",
        typos=None,
        important_pos=get_default_important_pos(),
        load_default_dict=True,
        load_typo_dict=False,
        load_multi_dict=False,
        num_workers=0,
        auto_save=True,
        save_path=str(PROJECT_ROOT / "sparse_index" / "unified_bm25"),
        save_user_dict=True,
    )

    build_time = time.time() - start
    print(f"    생성 완료: {build_time:.1f}초")

    # Step 5: 검증
    print("\n[5] 검증...")

    # 파일 크기
    index_path = PROJECT_ROOT / "sparse_index" / "unified_bm25" / "bm25_index.pkl"
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"    인덱스 크기: {size_mb:.2f} MB")

    # 로드 시간
    load_times = []
    for i in range(3):
        start = time.time()
        _ = KiwiBM25Retriever.load(str(PROJECT_ROOT / "sparse_index" / "unified_bm25"))
        load_times.append(time.time() - start)
    avg_load = sum(load_times) / len(load_times)
    print(f"    평균 로드 시간: {avg_load:.2f}초")

    # 검색 테스트
    test_queries = ["GPU 메모리 부족", "데이터 증강", "transformer"]
    for query in test_queries:
        start = time.time()
        results = retriever.invoke(query)
        elapsed = (time.time() - start) * 1000
        print(f"    '{query}': {elapsed:.1f}ms, {len(results)} results")

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

## Appendix B: 세종 품사 태그 체계

| 대분류 | 태그 | 설명 |
|--------|------|------|
| **체언** | NNG | 일반 명사 |
| | NNP | 고유 명사 |
| | NNB | 의존 명사 |
| | NR | 수사 |
| | NP | 대명사 |
| **용언** | VV | 동사 |
| | VA | 형용사 |
| | VX | 보조 용언 |
| **수식언** | MM | 관형사 |
| | MAG | 일반 부사 |
| | MAJ | 접속 부사 |
| **독립언** | IC | 감탄사 |
| **기호** | SF | 종결 부호 |
| | SP | 구분 부호 |
| | SS | 인용 부호 |
| | SE | 줄임표 |
| | SO | 붙임표 |
| | SW | 기타 특수 문자 |
| | SL | 알파벳 |
| | SH | 한자 |
| | SN | 숫자 |
| **Kiwi 확장** | W_URL | URL 주소 |
| | W_EMAIL | 이메일 주소 |
| | W_HASHTAG | 해시태그 |
| | W_MENTION | 멘션 |
| | W_SERIAL | 일련번호 |
| | W_EMOJI | 이모지 |
