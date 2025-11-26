# 📋 Document Processing Plan

> **상태**: ✅ 구현 완료  
> **최종 업데이트**: 2025-11-27  
> **목표**: 모든 교육 자료에서 Q&A 챗봇을 위한 의미있는 데이터 추출

---

## 📊 전체 현황

| 데이터 유형 | 폴더 | 파일 형식 | 처리 상태 | 우선순위 |
|------------|------|----------|----------|----------|
| Slack Q&A | `qa_dataset_from_slack/` | JSON | ✅ 완료 | - |
| 실습 코드 | `practice/` | .ipynb | ✅ 완료 | 🥇 1순위 |
| 과제 코드 | `home_work/` | .ipynb, .sh | ✅ 완료 | 🥇 1순위 |
| 강의 자료 | `lecture_content/` | PDF | ✅ 완료 | 🥈 2순위 |
| 주간 미션 | `weekly_mission/` | .ipynb, .xlsx, .docx | ✅ 완료 | 🥉 3순위 |

---

## 🎯 Phase 1: Jupyter Notebooks 처리 (practice/ + home_work/)

### 1.1 목표

Jupyter Notebook (.ipynb) 파일에서 RAG에 유용한 정보 추출:
- 마크다운 셀: 개념 설명, 학습 목표, 가이드
- 코드 셀: 구현 예시, 패턴, 주석
- 출력 셀: 실행 결과, 에러 메시지

### 1.2 폴더 구조

```
document_processing/
├── PROCESSING_PLAN.md          # 이 문서 (계획)
├── notebooks/                   # 📁 Notebook 처리 모듈
│   ├── __init__.py
│   ├── notebook_loader.py       # NotebookLoader 클래스
│   ├── notebook_chunker.py      # 청킹 전략
│   ├── process_all_notebooks.py # 일괄 처리 스크립트
│   └── README.md                # 사용 가이드
├── pdf/                         # 📁 PDF 처리 모듈 (Phase 2)
│   └── ...
└── mission/                     # 📁 주간 미션 처리 모듈 (Phase 3)
    └── ...
```

### 1.3 처리 대상 분석

#### practice/ (실습 자료)

| 과목 | 파일 수 | 주요 내용 |
|------|--------|----------|
| PyTorch | 8개 | Tensor, Linear Regression, Classification |
| AI Math | 6개 | 행렬, einsum, 확률론, 경사하강법 |
| ML LifeCycle | 6개 | NumPy, Back Propagation, Self-Attention |

#### home_work/ (과제)

| 과목 | 파일 수 | 주요 내용 |
|------|--------|----------|
| AI 개발 기초 | 1개 (.sh) | Shell 로그 처리 |
| MRC | 10개 | KorQuAD, TF-IDF, Dense Retrieval, FAISS, ODQA |

### 1.4 추출 전략

#### A. 마크다운 셀 (높은 가치 ⭐⭐⭐⭐⭐)

```python
# 추출 대상
- 학습 목표 / Learning Objectives
- 개념 설명 / Definitions
- 단계별 가이드 / Step-by-step instructions
- 주의사항 / Warnings

# 제외 대상
- 목차만 있는 셀
- 빈 마크다운
```

#### B. 코드 셀 - 정답 버전만 (높은 가치 ⭐⭐⭐⭐)

```python
# 추출 대상 (정답 파일에서)
- 핵심 함수 구현
- 클래스 정의
- 주석이 풍부한 코드
- 모델/데이터 처리 로직

# 제외 대상
- 문제 파일의 빈 코드 (TODO, pass)
- import만 있는 셀
- 단순 출력 확인 코드
```

#### C. 출력 셀 (선택적 가치 ⭐⭐)

```python
# 추출 대상
- 에러 메시지 및 트레이스백
- 모델 summary 출력
- 핵심 결과 (정확도, 손실 등)

# 제외 대상
- 매우 긴 출력 (> 50줄)
- 단순 tensor 출력
- 이미지/그래프 (텍스트로 변환 불가)
```

### 1.5 청킹 전략

```
┌─────────────────────────────────────────────────────┐
│ Chunk 1: 섹션 단위                                    │
├─────────────────────────────────────────────────────┤
│ [Markdown] 섹션 제목 + 설명                           │
│ [Code] 관련 코드 (100-300 토큰)                       │
│ [Output] 핵심 결과 (선택적)                           │
├─────────────────────────────────────────────────────┤
│ Metadata:                                            │
│   - source: "practice/01. AI Core/PyTorch/..."      │
│   - course: "PyTorch"                               │
│   - topic: "Linear Regression"                      │
│   - difficulty: "기본" | "심화"                       │
│   - file_type: "정답" | "문제"                       │
│   - cell_types: ["markdown", "code"]                │
└─────────────────────────────────────────────────────┘
```

### 1.6 구현 체크리스트

- [x] **Step 1**: `notebooks/` 폴더 생성 ✅ (2025-11-26)
- [x] **Step 2**: `notebook_loader.py` 구현 ✅ (2025-11-26)
  - [x] .ipynb JSON 파싱
  - [x] 셀 타입별 분류 (markdown/code/output)
  - [x] 문제/정답 파일 구분 로직
- [x] **Step 3**: `notebook_chunker.py` 구현 ✅ (2025-11-26)
  - [x] 섹션 경계 감지 (## 헤딩 기준)
  - [x] 관련 셀 그룹화
  - [x] 메타데이터 생성
- [x] **Step 4**: `process_all_notebooks.py` 구현 ✅ (2025-11-26)
  - [x] practice/, home_work/ 일괄 처리
  - [x] 통계 출력
- [x] **Step 5**: 출력 검증 및 테스트 ✅ (2025-11-27)
- [x] **Step 6**: 벡터DB 적재 (`ingest_all_to_vectordb.py` 통합) ✅ (2025-11-27)

### 1.7 출력 형식

```json
{
  "chunks": [
    {
      "id": "practice_pytorch_linear_regression_001",
      "content": "## Linear Regression with PyTorch\n\n선형 회귀 모델을...",
      "metadata": {
        "source_file": "practice/01. AI Core/01. PyTorch/(기본-2) Linear Regression/(기본-2) Linear Regression (정답).ipynb",
        "course": "PyTorch",
        "topic": "Linear Regression",
        "difficulty": "기본",
        "file_type": "정답",
        "cell_range": [3, 7],
        "cell_types": ["markdown", "code", "code"]
      }
    }
  ],
  "metadata": {
    "total_chunks": 150,
    "source_folder": "practice",
    "processed_at": "2025-11-26T20:30:00"
  }
}
```

### 1.8 예상 결과

| 지표 | 예상 값 |
|------|---------|
| 처리 파일 수 | ~30개 |
| 생성 청크 수 | ~150-200개 |
| 평균 청크 크기 | 300-500 토큰 |
| 처리 시간 | 1-2분 |

---

## 🎯 Phase 2: PDF 처리 (lecture_content/) - ✅ 완료

### 2.1 목표

PDF 강의 슬라이드에서 개념 설명 및 이론 추출

### 2.2 기술적 고려사항

- PDF 파싱: PyMuPDF4LLM (https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
- 한글 처리: 폰트 임베딩 확인
- 이미지: 별도 폴더에 저장, 청크에서 제외

### 2.3 구현 체크리스트

- [x] `pdf/pdf_loader.py` 구현 ✅ (2025-11-27)
- [x] `pdf/pdf_chunker.py` 구현 ✅ (2025-11-27)
- [x] `pdf/process_all_pdfs.py` 구현 ✅ (2025-11-27)
- [x] 테스트 및 검증 ✅ (2025-11-27)
- [x] `ingest_all_to_vectordb.py` 통합 ✅ (2025-11-27)

### 2.4 결과

- 총 2,818개 청크 생성
- 과목별 JSON 파일 출력

---

## 🎯 Phase 3: 주간 미션 (weekly_mission/) - ✅ 완료

### 3.1 목표

미션 문제 설명 및 힌트 추출 (정답 코드 제외)

### 3.2 주의사항

- ⚠️ 정답 코드 직접 제공 금지
- 학습 목표와 평가 기준만 추출
- 힌트 형태로 제공
- ⚠️ base64 이미지 데이터 필터링

### 3.3 구현 체크리스트

- [x] `mission/mission_loader.py` 구현 ✅ (2025-11-27)
  - [x] .ipynb 문제 파일 로드
  - [x] .xlsx/.docx 체점기준표 로드
  - [x] 정답 파일 자동 제외
- [x] `mission/mission_chunker.py` 구현 ✅ (2025-11-27)
  - [x] 마크다운 셈만 추출
  - [x] 코드 힌트 추출 (TODO, 힌트 주석)
  - [x] base64 이미지 필터링
- [x] `mission/process_all_missions.py` 구현 ✅ (2025-11-27)
- [x] 테스트 및 검증 ✅ (2025-11-27)
- [x] `ingest_all_to_vectordb.py` 통합 ✅ (2025-11-27)

### 3.4 결과

- 총 155개 청크 생성 (30개 문제 파일 + 26개 체점기준표)
- 18개 과목별 JSON 파일 출력

---

## 📈 진행 상황 트래커

### 전체 진행률

```
[██████████] 100% - 모든 Phase 구현 완료!
```

### 완료 항목

- [x] 계획 문서 작성 (2025-11-26)
- [x] **Phase 1**: notebooks/ 모듈 구현 (2025-11-26)
- [x] **Phase 2**: pdf/ 모듈 구현 (2025-11-27)
- [x] **Phase 3**: mission/ 모듈 구현 (2025-11-27)
- [x] `ingest_all_to_vectordb.py` 통합 (2025-11-27)
- [ ] VectorDB 적재 (진행 중 - Rate Limit)

---

## 📝 변경 이력

| 날짜 | 변경 내용 | 담당 |
|------|----------|------|
| 2025-11-26 | 초기 계획 문서 작성 | - |
| 2025-11-26 | Phase 1: notebooks/ 모듈 구현 완료 | - |
| 2025-11-27 | Phase 2: pdf/ 모듈 구현 완료 (2,818 청크) | - |
| 2025-11-27 | Phase 3: mission/ 모듈 구현 완료 (155 청크) | - |
| 2025-11-27 | ingest_all_to_vectordb.py 통합 (PDF, Mission Loader) | - |

---

## 🔗 관련 문서

- [Slack Q&A Loader README](./README.md)
- [Hybrid Search 가이드](./HYBRID_SEARCH_USAGE.md)
- [VectorDB 가이드](./VECTORDB_USAGE.md)
