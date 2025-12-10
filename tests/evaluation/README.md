# RAG Q&A 평가 데이터셋 및 프레임워크

## 개요

Naver Connect Boost Camp 챗봇의 RAG 시스템을 체계적으로 평가하기 위한 데이터셋과 프레임워크입니다.

## 데이터 분포 (Coverage Matrix)

### doc_type × course 매트릭스

| doc_type | CV | NLP | RecSys | PyTorch | AI Math | MRC | Data Eng | 기타 |
|----------|----|----|--------|---------|---------|-----|----------|------|
| slack_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lecture_transcript | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ |
| pdf | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| notebook | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ |
| weekly_mission | ✓ | ✓ | ✓ | ✓ | - | ✓ | - | ✓ |

## 질문 분류 체계 (Question Taxonomy)

### 1. 주제 내 질문 (In-Domain)

#### 1.1 개념 질문 (Concept)
- 정의, 설명, 원리 질문
- 예: "Transformer의 Self-Attention이란?"

#### 1.2 구현 질문 (Implementation)
- 코드, 설정, 방법 질문
- 예: "PyTorch에서 DataLoader 사용법?"

#### 1.3 트러블슈팅 (Troubleshooting)
- 에러, 문제 해결 질문
- 예: "CUDA out of memory 에러 해결?"

#### 1.4 비교/분석 (Comparison)
- 장단점, 차이점 질문
- 예: "Adam vs SGD 차이점?"

#### 1.5 과정 특정 (Course-Specific)
- 특정 과정/강의 관련
- 예: "CV 3강에서 설명한 ResNet 구조?"

#### 1.6 자료 특정 (Source-Specific)
- 특정 자료 유형 지정
- 예: "Slack에서 학습률 관련 답변?"

### 2. 주제 외 질문 (Out-of-Domain)

#### 2.1 관련 없는 주제 (Unrelated)
- 부스트캠프와 무관한 질문
- 예: "오늘 날씨 어때?"

#### 2.2 불완전/애매한 질문 (Ambiguous)
- 맥락 없는 질문
- 예: "그게 뭐야?", "어떻게 해?"

#### 2.3 할루시네이션 유도 (Hallucination-Inducing)
- 존재하지 않는 정보 질문
- 예: "부스트캠프 10기 커리큘럼?"

#### 2.4 경계 질문 (Boundary)
- AI/ML 관련이지만 자료에 없는
- 예: "GPT-5 아키텍처?"

## 평가 지표 (Evaluation Metrics)

### Retrieval 평가
- **Precision@K**: 상위 K개 문서 중 관련 문서 비율
- **Recall@K**: 전체 관련 문서 중 검색된 비율
- **MRR (Mean Reciprocal Rank)**: 첫 관련 문서 순위
- **Filter Accuracy**: 필터 적용 정확도

### Answer 평가
- **Relevance**: 질문에 대한 답변 관련성 (1-5)
- **Faithfulness**: 검색 문서 기반 답변 여부 (1-5)
- **Completeness**: 답변의 완전성 (1-5)
- **Hallucination Detection**: 환각 포함 여부 (True/False)

### Overall 평가
- **Coverage Score**: 데이터 분포 커버리지
- **Robustness Score**: OOD 질문 처리 능력
- **Latency**: 응답 시간

## 데이터셋 구조

```json
{
  "id": "qa_001",
  "category": "in_domain",
  "subcategory": "concept",
  "question": "Transformer의 Self-Attention이란?",
  "expected_filters": {
    "doc_type": null,
    "course": ["NLP", "NLP 이론"]
  },
  "expected_topics": ["transformer", "attention", "self-attention"],
  "ground_truth": {
    "relevant_doc_ids": ["doc_123", "doc_456"],
    "answer_keywords": ["query", "key", "value", "scaled dot-product"]
  },
  "metadata": {
    "difficulty": "medium",
    "requires_reasoning": false
  }
}
```

## 파일 구조

```
tests/evaluation/
├── README.md                    # 이 파일
├── qa_dataset.json              # Q&A 평가 데이터셋
├── evaluation_config.yaml       # 평가 설정
├── test_rag_evaluation.py       # 평가 테스트 코드
├── metrics/
│   ├── retrieval_metrics.py     # Retrieval 평가 지표
│   └── answer_metrics.py        # Answer 평가 지표
└── reports/
    └── .gitkeep                 # 평가 리포트 저장 폴더
```
