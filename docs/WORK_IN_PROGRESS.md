# Pre-Retriever 데이터 소스 선택 기능 - 작업 진행 상황

> 마지막 업데이트: 2025-12-09

## 완료된 작업

### 1. VectorDB Payload 인덱스 생성 ✅
- `scripts/create_payload_indexes.py` 생성
- 생성된 인덱스:
  - `course` (Tenant Index, is_tenant=true)
  - `doc_type`
  - `difficulty`
  - `instructor`
  - `file_type`
  - `topic`

### 2. Pre-Retriever 스키마 레지스트리 구현 ✅
- `app/naver_connect_chatbot/rag/schema_registry.py` 생성
- 서버 시작 시 VectorDB에서 데이터 분포 자동 로드
- Query Analyzer 프롬프트에 실제 데이터 소스 정보 주입

### 3. 애매한 질의 처리 기능 구현 ✅ (2025-12-09)

#### Priority 1: `course` 다중값 지원
- `course` 필드를 `str` → `list[str]`로 변경
- OR 조건 필터링 지원 (예: `["CV 이론", "level2_cv"]`)
- 하위 호환성 유지 (str 입력 시 list로 자동 변환)

#### Priority 2: Alias 매핑 (VectorDB 기반 동적 생성)
- `KEYWORD_PATTERNS` 정의: CV, NLP, RecSys, MRC, PyTorch 등 10개 키워드
- `_build_course_aliases()`: VectorDB 과정 목록에서 자동 alias 생성
- `resolve_course_aliases()`: 키워드 → 실제 과정 이름 목록 변환
- `get_alias_context_for_prompt()`: LLM 프롬프트에 alias 정보 주입

#### Priority 3: Fuzzy Matching
- `find_matching_courses()`: difflib.SequenceMatcher 기반 유사도 검색
- `resolve_course_with_fuzzy()`: Alias + Fuzzy 결합 해석
- 부분 문자열 매치 시 유사도 보너스 적용 (0.8)

#### Priority 4: Clarification
- `filter_confidence` 필드 추가 (0.0 ~ 1.0)
- `clarify_node`: 신뢰도 낮을 때 사용자에게 선택지 제시
- `should_clarify()` 라우팅 함수
- `enable_clarification`, `clarification_threshold` 설정 추가

### 수정된 파일
| 파일 | 변경 내용 |
|------|----------|
| `rag/schema_registry.py` | 신규 - 스키마 레지스트리 + Alias/Fuzzy 매핑 |
| `rag/__init__.py` | schema_registry export 추가 |
| `server.py` | lifespan에 스키마 로드 추가 |
| `prompts/templates/query_analysis.yaml` | v5.1 - 다중 course, alias 가이드 |
| `service/agents/query_analyzer.py` | course: list[str], filter_confidence 추가 |
| `service/graph/types.py` | course: list[str] 변경 |
| `service/graph/state.py` | filter_confidence 필드 추가 |
| `service/graph/nodes.py` | alias context 주입, fuzzy 후처리, clarify_node |
| `service/graph/workflow.py` | clarify 노드 및 조건부 라우팅 |
| `service/tool/retrieval_tool.py` | course OR 조건 필터링, TYPE_CHECKING |
| `config/settings/rag_settings.py` | enable_clarification 설정 추가 |

---

## 사용 방법

### Clarification 기능 활성화

```python
# 워크플로우 빌드 시
graph = build_adaptive_rag_graph(
    retriever=retriever,
    llm=llm,
    enable_clarification=True,  # 기본값: False
    clarification_threshold=0.5,  # 기본값: 0.5
)
```

또는 환경 변수:
```bash
ADAPTIVE_RAG_ENABLE_CLARIFICATION=true
ADAPTIVE_RAG_CLARIFICATION_THRESHOLD=0.5
```

### KEYWORD_PATTERNS 확장

```python
# schema_registry.py에서 수정
KEYWORD_PATTERNS = {
    "CV": ["CV", "cv", "Computer Vision", "컴퓨터비전"],
    # 새 키워드 추가...
}
```

---

## 완료됨 (이전 "다음 작업" 섹션)

### 문제 상황

사용자 질의가 애매한 경우 정확한 필터 추출이 어려움:

```
예시 1: "CV 관련 질문"
- doc_type 불명확: pdf? slack_qa? lecture_transcript?

예시 2: "추천시스템 강의"
- course 매칭 어려움:
  - "RecSys" (lecture_transcript)
  - "RecSys 이론" (lecture_transcript)
  - "level2_recsys" (slack_qa)
  - "MLforRecSys" (pdf, lecture_transcript)
  - "RecSys 기초 프로젝트" (pdf)

예시 3: "강의 내용에서 Transformer"
- pdf (슬라이드)? lecture_transcript (녹취록)?
```

### 해결 방안 후보

#### 방안 1: Fuzzy Course Matching
```python
# schema_registry.py에 추가
def find_matching_courses(self, query: str, threshold: float = 0.6) -> list[str]:
    """사용자 입력과 유사한 course 이름들을 반환"""
    from difflib import SequenceMatcher

    matches = []
    for ds in self._schema.data_sources:
        for course in ds.courses:
            ratio = SequenceMatcher(None, query.lower(), course.name.lower()).ratio()
            if ratio >= threshold:
                matches.append({
                    "course": course.name,
                    "doc_type": ds.doc_type,
                    "similarity": ratio,
                    "count": course.count
                })
    return sorted(matches, key=lambda x: -x["similarity"])
```

#### 방안 2: Multi-Source Retrieval
```python
# retrieval_filters에서 여러 doc_type/course 지원
retrieval_filters = {
    "doc_type": ["pdf", "lecture_transcript"],  # 애매하면 여러 소스
    "course": ["RecSys 이론", "level2_recsys", "MLforRecSys"]  # OR 조건
}
```

#### 방안 3: 프롬프트에 애매함 처리 가이드 추가
```yaml
# query_analysis.yaml에 추가
### Handling Ambiguous Queries:
- If doc_type is unclear, include multiple relevant types: ["pdf", "lecture_transcript"]
- If course name is ambiguous, include all matching variants
- Example: "RecSys 강의" → course: ["RecSys 이론", "MLforRecSys", "RecSys 기초 프로젝트"]
```

#### 방안 4: Course Alias 매핑 테이블
```python
COURSE_ALIASES = {
    "CV": ["CV 이론", "level2_cv", "Computer Vision"],
    "NLP": ["NLP", "NLP 이론", "level2_nlp"],
    "RecSys": ["RecSys", "RecSys 이론", "level2_recsys", "MLforRecSys"],
    "추천시스템": ["RecSys", "RecSys 이론", "level2_recsys", "MLforRecSys"],
    ...
}
```

### 권장 구현 순서

1. **즉시 적용 (프롬프트 개선)**
   - query_analysis.yaml에 애매함 처리 가이드 추가
   - 여러 doc_type/course를 배열로 반환하도록 권장

2. **단기 (Alias 매핑)**
   - 자주 사용되는 키워드 → 실제 course 매핑 테이블
   - schema_registry에 alias 조회 기능 추가

3. **중기 (Fuzzy Matching)**
   - 사용자 입력과 유사한 course 자동 탐색
   - 유사도 기반 다중 매칭

4. **장기 (Clarification)**
   - 확신도 낮으면 사용자에게 선택지 제공
   - "어떤 자료에서 찾을까요? [강의자료] [실습노트북] [슬랙Q&A]"

---

## 현재 VectorDB 데이터 분포

```
📊 총 문서: ~15,950개 (전체), 10,000개 (샘플)

doc_type별 분포:
├── pdf (3,287개, 32.9%)
│   └── top courses: Semantic Seg(1127), CV 이론(255), Object Det(192)...
├── notebook (2,987개, 29.9%)
│   └── courses: AI Core(2083), AI Production(875), MRC(29)
├── lecture_transcript (1,804개, 18.0%)
│   └── top courses: NLP(313), MLforRecSys(228), AI Math(192)...
├── slack_qa (1,773개, 17.7%)
│   └── top courses: level2_cv(436), level3_common(330), core_common(273)...
└── weekly_mission (149개, 1.5%)
    └── top courses: MRC(22), RecSys 기초 프로젝트(17), Object Detection(17)...
```

---

## 테스트 방법

### 스키마 레지스트리 테스트
```bash
uv run python3 -c "
from naver_connect_chatbot.rag.schema_registry import SchemaRegistry, get_data_source_context
from qdrant_client import QdrantClient

client = QdrantClient(url='http://localhost:6333')
registry = SchemaRegistry.get_instance()
schema = registry.load_from_qdrant(client, 'naver_connect_docs')

print(get_data_source_context(max_courses=5))
"
```

### 서버 시작 테스트
```bash
python -m naver_connect_chatbot.server
# 로그에서 "VectorDB 스키마 로드 완료" 확인
```

---

## 관련 파일 경로

```
app/naver_connect_chatbot/
├── rag/
│   ├── __init__.py                    # schema_registry export
│   └── schema_registry.py             # 스키마 레지스트리 (신규)
├── prompts/templates/
│   └── query_analysis.yaml            # v5.0 - 동적 데이터 소스
├── service/
│   ├── agents/
│   │   └── query_analyzer.py          # data_source_context 파라미터
│   └── graph/
│       └── nodes.py                   # analyze_query_node 스키마 주입
└── server.py                          # lifespan 스키마 로드

scripts/
└── create_payload_indexes.py          # Qdrant 인덱스 생성
```

---

## 참고 자료

- [Qdrant Filtering Guide](https://qdrant.tech/articles/vector-search-filtering/)
- [Qdrant Payload Indexing](https://qdrant.tech/documentation/concepts/indexing/)
- [Qdrant Tenant Indexing](https://qdrant.tech/documentation/guides/multiple-partitions/)
