# RAG 프롬프트 개선 구현 계획서

> **문서 버전**: 1.0
> **작성일**: 2025-12-10
> **작성자**: Claude Code Analysis
> **상태**: 계획 수립 완료

---

## Executive Summary

본 문서는 Naver Connect Chatbot의 RAG 프롬프트 시스템 분석 결과를 바탕으로 도출된 7가지 개선 사항에 대한 심도 깊은 구현 계획을 제시합니다.

### 개선 우선순위 요약

| Priority | 항목 | 영향도 | 예상 공수 | 위험도 |
|----------|------|--------|-----------|--------|
| P1 | 환각 방지 강화 | 🔴 높음 | 2-3시간 | 낮음 |
| P1 | Multi-Query 출력 형식 유연화 | 🔴 높음 | 1-2시간 | 중간 |
| P2 | Few-Shot 예시 추가 | 🟡 중간 | 2-3시간 | 낮음 |
| P2 | 엣지 케이스 처리 | 🟡 중간 | 2-3시간 | 낮음 |
| P3 | Query Analysis 프롬프트 분리 | 🟢 낮음 | 3-4시간 | 중간 |
| P3 | Self-Validation 추가 | 🟢 낮음 | 1-2시간 | 낮음 |
| P3 | 한국어 동의어 확장 | 🟢 낮음 | 2-3시간 | 낮음 |

---

## Priority 1: Critical

### 1.1 환각 방지 강화 (Hallucination Prevention)

#### 현재 상태 분석

**현재 구현**:
- 모든 프롬프트에 기본 fallback 문구 존재: `"제공된 정보로는 답을 확정할 수 없습니다."`
- 단순한 제약 조건: `"Do not invent information beyond the context"`

**문제점**:
1. 암묵적 추론과 명시적 사실의 구분이 불명확
2. 부분적으로 관련된 컨텍스트 처리 가이드 부재
3. 신뢰도 수준별 표현 방식 가이드 없음

#### 구현 계획

**수정 대상 파일**:
```
app/naver_connect_chatbot/prompts/templates/
├── answer_generation_simple.yaml      (v2.0 → v2.1)
├── answer_generation_complex.yaml     (v2.0 → v2.1)
└── answer_generation_exploratory.yaml (v2.0 → v2.1)
```

**추가할 섹션** (3개 파일 공통):

```yaml
## Hallucination Prevention (Critical)
모든 주장을 생성하기 전에 다음 검증 과정을 거친다:

### 1. 증거 분류 (Evidence Classification)
| 유형 | 조건 | 표현 방식 |
|------|------|----------|
| 직접 인용 | 컨텍스트에 정확히 존재 | 그대로 인용 (따옴표 사용) |
| 안전한 추론 | 컨텍스트에서 논리적으로 도출 가능 | "추론:" 또는 "이를 통해 알 수 있는 것은" 접두어 |
| 불확실한 추론 | 컨텍스트 부분 관련 | "문서에 명시되지 않았지만," + 근거 설명 |
| 외부 정보 | 컨텍스트에 없음 | ❌ 절대 사용 금지 → fallback 문구 |

### 2. 신뢰도 수준별 표현
- **100% 확신**: 직접 인용 또는 컨텍스트 패러프레이즈
- **80-99% 확신**: "일반적으로", "대체로" + 근거 명시
- **60-79% 확신**: "추론컨대", "문맥상" + 명확한 출처 표시
- **<60% 확신**: fallback 문구 사용

### 3. 모순 정보 처리
컨텍스트 내 정보가 상충할 경우:
- "문서 A에서는 [X]로 설명하고, 문서 B에서는 [Y]로 설명합니다."
- "이러한 차이는 [시점/버전/맥락]에 따른 것으로 보입니다."
- 가장 신뢰할 수 있는 출처를 우선시하되, 차이점을 명시

### 4. 자가 검증 체크리스트
답변 완성 전 각 주장에 대해 확인:
□ 이 정보가 컨텍스트에 있는가?
□ 직접 인용인가, 추론인가?
□ 추론이라면 논리적 근거가 있는가?
□ 불확실성을 적절히 표현했는가?
```

**각 프롬프트별 추가 위치**:

1. **answer_generation_simple.yaml** (Line 27 이후):
```yaml
## Hallucination Prevention (Critical)
[위 공통 섹션]

- Simple QA에서는 직접 인용을 우선시
- 추론이 필요한 경우 반드시 "추론:" 접두어 사용
- 5문장 제한 내에서 불확실성 표현 간결하게
```

2. **answer_generation_complex.yaml** (Line 30 이후):
```yaml
## Hallucination Prevention (Critical)
[위 공통 섹션]

- 복잡한 추론에서는 각 단계별로 근거 명시
- 논리적 연결이 컨텍스트에서 지원되는지 단계마다 검증
- 가정(assumption)은 "가정:" 접두어로 명시
```

3. **answer_generation_exploratory.yaml** (Line 44 이후):
```yaml
## Hallucination Prevention (Critical)
[위 공통 섹션]

- 탐색적 답변에서도 권장사항은 컨텍스트 기반
- 일반적인 모범 사례 언급 시 "일반적으로" 명시
- 학습 경로 제안은 컨텍스트에 언급된 내용 우선
```

#### 테스트 계획

```python
# tests/prompts/test_hallucination_prevention.py
class TestHallucinationPrevention:
    """환각 방지 프롬프트 테스트"""

    @pytest.mark.parametrize("strategy", ["simple", "complex", "exploratory"])
    async def test_direct_citation_used(self, strategy):
        """컨텍스트에 있는 정보는 직접 인용되는지 확인"""

    async def test_inference_marked(self):
        """추론된 정보에 적절한 마커가 있는지 확인"""

    async def test_contradiction_handling(self):
        """모순된 컨텍스트 처리가 적절한지 확인"""

    async def test_fallback_on_insufficient_context(self):
        """컨텍스트 부족 시 fallback 문구 사용 확인"""
```

---

### 1.2 Multi-Query 출력 형식 유연화

#### 현재 상태 분석

**현재 구현** (`multi_query_generation.yaml`):
```yaml
## Output Format
- Return exactly {num} queries (한국어로), one query per line
- No numbering, bullet points, headers, or commentary
- 줄 수가 {num}와 다르면 잘못된 출력으로 간주됨
```

**문제점**:
1. 너무 엄격한 형식 → LLM이 코멘트 추가 시 파싱 실패
2. 번호 매기기 금지 → 오히려 파싱 어려움
3. fallback 전략 부재

#### 구현 계획

**수정 대상 파일**:
```
app/naver_connect_chatbot/prompts/templates/multi_query_generation.yaml (v1.0 → v1.1)
app/naver_connect_chatbot/rag/retriever/multi_query_retriever.py
```

**프롬프트 수정**:

```yaml
# multi_query_generation.yaml v1.1
## Output Format
검색 쿼리를 {num}개 생성하여 반환한다.

### 허용되는 형식 (아래 중 하나 선택):
1. **줄바꿈 구분** (권장):
   ```
   첫 번째 검색 쿼리
   두 번째 검색 쿼리
   세 번째 검색 쿼리
   ```

2. **번호 형식**:
   ```
   1. 첫 번째 검색 쿼리
   2. 두 번째 검색 쿼리
   3. 세 번째 검색 쿼리
   ```

3. **불릿 형식**:
   ```
   - 첫 번째 검색 쿼리
   - 두 번째 검색 쿼리
   - 세 번째 검색 쿼리
   ```

### 규칙:
- 쿼리 개수는 정확히 {num}개
- 각 쿼리는 구체적이고 검색에 최적화
- 추가 설명이나 코멘트 없이 쿼리만 반환
- 최소 3개, 최대 5개 쿼리 생성

### 오류 방지:
- {num}개 미만 생성 시: 가능한 만큼 생성 (최소 2개)
- 중복 쿼리 생성 금지
- 원본 쿼리와 동일한 쿼리 포함 가능
```

**파서 로직 수정** (`multi_query_retriever.py`):

```python
# 현재 코드
def _parse_queries(self, text: str) -> list[str]:
    """Parse LLM output into list of queries."""
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]

# 개선된 코드
import re

def _parse_queries(self, text: str) -> list[str]:
    """
    Parse LLM output into list of queries.

    Handles multiple formats:
    - Plain newline-separated
    - Numbered (1. query, 2. query)
    - Bulleted (- query, * query)
    - Mixed formats
    """
    lines = text.strip().split('\n')
    queries = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes
        # Pattern: "1.", "1)", "1:", "-", "*", "•"
        cleaned = re.sub(r'^[\d]+[.\):\s]+|^[-*•]\s*', '', line).strip()

        # Skip empty or too short queries
        if len(cleaned) < 5:
            continue

        # Skip lines that look like headers or comments
        if cleaned.startswith('#') or cleaned.endswith(':'):
            continue

        queries.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        normalized = q.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_queries.append(q)

    return unique_queries[:self.num_queries]  # Limit to requested count
```

#### 테스트 케이스

```python
# tests/rag/test_multi_query_parser.py
class TestMultiQueryParser:
    """Multi-Query 파서 테스트"""

    @pytest.mark.parametrize("input_text,expected_count", [
        # 줄바꿈 형식
        ("쿼리1\n쿼리2\n쿼리3", 3),
        # 번호 형식
        ("1. 쿼리1\n2. 쿼리2\n3. 쿼리3", 3),
        # 불릿 형식
        ("- 쿼리1\n- 쿼리2\n- 쿼리3", 3),
        # 혼합 형식
        ("1. 쿼리1\n- 쿼리2\n쿼리3", 3),
        # 빈 줄 포함
        ("쿼리1\n\n쿼리2\n\n쿼리3", 3),
        # 코멘트 포함
        ("Here are queries:\n1. 쿼리1\n2. 쿼리2", 2),
    ])
    def test_parse_various_formats(self, input_text, expected_count):
        """다양한 형식 파싱 테스트"""

    def test_deduplication(self):
        """중복 쿼리 제거 테스트"""

    def test_empty_line_handling(self):
        """빈 줄 처리 테스트"""
```

---

## Priority 2: Important

### 2.1 Few-Shot 예시 추가

#### 현재 상태 분석

| 프롬프트 | 현재 예시 | 필요 여부 |
|----------|-----------|-----------|
| query_analysis.yaml | 3개 ✅ | 충분 |
| multi_query_generation.yaml | 1개 ✅ | 충분 |
| answer_generation_simple.yaml | 0개 ❌ | 필요 |
| answer_generation_complex.yaml | 0개 ❌ | 필요 |
| answer_generation_exploratory.yaml | 0개 ❌ | 필요 |

#### 구현 계획

**추가할 Few-Shot 예시**:

1. **answer_generation_simple.yaml**:

```yaml
## Example

### Input
Question: "PyTorch에서 GPU 사용하려면 어떻게 해야 하나요?"
Context: "PyTorch에서 GPU를 사용하려면 `.to('cuda')` 또는 `.cuda()` 메서드를 사용합니다.
먼저 `torch.cuda.is_available()`로 GPU 사용 가능 여부를 확인하는 것이 좋습니다.
모델과 데이터 모두 같은 디바이스에 있어야 연산이 가능합니다."

### Output
PyTorch에서 GPU를 사용하려면 모델과 데이터에 `.to('cuda')` 또는 `.cuda()` 메서드를 적용합니다 (문맥: "`.to('cuda')` 또는 `.cuda()` 메서드를 사용합니다"). 사용 전 `torch.cuda.is_available()`로 GPU 가용성을 확인하는 것이 권장됩니다. 예를 들어, `model = model.to('cuda')`와 같이 사용합니다. 주의할 점은 모델과 입력 데이터가 반드시 동일한 디바이스에 있어야 한다는 것입니다.
```

2. **answer_generation_complex.yaml**:

```yaml
## Example

### Input
Question: "Transformer에서 Self-Attention과 Cross-Attention의 차이점과 각각 어디에 사용되는지 설명해주세요"
Context: "Self-Attention은 같은 시퀀스 내에서 토큰 간의 관계를 계산합니다. Query, Key, Value가 모두 같은 입력에서 생성됩니다.
Cross-Attention은 두 개의 다른 시퀀스 간의 관계를 계산합니다. Query는 디코더에서, Key와 Value는 인코더에서 옵니다.
Encoder에서는 Self-Attention만 사용하고, Decoder에서는 Self-Attention과 Cross-Attention 모두 사용합니다."

### Output
## Concept Primer
Transformer의 Attention 메커니즘은 크게 Self-Attention과 Cross-Attention으로 구분됩니다. 두 방식 모두 Query, Key, Value를 사용하지만 입력 소스가 다릅니다.

## Analytical Breakdown
**1. Self-Attention의 작동 방식**
Self-Attention은 "같은 시퀀스 내에서 토큰 간의 관계를 계산"합니다 (문맥 인용). 즉, Q, K, V가 모두 동일한 입력 시퀀스에서 생성됩니다. 이를 통해 문장 내 단어들 간의 의존성을 학습합니다.

**2. Cross-Attention의 작동 방식**
Cross-Attention은 "두 개의 다른 시퀀스 간의 관계를 계산"합니다. 구체적으로 "Query는 디코더에서, Key와 Value는 인코더에서" 옵니다. 이를 통해 출력 생성 시 입력 정보를 참조합니다.

**3. 사용 위치**
- Encoder: Self-Attention만 사용 (입력 시퀀스 내부 관계 파악)
- Decoder: Self-Attention + Cross-Attention (출력 내부 관계 + 입력 참조)

## Conclusion
Self-Attention은 단일 시퀀스 내부의 관계를, Cross-Attention은 두 시퀀스 간의 관계를 모델링합니다.

## Practical Implication
번역 태스크에서 Decoder의 Cross-Attention은 현재 생성 중인 단어가 원문의 어떤 부분과 관련되는지 학습하여 정확한 번역을 가능하게 합니다.
```

3. **answer_generation_exploratory.yaml**:

```yaml
## Example

### Input
Question: "딥러닝 프로젝트를 처음 시작할 때 어떤 순서로 접근하면 좋을까요?"
Context: "딥러닝 프로젝트는 문제 정의, 데이터 수집, EDA, 전처리, 모델 선택, 학습, 평가, 배포 순으로 진행됩니다.
데이터 품질이 모델 성능의 80%를 결정한다고 알려져 있습니다.
초보자는 먼저 공개 데이터셋으로 연습하고, 점진적으로 복잡한 프로젝트로 확장하는 것이 좋습니다.
PyTorch Lightning이나 Keras 같은 고수준 API를 활용하면 보일러플레이트 코드를 줄일 수 있습니다."

### Output
## Overview
딥러닝 프로젝트는 문제 정의부터 배포까지 체계적인 파이프라인을 따릅니다 (근거: "문제 정의, 데이터 수집, EDA, 전처리, 모델 선택, 학습, 평가, 배포 순으로 진행").

## Key Components
- **문제 정의**: 해결하려는 문제와 성공 지표 명확화
- **데이터**: 수집, EDA, 전처리 (근거: "데이터 품질이 모델 성능의 80%를 결정")
- **모델링**: 적절한 아키텍처 선택 및 학습
- **평가/배포**: 성능 검증 및 실제 환경 적용

## Step-by-step Guide
1. 명확한 문제 정의 및 평가 지표 설정
2. 데이터 수집 및 탐색적 분석 (EDA)
3. 데이터 전처리 및 증강
4. 베이스라인 모델 구축
5. 실험 및 하이퍼파라미터 튜닝
6. 최종 평가 및 배포

## Practical Tips
- 처음에는 공개 데이터셋으로 연습 권장 (근거: "초보자는 먼저 공개 데이터셋으로 연습")
- PyTorch Lightning, Keras 등 고수준 API 활용 고려 (근거: "보일러플레이트 코드를 줄일 수 있습니다")
- 작은 모델과 데이터로 파이프라인 먼저 검증

## Next Steps
- Kaggle의 Getting Started 대회로 실전 경험 쌓기
- 논문 구현 연습으로 깊이 있는 이해 추구
- MLOps 기초 학습으로 배포 역량 확보
```

#### 구현 위치

각 프롬프트 파일의 `## Constraints` 섹션 바로 앞에 `## Example` 섹션 추가

---

### 2.2 엣지 케이스 처리

#### 현재 상태 분석

**처리되지 않는 엣지 케이스**:
1. 빈 컨텍스트 (0개 문서)
2. 매우 긴 컨텍스트 (>4000 토큰)
3. 모순된 정보가 있는 컨텍스트
4. 컨텍스트가 질문과 완전히 무관한 경우

#### 구현 계획

**추가할 섹션** (모든 answer_generation 프롬프트):

```yaml
## Edge Case Handling

### 1. 빈 컨텍스트 (Empty Context)
컨텍스트가 비어있거나 "[문서 없음]"인 경우:
→ 즉시 fallback: "제공된 정보로는 답을 확정할 수 없습니다. 질문을 더 구체적으로 해주시거나 관련 키워드를 포함해 주세요."

### 2. 매우 긴 컨텍스트 (>2000 tokens)
- 질문과 가장 관련 있는 상위 3-5개 문단에 집중
- 인용 시 출처 문서 번호 명시: "(문서 2에서:...)"
- 전체 요약보다 핵심 정보 추출 우선

### 3. 모순된 정보 (Contradictory Information)
컨텍스트 내 정보가 상충하는 경우:
```
"이 주제에 대해 제공된 문서들에서 다른 관점이 있습니다:
- [출처 A]에서는 [X]로 설명합니다.
- [출처 B]에서는 [Y]로 설명합니다.
[가능한 경우] 이러한 차이는 [맥락/버전/관점]에 따른 것으로 보입니다."
```

### 4. 무관한 컨텍스트 (Irrelevant Context)
검색된 문서가 질문과 관련 없는 경우:
→ fallback: "제공된 문서에서 질문과 직접 관련된 정보를 찾지 못했습니다. 다른 키워드로 검색해 보시거나 질문을 구체화해 주세요."

### 5. 부분 관련 컨텍스트 (Partial Relevance)
일부만 관련된 경우:
- 관련 부분만 활용하여 답변
- "제공된 문서에서 [관련 부분]에 대한 정보를 찾았습니다. [다른 부분]에 대해서는 추가 정보가 필요합니다."
```

---

## Priority 3: Enhancement

### 3.1 Query Analysis 프롬프트 분리

#### 현재 상태

`query_analysis.yaml`: **111줄** - 3가지 기능 통합
1. Quality Analysis (품질 평가)
2. Multi-Query Generation (다중 쿼리 생성)
3. Filter Extraction (필터 추출)

#### 문제점

- 긴 프롬프트로 인한 컨텍스트 부담
- 단일 기능만 필요할 때 비효율적
- 유지보수 어려움

#### 구현 계획

**분리 구조**:

```
prompts/templates/
├── query_analysis.yaml           (기존 - deprecated, 호환성 유지)
├── query_quality_analysis.yaml   (신규 - 품질 평가만)
└── query_expansion.yaml          (신규 - 다중 쿼리 + 필터)
```

**1. query_quality_analysis.yaml** (~40줄):

```yaml
_type: chat_messages
metadata:
  name: query_quality_analysis
  description: Analyze query quality only (clarity, specificity, searchability)
  version: "1.0"
  author: Adaptive RAG System
  last_updated: "2025-12-10"

messages:
  - role: system
    content: |
      ## Task
      Evaluate a Naver Boost Camp student's question quality across three dimensions.

      ## Evaluation Dimensions
      1. **Clarity** (0.0-1.0): 모호한 표현이나 누락된 주어가 있는지
      2. **Specificity** (0.0-1.0): 범위, 엔티티, 제약 조건이 명확한지
      3. **Searchability** (0.0-1.0): 검색 가능한 용어로 매핑되는지

      ## Output Format
      JSON 형식으로 반환:
      {
        "clarity_score": 0.8,
        "specificity_score": 0.7,
        "searchability_score": 0.9,
        "issues": ["이슈1", "이슈2"],
        "recommendations": ["권장1", "권장2"]
      }

  - role: human
    content: |
      Question: {question}
      Intent: {intent}

input_variables:
  - question
  - intent
```

**2. query_expansion.yaml** (~70줄):

```yaml
_type: chat_messages
metadata:
  name: query_expansion
  description: Generate multiple queries and extract retrieval filters
  version: "1.0"
  author: Adaptive RAG System
  last_updated: "2025-12-10"

messages:
  - role: system
    content: |
      ## Task
      Generate diverse search queries and extract metadata filters for targeted retrieval.

      {data_source_context}

      ## Multi-Query Generation
      [기존 query_analysis.yaml의 Step 2 내용]

      ## Filter Extraction
      [기존 query_analysis.yaml의 Step 3 내용]

      ## Output Format
      {
        "improved_queries": ["쿼리1", "쿼리2", "쿼리3"],
        "retrieval_filters": {
          "doc_type": ["pdf"],
          "course": ["CV 이론"],
          "course_topic": null,
          "generation": null
        }
      }

  - role: human
    content: |
      Question: {question}
      Intent: {intent}
      Quality Scores: clarity={clarity}, specificity={specificity}

input_variables:
  - question
  - intent
  - clarity
  - specificity
  - data_source_context
```

**코드 수정** (`query_analyzer.py`):

```python
# 옵션 1: 기존 호환성 유지하면서 새 프롬프트 지원
async def aanalyze_query(
    self,
    question: str,
    intent: str,
    use_split_prompts: bool = False,  # 새 파라미터
) -> QueryAnalysis:
    if use_split_prompts:
        # 분리된 프롬프트 사용
        quality = await self._analyze_quality(question, intent)
        expansion = await self._expand_query(question, intent, quality)
        return self._merge_results(quality, expansion)
    else:
        # 기존 통합 프롬프트 사용
        return await self._analyze_with_combined_prompt(question, intent)
```

---

### 3.2 Self-Validation 추가

#### 구현 계획

각 프롬프트에 `## Self-Validation` 섹션 추가:

**answer_generation_*.yaml**:
```yaml
## Self-Validation (Before Returning)
답변 완성 후, 반환 전에 다음을 확인한다:

1. **사실 검증**: 모든 주장이 컨텍스트에서 지원되는가?
2. **구조 검증**: 요구된 형식(문장 수, 섹션 구조)을 따랐는가?
3. **언어 검증**: 한국어로 자연스럽게 작성되었는가?
4. **인용 검증**: 최소 1개 이상의 컨텍스트 인용이 있는가?
5. **완전성 검증**: 질문의 모든 부분에 답변했는가?

검증 실패 시: 해당 부분 수정 또는 불확실성 표시
```

**multi_query_generation.yaml**:
```yaml
## Self-Validation (Before Returning)
쿼리 생성 후, 반환 전에 다음을 확인한다:

1. **개수 검증**: 정확히 {num}개의 쿼리가 있는가?
2. **다양성 검증**: 각 쿼리가 서로 다른 관점을 다루는가?
3. **검색성 검증**: 각 쿼리가 검색에 최적화되어 있는가?
4. **중복 검증**: 동일하거나 매우 유사한 쿼리가 없는가?
5. **관련성 검증**: 모든 쿼리가 원본 질문과 관련 있는가?

검증 실패 시: 문제 있는 쿼리 수정 또는 대체
```

---

### 3.3 한국어 동의어 확장

#### 구현 계획

**query_analysis.yaml에 추가할 섹션**:

```yaml
## Korean Synonym Expansion
검색 쿼리 생성 시 한국어 동의어와 관련 표현을 활용한다:

### 일반 학습 용어
| 원어 | 동의어/관련어 |
|------|--------------|
| 학습 | 훈련, 트레이닝, learning |
| 모델 | 모형, 네트워크, 아키텍처 |
| 예측 | 추론, inference, prediction |
| 정확도 | 성능, accuracy, 정밀도 |
| 손실 | 로스, loss, 오차 |
| 가중치 | weight, 파라미터, parameter |

### 분야별 용어
| 분야 | 용어 변형 |
|------|----------|
| CV | 컴퓨터 비전, Computer Vision, 영상 처리 |
| NLP | 자연어 처리, Natural Language Processing, 텍스트 |
| RecSys | 추천 시스템, Recommendation, 추천 알고리즘 |

### 확장 규칙
1. 원본 쿼리의 핵심 용어에 대해 동의어 버전 쿼리 1개 포함
2. 영어-한국어 혼용 쿼리로 검색 범위 확대
3. 약어 사용 시 풀네임 버전도 고려 (CNN → Convolutional Neural Network)
```

---

## 구현 일정

| 주차 | 작업 | 담당 | 상태 |
|------|------|------|------|
| Week 1 | P1: 환각 방지 강화 | - | ⬜ 대기 |
| Week 1 | P1: Multi-Query 형식 유연화 | - | ⬜ 대기 |
| Week 2 | P2: Few-Shot 예시 추가 | - | ⬜ 대기 |
| Week 2 | P2: 엣지 케이스 처리 | - | ⬜ 대기 |
| Week 3 | P3: Query Analysis 분리 | - | ⬜ 대기 |
| Week 3 | P3: Self-Validation | - | ⬜ 대기 |
| Week 3 | P3: 한국어 동의어 | - | ⬜ 대기 |
| Week 4 | 통합 테스트 & 평가 | - | ⬜ 대기 |

---

## 테스트 계획

### 단위 테스트

```bash
# 프롬프트 파싱 테스트
pytest tests/prompts/ -v

# Multi-Query 파서 테스트
pytest tests/rag/test_multi_query_parser.py -v
```

### 통합 테스트

```bash
# 전체 RAG 파이프라인 테스트
pytest tests/test_adaptive_rag_integration.py -v -m integration
```

### 평가 테스트

```bash
# LLM-as-Judge 평가 (개선 전후 비교)
pytest tests/evaluation/test_rag_evaluation_v2.py -v -m integration
```

### 평가 지표

| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|----------|
| 환각률 | TBD | <5% | LLM-as-Judge hallucination_detected |
| 통과율 | ~75% | >80% | test_full_evaluation_v2 pass_rate |
| 평균 점수 | ~0.85 | >0.88 | overall_score |

---

## 롤백 계획

### 버전 관리

모든 프롬프트 파일에 버전 명시:
```yaml
metadata:
  version: "2.1"  # 변경 시 버전 증가
  previous_version: "2.0"  # 롤백 대상
```

### 롤백 절차

1. **즉시 롤백** (5분 이내):
   ```bash
   git checkout HEAD~1 -- app/naver_connect_chatbot/prompts/templates/
   ```

2. **선택적 롤백** (특정 프롬프트만):
   ```bash
   git checkout HEAD~1 -- app/naver_connect_chatbot/prompts/templates/answer_generation_simple.yaml
   ```

3. **A/B 테스트 롤백**:
   - Feature flag로 새 프롬프트 비활성화
   - 점진적 롤아웃 중단

---

## 참고 자료

- [LangChain Prompt Engineering Guide](https://python.langchain.com/docs/concepts/prompt_templates/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-10 | 1.0 | 초기 문서 작성 |
