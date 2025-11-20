# Slack Q&A Loader

Slack 메시지 JSON 파일에서 학생들의 질문과 답변을 추출하는 도구입니다.

## 개요

이 모듈은 Slack에서 추출한 JSON 파일을 분석하여:
- ✅ 시스템 메시지 제거 (채널 참여, 초대 등)
- ✅ 스레드 구조 분석
- ✅ 질문-답변 쌍 추출
- ✅ 구조화된 JSON 형식으로 저장

## 주요 기능

### 1. `SlackQALoader` 클래스

Slack JSON 파일을 로드하고 Q&A 쌍을 추출하는 메인 클래스입니다.

```python
from slack_qa_loader import SlackQALoader

# 로더 생성
loader = SlackQALoader(exclude_bot_messages=False)

# 단일 파일 처리
qa_pairs = loader.load_from_file("path/to/slack/2021-10-08.json")

# 디렉토리 내 모든 파일 처리
results = loader.load_from_directory("path/to/slack/level2_cv/")

# JSON으로 저장
loader.export_to_json(qa_pairs, "output.json")
```

### 2. 일괄 처리 함수

여러 디렉토리의 파일을 한 번에 처리합니다.

```python
from slack_qa_loader import batch_process_directory

# 전체 디렉토리 일괄 처리
stats = batch_process_directory(
    input_dir="target_documents/qa_dataset_from_slack/2/",
    output_dir="document_chunks/processed/",
    exclude_bot_messages=False,
    recursive=True
)
```

## 사용 예제

### 예제 1: 단일 파일에서 Q&A 추출

```python
from slack_qa_loader import SlackQALoader

loader = SlackQALoader()
qa_pairs = loader.load_from_file("2021-10-08.json")

for qa in qa_pairs:
    print(f"Q: {qa.question.text}")
    print(f"사용자: {qa.question.user_name}")
    
    for answer in qa.answers:
        print(f"  A: {answer.text}")
        print(f"  답변자: {answer.user_name}")
```

### 예제 2: 전체 데이터셋 처리

```python
from document_processing.process_all_slack_data import process_all_slack_data
from pathlib import Path

input_dir = Path("target_documents/qa_dataset_from_slack")
output_dir = Path("document_chunks/slack_qa_processed")

process_all_slack_data(input_dir, output_dir)
```

### 예제 3: 봇 메시지 제외하기

```python
from slack_qa_loader import SlackQALoader

# 봇 메시지를 제외하고 싶은 경우
loader = SlackQALoader(exclude_bot_messages=True)
qa_pairs = loader.load_from_file("2021-10-08.json")
```

## 출력 형식

추출된 Q&A 쌍은 다음과 같은 JSON 형식으로 저장됩니다:

```json
{
  "qa_pairs": [
    {
      "question": {
        "text": "질문 내용...",
        "user": "U02E375H869",
        "user_name": "홍길동",
        "timestamp": "1633683144.000200",
        "is_bot": false,
        "metadata": {
          "edited": null,
          "reactions": [...],
          "reply_count": 2
        }
      },
      "answers": [
        {
          "text": "답변 내용...",
          "user": "U02E375H870",
          "user_name": "김철수",
          "timestamp": "1633683200.000300",
          "is_bot": false,
          "metadata": {...}
        }
      ]
    }
  ]
}
```

## 필터링 규칙

### 제외되는 메시지

1. **시스템 메시지**
   - 채널 참여/퇴장 (`channel_join`, `channel_leave`)
   - 채널 설정 변경 (`channel_archive`, `channel_topic` 등)
   - 초대 메시지 (inviter 필드 존재)

2. **빈 메시지**
   - text 필드가 없거나 비어있는 메시지

3. **봇 메시지** (옵션)
   - `exclude_bot_messages=True`로 설정 시 제외
   - 기본값은 `False` (최신 기수는 답변 봇이 유용한 정보 제공)

### 유지되는 메시지

1. **일반 사용자 메시지**
   - type이 "message"
   - user 필드 존재
   - text 필드에 내용이 있음

2. **스레드 구조**
   - `thread_ts` 필드로 질문-답변 그룹화
   - 최상위 메시지 = 질문
   - 하위 메시지들 = 답변들

## 처리 통계 (2024-11-20 기준)

```
총 처리 파일: 937개
성공: 921개 (98.3%)
실패: 16개 (빈 파일 또는 JSON 오류)

추출된 Q&A 쌍: 1,273개
```

### 기수별 통계

| 기수 | 파일 수 | Q&A 쌍 |
|------|---------|--------|
| 2기 | ~300 | 176 |
| 3기 | ~200 | 209 |
| 4기 | ~150 | 260 |
| 5기 | ~100 | 157 |
| 6기 | ~50 | 65 |
| 7기 | ~100 | 138 |
| 8기 | ~70 | 268 |

## 스크립트 실행

### 테스트 실행

```bash
cd document_processing
python test_slack_qa_loader.py
```

### 전체 데이터 처리

```bash
cd document_processing
python process_all_slack_data.py
```

### 봇 메시지 제외 모드

```bash
cd document_processing
python process_all_slack_data.py --exclude-bot
```

## 데이터 구조

### Message 클래스

```python
@dataclass
class Message:
    text: str                    # 메시지 텍스트
    user: str                    # 사용자 ID
    timestamp: str               # 타임스탬프
    is_bot: bool                 # 봇 메시지 여부
    user_name: str | None        # 사용자 실명
    metadata: dict[str, Any]     # 추가 메타데이터
```

### QAPair 클래스

```python
@dataclass
class QAPair:
    question: Message            # 질문 메시지
    answers: list[Message]       # 답변 메시지 리스트
```

## 주의사항

1. **빈 파일**: 일부 JSON 파일은 빈 배열 `[]`이거나 잘못된 형식일 수 있습니다.
2. **봇 메시지**: 8기부터는 답변 봇이 유용한 답변을 제공하므로 기본값으로 포함됩니다.
3. **스레드 구조**: thread_ts가 없는 메시지는 독립 메시지로 간주되어 질문으로 분류되지만, 답변이 없으면 Q&A 쌍에서 제외됩니다.
4. **한글 인코딩**: UTF-8 인코딩을 사용하므로 한글이 정상적으로 처리됩니다.

## 문제 해결

### JSON 파싱 오류

```
Expecting value: line 1 column 1 (char 0)
```

→ 빈 파일이거나 잘못된 JSON 형식입니다. 해당 파일을 확인하세요.

### Q&A가 추출되지 않음

1. 시스템 메시지만 있는 경우
2. 답변이 없는 독립 메시지만 있는 경우
3. 모든 메시지가 필터링 규칙에 의해 제외된 경우

### 메모리 부족

대량의 파일을 처리할 때 메모리 부족이 발생하면:
- 디렉토리를 분할하여 처리
- `recursive=False`로 설정하여 하위 디렉토리 제외

## 라이센스

이 프로젝트는 내부 사용을 위한 도구입니다.

## 과정별 병합 기능

### `merge_qa_by_course.py`

모든 기수의 Q&A를 과정별로 하나의 JSON 파일로 병합합니다.

```bash
cd document_processing
python merge_qa_by_course.py
```

### 병합 결과

총 15개 과정으로 병합되었습니다:

| 과정 | Q&A 개수 | 기수 | 파일 크기 |
|------|----------|------|-----------|
| level2_cv | 375 | 2-7기 (6개) | 1.2 MB |
| level2_nlp | 197 | 2-7기 (6개) | 607 KB |
| level2_recsys | 113 | 3-7기 (5개) | 355 KB |
| core_common | 194 | 8기 | 679 KB |
| level3_common | 124 | 4-7기 (4개) | 697 KB |
| level3_product_serving | 95 | 2-3기 (2개) | 485 KB |
| bot_common | 74 | 8기 | 427 KB |
| 기타 과정 | 101 | - | - |

### 병합 파일 구조

```json
{
  "course": "level2_cv",
  "metadata": {
    "total_qa_pairs": 375,
    "generations": ["2", "3", "4", "5", "6", "7"],
    "generation_count": 6,
    "statistics": {
      "by_generation": {
        "2": 21,
        "3": 82,
        "4": 155,
        "5": 64,
        "6": 12,
        "7": 41
      }
    },
    "date_range": {
      "start": "2021-09-06",
      "end": "2024-11-28"
    },
    "source_files": {
      "count": 153,
      "files": [...]
    }
  },
  "qa_pairs": [
    {
      "generation": "2",
      "date": "2021-09-06",
      "source_file": "2021-09-06_qa.json",
      "course": "level2_cv",
      "question": {...},
      "answers": [...]
    }
  ]
}
```

### 출력 위치

```
document_chunks/slack_qa_merged/
├── _summary.json                      # 전체 통계
├── level2_cv_merged.json              # 375 Q&A
├── level2_nlp_merged.json             # 197 Q&A
├── level2_recsys_merged.json          # 113 Q&A
├── core_common_merged.json            # 194 Q&A
├── level3_common_merged.json          # 124 Q&A
├── level3_product_serving_merged.json # 95 Q&A
└── ... (15개 파일 총 1,273 Q&A)
```

## 업데이트 이력

### 2024-11-20
- ✅ 초기 버전 릴리스
- ✅ SlackQALoader 클래스 구현
- ✅ 일괄 처리 기능 추가
- ✅ 전체 데이터셋 처리 완료 (1,273 Q&A 쌍 추출)
- ✅ 과정별 병합 기능 추가
- ✅ 메타데이터 포함 병합 파일 생성 (15개 과정)

