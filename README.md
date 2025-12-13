# Naver Bot - TPC

Naver Connect 재단에서 운영하는 **Boost Camp RAG Q&A 챗봇**입니다.
Slack Bot을 통해 부스트캠프 관련 문서에 대한 질문에 답변합니다.

---

## 아키텍처

```mermaid
graph TD
    A[Slack Message] --> B[Slack Handler]
    B --> C[Intent Classifier]
    C --> D{Intent?}
    D -->|QA| E[Query Analyzer]
    D -->|Greeting| F[Simple Response]
    E --> G[Hybrid Retriever]
    G --> H[Dense Search<br/>Qdrant]
    G --> I[Sparse Search<br/>BM25]
    H --> J[Reranker]
    I --> J
    J --> L[Answer Generator - Reasoning]
    L --> M[Response Parser]
    M --> N[Slack Response]
    F --> N
```

---

## 기술 스택

| 카테고리 | 기술 |
|---------|-----|
| **LLM** | Naver Clova Studio (HCX-007) |
| **Reranker** | Clova Reranker |
| **Embedding** | BGE-M3 (via Clova Studio API) |
| **Vector DB** | Qdrant (Hybrid Search: Dense + BM25) |
| **BM25** | kiwipiepy (한국어 형태소 분석) |
| **Web Server** | FastAPI + uvicorn |
| **Messaging** | Slack Bolt (Socket Mode 우선) |
| **Checkpointing** | SQLite |

---

## 프로젝트 구조

```
app/naver_connect_chatbot/
├── server.py           # FastAPI 웹 서버 (Socket/HTTP Mode)
├── config/             # 설정 및 로깅
│   ├── __init__.py     # ProjectSettings 정의
│   ├── embedding.py    # 임베딩 설정
│   ├── llm.py          # Clova LLM 설정
│   ├── log.py          # Loguru 기반 로깅
│   ├── monitoring.py   # Langfuse 모니터링
│   └── settings/       # 개별 설정 모듈들
├── prompts/            # 프롬프트 템플릿
│   ├── loader.py       # 템플릿 로더
│   └── templates/      # 한국어 프롬프트 템플릿
├── rag/                # RAG 핵심 로직

│   ├── rerank.py           # Clova Reranker
│   ├── retriever/          # 하이브리드 검색 (Dense + BM25)
│   ├── retriever_factory.py
│   ├── schema_registry.py  # VDB 스키마 관리
│   ├── segmentation.py     # 문서 분할
│   └── summarization.py    # 요약
├── service/            # LangGraph 기반 서비스
│   ├── agents/         # 에이전트들
│   │   ├── intent_classifier.py    # 의도 분류
│   │   ├── query_analyzer.py       # 쿼리 분석
│   │   ├── answer_generator.py     # 답변 생성
│   │   └── response_parser.py      # 응답 파싱
│   ├── graph/          # LangGraph 워크플로우
│   │   ├── nodes.py        # 그래프 노드
│   │   ├── routing.py      # 라우팅 로직
│   │   ├── state.py        # 상태 정의
│   │   └── workflow.py     # 워크플로우 구성
│   └── tool/           # 도구들(Retriever Tool)
└── slack/              # Slack 통합
    └── handler.py      # Slack 이벤트 핸들러
```

---

## ⚙️ 설치 및 설정

### 요구사항
- **Python 3.13** 이상
- **uv** 패키지 매니저

### 설치

```bash
# 의존성 설치
uv sync

# 개발 도구 포함 설치
uv sync --group dev
```

### 환경 설정

```bash
# .env 파일 생성
make env
# 또는
cp env.example .env

# .env 파일을 열어 API 키 설정
# - CLOVASTUDIO_API_KEY
# - SLACK_BOT_TOKEN
# - SLACK_SIGNING_SECRET
# - SLACK_APP_TOKEN (Socket Mode 사용 시)
```

---

## 🤖 Slack 앱 설정 (Socket Mode)

Slack Bot을 사용하려면 [api.slack.com/apps](https://api.slack.com/apps)에서 앱을 생성하고 설정해야 합니다.

### 1️⃣ 앱 생성

1. [api.slack.com/apps](https://api.slack.com/apps) → **Create New App**
2. **From scratch** 선택
3. App Name 입력 (예: `Boost Camp Bot`)
4. 워크스페이스 선택 → **Create App**

### 2️⃣ Socket Mode 활성화

> **Socket Mode**는 HTTP URL 없이 WebSocket으로 이벤트를 수신합니다.
> 로컬 개발 시 ngrok 같은 터널링 도구가 필요 없어 편리합니다.

1. 좌측 메뉴 → **Socket Mode** 클릭
2. **Enable Socket Mode** 토글 **ON**
3. App-Level Token 생성 팝업:
   - Token Name: `socket-mode-token` (임의 이름)
   - Scope: `connections:write` 선택
   - **Generate** 클릭
4. 생성된 `xapp-...` 토큰을 `.env`의 `SLACK_APP_TOKEN`에 저장

### 3️⃣ Event Subscriptions 설정

> ⚠️ **중요**: Socket Mode를 사용해도 Event Subscriptions 설정은 필수입니다!

1. 좌측 메뉴 → **Event Subscriptions** 클릭
2. **Enable Events** 토글 **ON**
3. **Subscribe to bot events** 섹션 펼치기
4. **Add Bot User Event** 클릭 → 다음 이벤트 추가:

| 이벤트 | 설명 |
|--------|------|
| `app_mention` | 봇이 멘션될 때 (`@봇이름`) |
| `message.im` | DM 메시지 수신 (선택) |

5. 우측 하단 **Save Changes** 클릭

### 4️⃣ OAuth & Permissions 설정

1. 좌측 메뉴 → **OAuth & Permissions** 클릭
2. **Bot Token Scopes** 섹션에서 다음 스코프 추가:

| 스코프 | 설명 |
|--------|------|
| `app_mentions:read` | 멘션 이벤트 수신 |
| `chat:write` | 메시지 전송 |
| `channels:history` | 채널 메시지 읽기 (스레드 컨텍스트용) |
| `im:history` | DM 메시지 읽기 (선택) |

3. 페이지 상단 **Install to Workspace** (또는 **Reinstall**) 클릭
4. **Bot User OAuth Token** (`xoxb-...`)을 `.env`의 `SLACK_BOT_TOKEN`에 저장

### 5️⃣ Signing Secret 확인

1. 좌측 메뉴 → **Basic Information** 클릭
2. **App Credentials** 섹션 → **Signing Secret** 복사
3. `.env`의 `SLACK_SIGNING_SECRET`에 저장

### 6️⃣ 채널에 봇 초대

Slack 채널에서 봇을 사용하려면 초대가 필요합니다:

```
/invite @봇이름
```

### 📝 .env 설정 요약

```bash
# Slack Bot 필수 설정
SLACK_BOT_TOKEN=xoxb-your-bot-token      # Bot User OAuth Token
SLACK_SIGNING_SECRET=your-signing-secret  # App Signing Secret
SLACK_APP_TOKEN=xapp-your-app-token       # App-Level Token (Socket Mode)
```

### ✅ 설정 확인 체크리스트

서버 실행 시 다음 로그가 보이면 정상입니다:

```
Slack 모드: Socket Mode ✅
Socket Mode 연결 시작...
✓ Socket Mode 연결 완료 - Slack 이벤트 수신 준비됨
⚡️ Bolt app is running!
```

### 🔧 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Slack 모드: HTTP Mode` | `SLACK_APP_TOKEN` 미설정 | `.env`에 `xapp-...` 토큰 추가 |
| 멘션해도 응답 없음 | `app_mention` 이벤트 미구독 | Event Subscriptions에서 추가 |
| `invalid_auth` 에러 | 토큰 만료 또는 잘못됨 | OAuth & Permissions에서 재설치 |
| 채널에서 봇 안 보임 | 봇 미초대 | `/invite @봇이름` 실행 |

---

## 실행 방법

### 개발 모드 (로컬)

```bash
uv run uvicorn naver_connect_chatbot.server:api --reload --port 8000
# 또는
uv run python -m naver_connect_chatbot.server
```

### 운영 모드 (서버)
```bash
sh run_server.sh start
```

---

## 테스트

```bash
# 단위 테스트 (빠른 테스트)
uv run pytest -k "not integration"

# 통합 테스트 (실제 API 호출, Docker 서비스 필요)
uv run pytest -m integration -v
```
