# ============================================================================
# Stage 1: Builder
# UV를 사용하여 의존성을 설치하고 애플리케이션을 빌드합니다.
# ============================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# 작업 디렉토리 설정
WORKDIR /build

# UV 캐시 디렉토리 설정 (빌드 속도 최적화)
ENV UV_CACHE_DIR=/tmp/.uv-cache \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 의존성 파일 먼저 복사 (레이어 캐싱 최적화)
COPY pyproject.toml uv.lock ./

# 프로덕션 의존성만 설치 (dev 의존성 제외)
RUN --mount=type=cache,target=/tmp/.uv-cache \
    uv sync --frozen --no-install-project --no-dev

# 소스 코드 복사
COPY app ./app

# 애플리케이션 패키지 설치
RUN --mount=type=cache,target=/tmp/.uv-cache \
    uv sync --frozen --no-dev

# 데이터 파일 복사
# document_chunks: BM25 인덱스 및 처리된 문서
COPY document_chunks ./document_chunks

# sparse_index: BM25 검색용 인덱스 저장소
COPY sparse_index ./sparse_index

# ============================================================================
# Stage 2: Runtime
# 최소한의 이미지 크기로 애플리케이션을 실행합니다.
# ============================================================================
FROM python:3.13-slim

# 메타데이터
LABEL maintainer="Naver Connect Chatbot Team"
LABEL description="Slack chatbot for Naver Connect documentation Q&A"
LABEL version="0.0.1"

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
# - ca-certificates: HTTPS 통신을 위한 인증서
# - tini: 좀비 프로세스 방지를 위한 init 시스템
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tini && \
    rm -rf /var/lib/apt/lists/*

# 비root 사용자 생성 (보안 강화)
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -d /app -s /bin/bash appuser

# Builder 스테이지에서 가상환경 복사
COPY --from=builder --chown=appuser:appuser /build/.venv /app/.venv

# Builder 스테이지에서 데이터 파일 복사
COPY --from=builder --chown=appuser:appuser /build/document_chunks /app/document_chunks
COPY --from=builder --chown=appuser:appuser /build/sparse_index /app/sparse_index

# 로그 디렉토리 생성 (볼륨 마운트용)
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs

# 환경변수 설정
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/.venv/lib/python3.13/site-packages

# 비root 사용자로 전환
USER appuser

# 포트 노출
EXPOSE 8000

# Health check 설정
# - FastAPI의 /health 엔드포인트를 30초마다 확인
# - 3회 연속 실패 시 unhealthy로 판단
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# tini를 init 프로세스로 사용하여 좀비 프로세스 방지
ENTRYPOINT ["/usr/bin/tini", "--"]

# uvicorn으로 FastAPI 애플리케이션 실행
# - host: 0.0.0.0 (모든 인터페이스에서 수신)
# - port: 8000
# - log-level: info
# - no-access-log: 액세스 로그 비활성화 (성능 최적화, loguru로 별도 관리)
CMD ["uvicorn", "naver_connect_chatbot.server:api", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--no-access-log"]
