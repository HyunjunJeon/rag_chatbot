#!/bin/sh
# Qdrant Vector Database Docker Management Script
# Usage: ./run_qdrant.sh [start|stop|restart|status|logs]
# POSIX sh compatible

set -e

# =============================================================================
# 설정
# =============================================================================
CONTAINER_NAME="qdrant-vectordb"
QDRANT_IMAGE="qdrant/qdrant:latest"
HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 볼륨 마운트 경로 (프로젝트 내 vdb_store 폴더)
VDB_STORE_PATH="$SCRIPT_DIR/vdb_store"

# 색상 정의 (POSIX 호환 방식)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# 유틸리티 함수
# =============================================================================

log_info() {
    printf "${GREEN}[INFO]${NC} %s\n" "$1"
}

log_warn() {
    printf "${YELLOW}[WARN]${NC} %s\n" "$1"
}

log_error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1"
}

is_running() {
    docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

container_exists() {
    docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

ensure_vdb_store() {
    if [ ! -d "$VDB_STORE_PATH" ]; then
        log_info "Creating volume directory: $VDB_STORE_PATH"
        mkdir -p "$VDB_STORE_PATH"
        chmod 755 "$VDB_STORE_PATH"
        log_info "Volume directory created successfully"
    else
        log_info "Volume directory exists: $VDB_STORE_PATH"
    fi
}

# =============================================================================
# 명령어 함수
# =============================================================================

start_qdrant() {
    if is_running; then
        log_warn "Qdrant is already running"
        show_status
        exit 0
    fi

    # 볼륨 디렉토리 확인/생성
    ensure_vdb_store

    # 기존 중지된 컨테이너가 있으면 삭제
    if container_exists; then
        log_info "Removing stopped container..."
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1
    fi

    echo ""
    printf "${BLUE}🚀 Starting Qdrant Vector Database${NC}\n"
    echo "   ├─ Container: $CONTAINER_NAME"
    echo "   ├─ Image: $QDRANT_IMAGE"
    echo "   ├─ HTTP Port: $HTTP_PORT"
    echo "   ├─ gRPC Port: $GRPC_PORT"
    echo "   └─ Volume: $VDB_STORE_PATH"
    echo ""

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${HTTP_PORT}:6333" \
        -p "${GRPC_PORT}:6334" \
        -v "${VDB_STORE_PATH}:/qdrant/storage:z" \
        "$QDRANT_IMAGE"

    # 시작 대기
    sleep 2

    if is_running; then
        log_info "✅ Qdrant started successfully"
        echo ""
        echo "   Dashboard: http://localhost:${HTTP_PORT}/dashboard"
        echo "   API Docs:  http://localhost:${HTTP_PORT}/openapi"
        echo ""

        # Health check
        printf "   Health check: "
        if curl -sf "http://localhost:${HTTP_PORT}/readyz" >/dev/null 2>&1; then
            printf "${GREEN}✓ Ready${NC}\n"
        else
            printf "${YELLOW}⏳ Starting up...${NC}\n"
        fi
    else
        log_error "Failed to start Qdrant. Check logs: docker logs $CONTAINER_NAME"
        exit 1
    fi
}

stop_qdrant() {
    if ! is_running; then
        log_warn "Qdrant is not running"
        # 중지된 컨테이너가 있으면 삭제
        if container_exists; then
            docker rm "$CONTAINER_NAME" >/dev/null 2>&1
            log_info "Removed stopped container"
        fi
        return 0
    fi

    printf "${YELLOW}🛑 Stopping Qdrant...${NC}\n"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1

    # 컨테이너 삭제 (볼륨은 유지)
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1

    log_info "✅ Qdrant stopped"
    echo "   Volume data preserved at: $VDB_STORE_PATH"
}

restart_qdrant() {
    echo "🔄 Restarting Qdrant..."
    stop_qdrant
    sleep 1
    start_qdrant
}

show_status() {
    echo ""
    printf "${BLUE}📊 Qdrant Status${NC}\n"
    echo "─────────────────────────────────────────"

    if is_running; then
        printf "   Status:    ${GREEN}● Running${NC}\n"

        # 컨테이너 정보
        local container_info=$(docker ps --filter "name=$CONTAINER_NAME" --format "{{.ID}}\t{{.Status}}\t{{.Ports}}")
        local container_id=$(echo "$container_info" | cut -f1)
        local uptime=$(echo "$container_info" | cut -f2)

        echo "   Container: $container_id"
        echo "   Uptime:    $uptime"
        echo "   HTTP Port: $HTTP_PORT"
        echo "   gRPC Port: $GRPC_PORT"
        echo "   Volume:    $VDB_STORE_PATH"

        # 디스크 사용량
        if [ -d "$VDB_STORE_PATH" ]; then
            local size=$(du -sh "$VDB_STORE_PATH" 2>/dev/null | cut -f1)
            echo "   Disk Used: $size"
        fi

        echo ""
        echo "   Dashboard: http://localhost:${HTTP_PORT}/dashboard"

        # Health check
        echo ""
        printf "   Health:    "
        if curl -sf "http://localhost:${HTTP_PORT}/readyz" >/dev/null 2>&1; then
            printf "${GREEN}✓ Healthy${NC}\n"
        else
            printf "${YELLOW}⚠ Not ready${NC}\n"
        fi
    else
        printf "   Status:    ${RED}● Stopped${NC}\n"

        if [ -d "$VDB_STORE_PATH" ]; then
            local size=$(du -sh "$VDB_STORE_PATH" 2>/dev/null | cut -f1)
            echo "   Volume:    $VDB_STORE_PATH ($size)"
        fi
    fi
    echo ""
}

show_logs() {
    if ! container_exists; then
        log_error "No container found. Start Qdrant first."
        exit 1
    fi

    printf "${BLUE}📜 Qdrant Logs${NC} (Ctrl+C to exit)\n"
    echo "─────────────────────────────────────────"
    docker logs -f "$CONTAINER_NAME"
}

pull_image() {
    printf "${BLUE}📥 Pulling latest Qdrant image...${NC}\n"
    docker pull "$QDRANT_IMAGE"
    log_info "Image updated successfully"
}

show_help() {
    echo ""
    printf "${BLUE}Qdrant Docker Management Script${NC}\n"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start Qdrant container (creates vdb_store if needed)"
    echo "  stop      Stop and remove the container (preserves data)"
    echo "  restart   Restart the container"
    echo "  status    Show container status and health"
    echo "  logs      Follow container logs"
    echo "  pull      Pull latest Qdrant image"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  QDRANT_HTTP_PORT   HTTP/REST port (default: 6333)"
    echo "  QDRANT_GRPC_PORT   gRPC port (default: 6334)"
    echo ""
    echo "Data Storage:"
    echo "  All data is persisted in: ./vdb_store/"
    echo ""
}

# =============================================================================
# 메인 명령어 처리
# =============================================================================

# Docker 설치 확인
if ! docker --version > /dev/null 2>&1; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

case "${1:-help}" in
    start)
        start_qdrant
        ;;
    stop)
        stop_qdrant
        ;;
    restart)
        restart_qdrant
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    pull)
        pull_image
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
