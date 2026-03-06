#!/bin/bash
# FastAPI Server Management Script
# Usage: ./run_server.sh [start|stop|restart|status|dev]

set -e

# 설정
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 파일 경로
PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/logs/server.log"

# 로그 디렉토리 생성
mkdir -p "$(dirname "$LOG_FILE")"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

get_pid() {
    # 1) PID 파일에서 읽기
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return
        fi
    fi
    # 2) PID 파일이 없거나 유효하지 않으면 프로세스명으로 검색
    pgrep -f "uvicorn naver_connect_chatbot.server:api" 2>/dev/null | head -1
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# 포트를 점유 중인 모든 프로세스를 종료
kill_port_holders() {
    if command -v fuser &> /dev/null; then
        fuser -k "$PORT/tcp" 2>/dev/null || true
    else
        # fuser 없으면 ss + kill로 대체
        local pids
        pids=$(ss -tlnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
        for p in $pids; do
            kill -TERM "$p" 2>/dev/null || true
        done
    fi
}

start_server() {
    if is_running; then
        echo -e "${YELLOW}Server is already running (PID: $(get_pid))${NC}"
        exit 1
    fi

    # 포트가 점유되어 있으면 정리
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo -e "${YELLOW}Port $PORT is in use. Cleaning up...${NC}"
        kill_port_holders
        sleep 2
    fi

    echo -e "${GREEN}Starting Naver Connect Chatbot Server${NC}"
    echo "   Host: $HOST"
    echo "   Port: $PORT"
    echo "   Workers: $WORKERS"
    echo "   Log: $LOG_FILE"
    echo ""

    # 백그라운드로 서버 시작
    nohup uv run uvicorn naver_connect_chatbot.server:api \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        --no-access-log \
        >> "$LOG_FILE" 2>&1 &

    # PID 저장
    echo $! > "$PID_FILE"

    # 서버 시작 대기
    sleep 2

    # PID 파일이 유효하지 않을 수 있으므로 프로세스명으로 재확인
    if ! is_running; then
        # nohup 래퍼 PID가 종료되었을 수 있으므로 실제 uvicorn PID로 갱신
        local real_pid
        real_pid=$(pgrep -f "uvicorn naver_connect_chatbot.server:api" 2>/dev/null | head -1)
        if [ -n "$real_pid" ]; then
            echo "$real_pid" > "$PID_FILE"
        fi
    fi

    if is_running; then
        echo -e "${GREEN}Server started successfully (PID: $(get_pid))${NC}"
        echo "   View logs: tail -f $LOG_FILE"
    else
        echo -e "${RED}Failed to start server. Check logs: $LOG_FILE${NC}"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop_server() {
    if ! is_running; then
        # PID 파일로 찾지 못해도 포트로 한번 더 확인
        if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
            echo -e "${YELLOW}No tracked PID, but port $PORT is in use. Cleaning up...${NC}"
            kill_port_holders
            sleep 2
            rm -f "$PID_FILE"
            echo -e "${GREEN}Server stopped${NC}"
            return 0
        fi
        echo -e "${YELLOW}Server is not running${NC}"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    echo -e "${YELLOW}🛑 Stopping server (PID: $pid)...${NC}"

    # SIGTERM으로 graceful shutdown
    kill -TERM "$pid" 2>/dev/null

    # 최대 10초 대기
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""

    # 여전히 실행 중이면 SIGKILL
    if is_running; then
        echo -e "${RED}Force killing...${NC}"
        kill -KILL "$pid" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ Server stopped${NC}"
}

restart_server() {
    echo "🔄 Restarting server..."
    stop_server
    sleep 1
    start_server
}

show_status() {
    if is_running; then
        local pid=$(get_pid)
        echo -e "${GREEN}✅ Server is running${NC}"
        echo "   PID: $pid"
        echo "   Port: $PORT"
        echo "   Log: $LOG_FILE"
        echo ""
        echo "📊 Process info:"
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,command 2>/dev/null || true
    else
        echo -e "${RED}❌ Server is not running${NC}"
        rm -f "$PID_FILE"
    fi
}

dev_mode() {
    if is_running; then
        echo -e "${YELLOW}⚠️  Production server is running. Stop it first: ./run_server.sh stop${NC}"
        exit 1
    fi

    echo -e "${GREEN}📝 Starting development server (foreground, hot reload)${NC}"
    echo "   Host: $HOST"
    echo "   Port: $PORT"
    echo ""

    uv run uvicorn naver_connect_chatbot.server:api \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --reload-dir app \
        --log-level debug
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo -e "${YELLOW}⚠️  Log file not found: $LOG_FILE${NC}"
    fi
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start server in background (production mode)"
    echo "  stop      Stop the running server"
    echo "  restart   Restart the server"
    echo "  status    Show server status"
    echo "  dev       Start in development mode (foreground, hot reload)"
    echo "  logs      Tail the log file"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  HOST      Server host (default: 0.0.0.0)"
    echo "  PORT      Server port (default: 8000)"
    echo "  WORKERS   Number of workers (default: 4)"
}

# 메인 명령어 처리
case "${1:-start}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    dev)
        dev_mode
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
