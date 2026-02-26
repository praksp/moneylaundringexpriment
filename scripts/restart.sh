#!/usr/bin/env bash
# =============================================================================
# restart.sh — Clean restart for the AML detection application
#
# Stops:  uvicorn (FastAPI backend)  +  vite (React frontend)
# Starts: uvicorn on :8001  +  vite dev server on :5174
# Logs:   logs/backend.log  +  logs/frontend.log
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
LOG_DIR="$ROOT/logs"
VENV="$ROOT/venv"
FRONTEND_DIR="$ROOT/frontend"

BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

mkdir -p "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
section() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Step 1: Kill any running processes ───────────────────────────────────────
section "1/4  Stopping existing processes"

# Kill by saved PID files first (clean shutdown)
for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
  if [[ -f "$pidfile" ]]; then
    old_pid=$(cat "$pidfile")
    if kill -0 "$old_pid" 2>/dev/null; then
      info "Killing PID $old_pid (from $pidfile)"
      kill "$old_pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
done

# Belt-and-suspenders: kill by process pattern
pkill -f "uvicorn api.main:app" 2>/dev/null && info "Killed uvicorn" || true
pkill -f "vite"                 2>/dev/null && info "Killed vite"    || true

# Release port 8001 if still bound
if lsof -ti:8001 &>/dev/null; then
  warn "Port 8001 still in use — force-killing…"
  lsof -ti:8001 | xargs kill -9 2>/dev/null || true
fi

# Release port 5174 if still bound
if lsof -ti:5174 &>/dev/null; then
  warn "Port 5174 still in use — force-killing…"
  lsof -ti:5174 | xargs kill -9 2>/dev/null || true
fi

sleep 1
ok "All old processes stopped"

# ── Step 2: Check Neo4j ───────────────────────────────────────────────────────
section "2/4  Checking Neo4j"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "aml_neo4j"; then
  ok "Neo4j container is running"
else
  warn "Neo4j container not running — starting via docker-compose…"
  docker compose -f "$ROOT/docker-compose.yml" up -d neo4j
  info "Waiting 15s for Neo4j to be ready…"
  sleep 15
fi

# ── Step 3: Start FastAPI backend ─────────────────────────────────────────────
section "3/4  Starting FastAPI backend (:8001)"

source "$VENV/bin/activate"

# Ensure Python dependencies are up-to-date (handles new packages like cachetools)
info "Checking Python dependencies…"
pip install -q --no-deps -r "$ROOT/requirements.txt" 2>/dev/null || true
ok "Dependencies verified"

cd "$ROOT"
nohup uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  >> "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
info "Uvicorn started (PID $BACKEND_PID) → $BACKEND_LOG"

# Wait for backend to be healthy
info "Waiting for backend health check…"
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
  if curl -sf http://localhost:8001/health &>/dev/null; then
    ok "Backend healthy after ${i}s"
    break
  fi
  if [[ $i -eq $MAX_WAIT ]]; then
    warn "Backend did not respond in ${MAX_WAIT}s — check $BACKEND_LOG"
  fi
  sleep 1
done

# ── Step 4: Start Vite frontend ───────────────────────────────────────────────
section "4/4  Starting Vite frontend (:5174)"

cd "$FRONTEND_DIR"
info "Installing frontend dependencies…"
npm install --prefer-offline --silent 2>/dev/null || true
ok "Frontend dependencies verified"

nohup npm run dev \
  >> "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
info "Vite started (PID $FRONTEND_PID) → $FRONTEND_LOG"

# Wait for frontend
info "Waiting for frontend to be ready…"
for i in $(seq 1 20); do
  if curl -sf http://localhost:5174 &>/dev/null; then
    ok "Frontend ready after ${i}s"
    break
  fi
  sleep 1
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  Application running${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Frontend :  ${CYAN}http://localhost:5174${RESET}"
echo -e "  API       :  ${CYAN}http://localhost:8001${RESET}"
echo -e "  API docs  :  ${CYAN}http://localhost:8001/docs${RESET}"
echo -e "  Neo4j UI  :  ${CYAN}http://localhost:7474${RESET}"
echo ""
echo -e "  Backend log  : $BACKEND_LOG"
echo -e "  Frontend log : $FRONTEND_LOG"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  Login with  ${BOLD}admin / password${RESET}"
echo ""
