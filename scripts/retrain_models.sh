#!/usr/bin/env bash
# =============================================================================
# scripts/retrain_models.sh  —  Clean & retrain all AML models
# =============================================================================
# Models trained:
#   1. XGBoost (aml_model.joblib + xgb_model_fp32.onnx)
#   2. SGD/SVM  (svm_model.joblib + svm_model_fp32.onnx)
#   3. KNN Anomaly Detector (anomaly_index.faiss + anomaly_meta.pkl)
#   4. GraphSAGE Mule Detector (graphsage_model.npz + graphsage_meta.pkl)
#   + training_metadata.json for drift detection
#
# Usage:
#   bash scripts/retrain_models.sh               # retrain all
#   bash scripts/retrain_models.sh --skip-sage   # skip GraphSAGE (slow)
#   bash scripts/retrain_models.sh --skip-anomaly # skip KNN anomaly detector
#   bash scripts/retrain_models.sh --clean-only   # wipe models, don't retrain
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_DIR/models_saved"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"
VENV="$REPO_DIR/venv"
PYTHON="$VENV/bin/python3"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Parse flags ────────────────────────────────────────────────────────────────
SKIP_SAGE=false
SKIP_ANOMALY=false
CLEAN_ONLY=false

for arg in "$@"; do
  case $arg in
    --skip-sage)    SKIP_SAGE=true ;;
    --skip-anomaly) SKIP_ANOMALY=true ;;
    --clean-only)   CLEAN_ONLY=true ;;
    *)              warn "Unknown flag: $arg" ;;
  esac
done

mkdir -p "$LOG_DIR"

echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  AML Model Retrain Pipeline${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo "  Log : $LOG_FILE  (tail -f $LOG_FILE)"
echo "  Time: $(date)"
echo ""

# ── Step 1: Verify prerequisites ───────────────────────────────────────────────
section "1/5  Checking prerequisites"

if [[ ! -f "$PYTHON" ]]; then
  error "Python not found at $PYTHON — run: python3 -m venv $VENV && $VENV/bin/pip install -r requirements.txt"
  exit 1
fi
ok "Python: $($PYTHON --version)"

# Verify Neo4j is reachable
info "Checking Neo4j connection…"
"$PYTHON" -c "
from db.client import neo4j_session
with neo4j_session() as s:
    n = s.run('MATCH (t:Transaction) RETURN count(t) AS n').single()['n']
    print(f'Neo4j connected — {n:,} transactions')
" || { error "Cannot connect to Neo4j. Make sure the container is running."; exit 1; }
ok "Neo4j reachable"

# ── Step 2: Clean existing model files ─────────────────────────────────────────
section "2/5  Cleaning existing models"

MODEL_FILES=(
  "$MODELS_DIR/aml_model.joblib"
  "$MODELS_DIR/aml_scaler.joblib"
  "$MODELS_DIR/svm_model.joblib"
  "$MODELS_DIR/svm_model_fp32.onnx"
  "$MODELS_DIR/xgb_model_fp32.onnx"
  "$MODELS_DIR/knn_model.joblib"          # legacy large file
  "$MODELS_DIR/anomaly_index.faiss"
  "$MODELS_DIR/anomaly_meta.pkl"
  "$MODELS_DIR/graphsage_model.npz"
  "$MODELS_DIR/graphsage_meta.pkl"
  "$MODELS_DIR/training_metadata.json"
)

for f in "${MODEL_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    rm -f "$f"
    info "  Removed: $(basename "$f")"
  fi
done
ok "Model directory cleaned"

if [[ "$CLEAN_ONLY" == "true" ]]; then
  echo ""
  ok "Clean-only mode: done. Models not retrained."
  exit 0
fi

# ── Step 3: Invalidate API caches (restart backend if running) ─────────────────
section "3/5  Clearing API caches"

BACKEND_PID_FILE="$REPO_DIR/logs/backend.pid"
BACKEND_WAS_RUNNING=false

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  info "Backend is running (PID $(cat "$BACKEND_PID_FILE")) — will restart after training"
  BACKEND_WAS_RUNNING=true
fi

ok "Cache will be refreshed at backend restart"

# ── Step 4: Retrain all models ──────────────────────────────────────────────────
section "4/5  Training models"

# Prepare env vars to control which models are trained
EXTRA_ENV=""
if [[ "$SKIP_SAGE" == "true" ]];    then EXTRA_ENV="$EXTRA_ENV ENABLE_GRAPHSAGE=false"; fi
if [[ "$SKIP_ANOMALY" == "true" ]]; then EXTRA_ENV="$EXTRA_ENV ENABLE_KNN_ANOMALY=false"; fi

START_TIME=$SECONDS

info "Running train_and_save_all()…"
info "  (This may take 5–30 min depending on dataset size)"
echo ""

env $EXTRA_ENV "$PYTHON" -c "
import sys, os
sys.path.insert(0, '${REPO_DIR}')

from rich.console import Console
console = Console()

console.print('[bold cyan]═══════════════════════════════════════[/]')
console.print('[bold cyan]  AML Full Model Training Pipeline     [/]')
console.print('[bold cyan]═══════════════════════════════════════[/]')

from ml.train import train_and_save_all
metrics = train_and_save_all()

console.print('')
console.print('[bold green]════════  Training Complete  ════════[/]')
for model, m in metrics.items():
    if isinstance(m, dict) and 'error' not in m:
        auc   = m.get('roc_auc', m.get('n_normal_vectors', '—'))
        label = f'ROC-AUC={auc}' if 'roc_auc' in m else f'vectors={auc}'
        console.print(f'  [green]✓[/] {model.upper():<12} {label}')
    elif isinstance(m, dict) and 'error' in m:
        console.print(f'  [yellow]⚠[/] {model.upper():<12} FAILED: {m[\"error\"]}')
"

ELAPSED=$((SECONDS - START_TIME))
echo ""
ok "Training finished in ${ELAPSED}s ($(date -u -r $ELAPSED '+%M min %S sec' 2>/dev/null || echo "${ELAPSED}s"))"

# ── Step 5: Scan accounts with new anomaly model ────────────────────────────────
section "5/5  Post-train tasks"

if [[ "$SKIP_ANOMALY" != "true" ]]; then
  info "Scanning accounts with new anomaly detector (5,000 accounts)…"
  "$PYTHON" -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from ml.anomaly import get_detector, reset_detector
import time
reset_detector()
d = get_detector()
if d.is_trained:
    t0 = time.time()
    suspects = d.scan_all_accounts(batch_size=500, force=True, max_accounts=5000)
    print(f'  Scanned 5,000 accounts in {time.time()-t0:.1f}s — {len(suspects)} mule suspects flagged')
else:
    print('  [WARN] Anomaly detector not trained — skipping scan')
  "
  ok "Account scan complete"
fi

# ── Restart backend if it was running ──────────────────────────────────────────
if [[ "$BACKEND_WAS_RUNNING" == "true" ]]; then
  info "Restarting backend to load new models…"
  bash "$REPO_DIR/scripts/restart.sh"
else
  echo ""
  info "Backend was not running."
  info "Start it with:  bash scripts/restart.sh"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  All models retrained — new baseline version registered${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo "  Saved models:"
ls -lh "$MODELS_DIR"/*.joblib "$MODELS_DIR"/*.onnx \
        "$MODELS_DIR"/*.pkl    "$MODELS_DIR"/*.faiss \
        "$MODELS_DIR"/*.npz    "$MODELS_DIR"/*.json 2>/dev/null \
  | awk '{printf "  %-40s %s\n", $NF, $5}'
echo ""
echo "  Full log: $LOG_FILE"
echo ""
