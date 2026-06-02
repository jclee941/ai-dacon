#!/usr/bin/env bash
# Auto-submit the next-window top-5 DACON candidates once the daily cap resets.
# Self-contained: builds an isolated venv with dacon_submit_api, then submits each
# candidate, retrying every 10 min while the daily-cap error persists.
#
# Usage: scripts/auto_submit_next_window.sh [TEAM]
# Logs to: /tmp/dacon_auto_submit.log
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
TEAM="${1:-qws941}"
LOG=/tmp/dacon_auto_submit.log
WHL=/tmp/dacon_submit_api-0.1.2-py3-none-any.whl
VENV=/tmp/dacon_auto_venv

CANDIDATES=(
  "artifacts/final/qwen3vl_8b.csv"
  "artifacts/final/internvl3_8b.csv"
  "artifacts/final/qwen3vl_8b_hires.csv"
  "artifacts/final/qwen3vl_8b_lowres.csv"
  "artifacts/final/qwen25vl_7b.csv"
)

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }

log "auto-submit start; team=$TEAM repo=$REPO"

# Build isolated venv if missing
if [ ! -x "$VENV/bin/python" ]; then
  curl -fsSL -o "$WHL" "https://cfiles.dacon.co.kr/competitions/api/dacon_submit_api-0.1.2-py3-none-any.whl" || { log "wheel download failed"; exit 1; }
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q "$WHL" || { log "venv/install failed"; exit 1; }
fi

submit_one() {
  local csv="$1" name memo
  name="$(basename "$csv")"
  memo="auto-submit next-window: $name"
  while true; do
    out="$("$VENV/bin/python" scripts/submit_dacon.py --submission "$csv" --team "$TEAM" --memo "$memo" 2>&1)"
    log "submit $name -> $out"
    if echo "$out" | grep -q "isSubmitted': True"; then
      return 0
    fi
    if echo "$out" | grep -qi "Over max submission count"; then
      log "daily cap not reset yet; retry in 10min"
      sleep 600
      continue
    fi
    log "non-cap error for $name; giving up on this file"
    return 1
  done
}

for c in "${CANDIDATES[@]}"; do
  [ -f "$c" ] || { log "MISSING $c"; continue; }
  submit_one "$c"
  sleep 5
done

log "auto-submit done"
