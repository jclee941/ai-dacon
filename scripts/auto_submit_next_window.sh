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
VENV="$REPO/.dacon_auto_venv"   # repo-local so it survives reboot (not /tmp)
STATE="$REPO/.dacon_auto_state" # idempotency markers: <name>.done per submitted file
mkdir -p "$STATE"

CANDIDATES=(
  "artifacts/final/qwen3vl_8b.csv"
  "artifacts/final/internvl3_8b.csv"
  "artifacts/final/qwen3vl_8b_hires.csv"
  "artifacts/final/qwen3vl_8b_lowres.csv"
  "artifacts/final/qwen3vl_8b_bgv2.csv"
)

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }

# Optional Telegram notification (opt-in). Silently skipped if .env lacks the keys,
# so this never breaks the submit flow. Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
notify() {
  local msg="$1" tok chat
  tok="$(grep -E '^TELEGRAM_BOT_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2-)"
  chat="$(grep -E '^TELEGRAM_CHAT_ID=' .env 2>/dev/null | head -1 | cut -d= -f2-)"
  [ -n "$tok" ] && [ -n "$chat" ] || return 0
  curl -fsS -m 20 -X POST "https://api.telegram.org/bot$tok/sendMessage" \
    -d chat_id="$chat" --data-urlencode text="$msg" >/dev/null 2>&1 || true
}

# Fatal exit with notification, so a midnight failure is never silent.
fail() { log "$1"; notify "DACON auto-submit FAILED: $1 (team $TEAM). See $LOG"; exit 1; }

log "auto-submit start; team=$TEAM repo=$REPO"

# Build isolated venv if missing
if [ ! -x "$VENV/bin/python" ]; then
  curl -fsSL -o "$WHL" "https://cfiles.dacon.co.kr/competitions/api/dacon_submit_api-0.1.2-py3-none-any.whl" || fail "wheel download failed"
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q "$WHL" || fail "venv/install failed"
fi

# Preflight: validate env/token/file/api for every candidate before posting anything.
# Catches a bad token, missing file, or broken venv up front instead of burning
# the daily-cap window on doomed submissions.
for c in "${CANDIDATES[@]}"; do
  if ! out="$("$VENV/bin/python" scripts/submit_dacon.py --submission "$c" --team "$TEAM" --dry-run 2>&1)"; then
    fail "PREFLIGHT FAILED for $(basename "$c"): $out"
  fi
  log "preflight ok: $(basename "$c")"
done

submit_one() {
  local csv="$1" name memo noncap=0
  name="$(basename "$csv")"
  memo="auto-submit next-window: $name"
  while true; do
    out="$("$VENV/bin/python" scripts/submit_dacon.py --submission "$csv" --team "$TEAM" --memo "$memo" 2>&1)"
    log "submit $name -> $out"
    if echo "$out" | grep -q "isSubmitted': True"; then
      touch "$STATE/$name.done" || { log "WARN failed to write idempotency marker for $name"; return 1; }
      return 0
    fi
    if echo "$out" | grep -qi "Over max submission count"; then
      log "daily cap not reset yet; retry in 10min"
      sleep 600
      continue
    fi
    # non-cap error (transient network/API): retry a few times before giving up
    noncap=$((noncap + 1))
    if [ "$noncap" -le 5 ]; then
      log "non-cap error for $name (attempt $noncap/5); retry in 60s"
      sleep 60
      continue
    fi
    log "non-cap error for $name persisted after 5 retries; giving up on this file"
    return 1
  done
}

failures=0
for c in "${CANDIDATES[@]}"; do
  [ -f "$c" ] || { log "MISSING $c"; failures=$((failures + 1)); continue; }
  if [ -f "$STATE/$(basename "$c").done" ]; then
    log "SKIP $(basename "$c") (already submitted; idempotency marker present)"
    continue
  fi

  submit_one "$c" || failures=$((failures + 1))
  sleep 5
done

if [ "$failures" -ne 0 ]; then
  log "auto-submit FAILED for $failures candidate(s)"
  notify "DACON auto-submit FAILED for $failures candidate(s) (team $TEAM). See $LOG"
  exit 1
fi
log "auto-submit done; all candidates submitted or already marked"
notify "DACON auto-submit OK: all 5 candidates submitted (team $TEAM)."
