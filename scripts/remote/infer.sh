#!/usr/bin/env bash
# youtube 서버에서 원격 추론 실행 후 산출물을 로컬로 회수.
# 사용: scripts/remote/infer.sh CONFIG [REMOTE_HOST] [REMOTE_DIR]
set -euo pipefail

CONFIG="${1:?config path required (e.g. configs/experiment/baseline_llava_ov.yaml)}"
REMOTE_HOST="${2:-youtube}"
REMOTE_DIR="${3:-~/ai-dacon}"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# 1) 코드 동기화
"$(dirname "$0")/sync.sh" "$REMOTE_HOST" "$REMOTE_DIR"

# 2) 원격 추론
ssh "$REMOTE_HOST" bash -lc "'
  set -euo pipefail
  cd $REMOTE_DIR
  . .venv/bin/activate
  python scripts/run_inference.py --config $CONFIG
  python scripts/make_submission.py --config $CONFIG
'"

# 3) 산출물 회수
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/outputs/" "$LOCAL_DIR/outputs/"
echo "[infer] outputs synced back to $LOCAL_DIR/outputs/"
