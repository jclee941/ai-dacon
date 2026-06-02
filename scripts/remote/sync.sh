#!/usr/bin/env bash
# 로컬 → youtube 서버로 코드/설정 동기화.
# 기본은 코드만. WITH_DATA=1 이면 대회 test 데이터(csv+images)도 동기화.
# 사용: [WITH_DATA=1] scripts/remote/sync.sh [REMOTE_HOST] [REMOTE_DIR]
set -euo pipefail

REMOTE_HOST="${1:-youtube}"
REMOTE_DIR="${2:-~/ai-dacon}"
WITH_DATA="${WITH_DATA:-0}"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)/"

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# 코드 동기화 (대용량/생성물 제외)
rsync -avz --delete \
  --exclude '.git' \
  --exclude 'open.zip' \
  --exclude 'data/raw' \
  --exclude 'data/interim' \
  --exclude 'data/processed' \
  --exclude 'outputs' \
  --exclude 'artifacts/lora_adapters' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude '*.pyc' \
  "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR/"

echo "[sync] code -> $REMOTE_HOST:$REMOTE_DIR"

# 데이터 동기화 (옵션): 추론에 필요한 test.csv, images, sample_submission
if [ "$WITH_DATA" = "1" ]; then
  echo "[sync] WITH_DATA=1 -> syncing competition test data (this is large)"
  ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/data/raw/test/images"
  rsync -avz "${LOCAL_DIR}data/raw/test/test.csv" "$REMOTE_HOST:$REMOTE_DIR/data/raw/test/test.csv"
  rsync -avz "${LOCAL_DIR}data/raw/sample_submission.csv" "$REMOTE_HOST:$REMOTE_DIR/data/raw/sample_submission.csv"
  rsync -avz --delete "${LOCAL_DIR}data/raw/test/images/" "$REMOTE_HOST:$REMOTE_DIR/data/raw/test/images/"
  echo "[sync] test data -> $REMOTE_HOST:$REMOTE_DIR/data/raw/test"
fi
