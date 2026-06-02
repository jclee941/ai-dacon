#!/usr/bin/env bash
# youtube 서버 환경 셋업. RTX 5070 Ti (Blackwell sm_120) 대응.
# 공식 wheel은 sm_120 커널 미포함 → cu128 pre-release torch를 vLLM보다 먼저 설치.
# 사용: scripts/remote/setup.sh [REMOTE_HOST] [REMOTE_DIR]
set -euo pipefail

REMOTE_HOST="${1:-youtube}"
REMOTE_DIR="${2:-~/ai-dacon}"

ssh "$REMOTE_HOST" bash -lc "'
  set -euo pipefail
  cd $REMOTE_DIR
  python3 -m venv .venv 2>/dev/null || true
  . .venv/bin/activate
  pip install -U pip setuptools wheel

  # 1) Blackwell sm_120: cu128 pre-release torch 먼저
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128

  # 2) 프로젝트 + 경량 의존성 (torch는 위에서 이미 설치됨)
  pip install -e . || true
  pip install pandas pyyaml pydantic pillow tqdm scikit-learn accelerate

  # 3) vLLM (pre-release) — torch 설치 이후
  pip install --pre vllm || pip install vllm

  # 4) fallback 대비
  pip install qwen-vl-utils bitsandbytes || true

  python -c \"import torch; print(\\\"torch\\\", torch.__version__, \\\"cuda\\\", torch.cuda.is_available(), \\\"cap\\\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)\"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'"
echo "[setup] done on $REMOTE_HOST"
