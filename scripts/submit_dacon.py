"""DACON 제출 CLI. .env(DACON_TOKEN/DACON_COMPETITION_ID) + --team 으로 재현 가능 제출.

dacon_submit_api 는 PEP 668 때문에 격리 venv 에 설치해야 할 수 있다:
  python -m venv /tmp/dacon_venv
  /tmp/dacon_venv/bin/pip install <dacon_submit_api-*.whl>
  /tmp/dacon_venv/bin/python scripts/submit_dacon.py --submission <csv> --team <name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skku_vqa.submission.dacon import build_submission_args, load_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--memo", default="")
    args = ap.parse_args()

    env = load_env(args.env)
    token = env.get("DACON_TOKEN", "")
    comp = env.get("DACON_COMPETITION_ID", "")
    call_args = build_submission_args(args.submission, token, comp, args.team, args.memo)

    from dacon_submit_api import dacon_submit_api

    result = dacon_submit_api.post_submission_file(*call_args)
    print(f"[submit_dacon] {result}")
    if not result.get("isSubmitted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
