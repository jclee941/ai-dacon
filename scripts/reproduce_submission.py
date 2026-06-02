"""최종 제출 재현: artifacts/final/config.yaml → 추론 → 제출 → 검증."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="artifacts/final/config.yaml")
    args = ap.parse_args()

    py = sys.executable
    run([py, "scripts/run_inference.py", "--config", args.config])
    run([py, "scripts/make_submission.py", "--config", args.config])
    print("[reproduce] done. validate with scripts/validate_submission.py")


if __name__ == "__main__":
    main()
