"""제출 파일 형식 검증 (sample_id,label / 행수 / ID 일치 / null)."""

from __future__ import annotations

import argparse
import sys

from skku_vqa.data.validation import validate_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample-submission", default="data/raw/sample_submission.csv")
    ap.add_argument("--id-column", default="sample_id")
    ap.add_argument("--answer-column", default="label")
    args = ap.parse_args()

    errors = validate_submission(
        args.submission, args.sample_submission, args.id_column, args.answer_column
    )
    if errors:
        print("[validate] FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("[validate] OK")


if __name__ == "__main__":
    main()
