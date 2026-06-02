"""제출 파일 형식 검증."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_submission(
    submission_path: str | Path,
    sample_submission_path: str | Path,
    id_column: str = "sample_id",
    answer_column: str = "label",
) -> list[str]:
    """제출 파일을 sample_submission 기준으로 검증한다. 문제 목록을 반환(빈 리스트=통과)."""
    errors: list[str] = []
    sub = pd.read_csv(submission_path)
    ref = pd.read_csv(sample_submission_path)

    for col in (id_column, answer_column):
        if col not in sub.columns:
            errors.append(f"missing column: {col}")
    if errors:
        return errors

    if len(sub) != len(ref):
        errors.append(f"row count mismatch: submission={len(sub)} expected={len(ref)}")

    if id_column in ref.columns:
        ref_ids = set(ref[id_column].astype(str))
        sub_ids = set(sub[id_column].astype(str))
        if ref_ids != sub_ids:
            missing = len(ref_ids - sub_ids)
            extra = len(sub_ids - ref_ids)
            errors.append(f"ID set mismatch: missing={missing} extra={extra}")

    if sub[answer_column].isna().any():
        errors.append(f"null answers: {int(sub[answer_column].isna().sum())}")

    return errors
