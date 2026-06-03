"""로컬 stratified 검증 하니스 (분석 전용, 규칙 위반 아님).

학습셋 라벨(train.csv)과 예측(predictions.csv) 을 비교해 정확도/그룹별 정확도/
balanced accuracy/parse 실패율을 계산한다. group 컬럼이 있으면 ambiguous vs
disambiguated 로 층화해 평가한다. 모델 추론은 하지 않으며, 제출 라벨을 바꾸지 않는다.

train.csv 의 group 컬럼은 dataset.load_samples 가 채우지 않으므로 여기서 직접 읽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skku_vqa.evaluation.metrics import balanced_accuracy  # noqa: E402


def _truthy(v: object) -> bool:
    """parse_ok를 bool/문자열("True"/"true"/"False"/"false"/"0"/"1")에서 안전하게 해석."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "t"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--id-column", default="sample_id")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--group-column", default="group")
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    preds = pd.read_csv(args.predictions)

    if len(train) == 0:
        print("[validate_local] FAIL: train csv is empty", file=sys.stderr)
        sys.exit(1)


    idc, labc, grpc = args.id_column, args.label_column, args.group_column

    for col, df, name in ((idc, train, "train"), (labc, train, "train"),
                          (idc, preds, "predictions"), (labc, preds, "predictions")):
        if col not in df.columns:
            print(f"[validate_local] FAIL: '{col}' column missing in {name}", file=sys.stderr)
            sys.exit(1)

    train_ids = [str(x) for x in train[idc].tolist()]
    pred_ids = [str(x) for x in preds[idc].tolist()]
    dup = sorted({i for i in pred_ids if pred_ids.count(i) > 1})
    if dup:
        print(
            f"[validate_local] FAIL: duplicate prediction id(s): {', '.join(dup[:10])}",
            file=sys.stderr,
        )
        sys.exit(1)
    pred_map = {str(r[idc]): r for _, r in preds.iterrows()}

    missing = [i for i in train_ids if i not in pred_map]
    if missing:
        print(
            f"[validate_local] FAIL: predictions missing {len(missing)} id(s): "
            f"{', '.join(missing[:10])}",
            file=sys.stderr,
        )
        sys.exit(1)

    y_true = [int(t) for t in train[labc].tolist()]
    y_pred = [int(pred_map[i][labc]) for i in train_ids]

    has_parse_ok = "parse_ok" in preds.columns
    if has_parse_ok:
        flags = [_truthy(pred_map[i]["parse_ok"]) for i in train_ids]
        parse_fail_rate = sum(1 for f in flags if not f) / len(flags) if flags else 0.0
    else:
        parse_fail_rate = 0.0

    report: dict[str, object] = {
        "n": len(train_ids),
        "accuracy": sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true),
        "parse_fail_rate": parse_fail_rate,
    }

    if grpc in train.columns:
        groups = [str(g) for g in train[grpc].tolist()]
        report.update(balanced_accuracy(y_true, y_pred, groups))

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
