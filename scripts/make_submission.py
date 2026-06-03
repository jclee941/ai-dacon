"""raw_outputs.jsonl → outputs/submissions/<exp>.csv (sample_id,label, UTF-8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skku_vqa.config import load_config  # noqa: E402
from skku_vqa.utils.io import ensure_dir, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg.paths.output_dir) / "predictions" / cfg.experiment_name
    rows = list(read_jsonl(pred_dir / "raw_outputs.jsonl"))

    # 규칙 #5: 최종 label은 LLM 생성 텍스트에서 파싱된 것이어야 한다. parse_ok 가 명시적
    # True가 아닌 행(False/누락/불립)은 코드 폴백(label 0)일 수 있으므로 제출하면 안 된다.
    # fail-closed: 증명된 LLM-파싱 행만 통과시킨다.
    bad = [str(r["sample_id"]) for r in rows if r.get("parse_ok") is not True]
    if bad:
        print(
            "[make_submission] FAIL: "
            f"{len(bad)} row(s) lack parse_ok=true (code fallback, not LLM-decided); "
            f"refusing to submit. ids: {', '.join(bad[:10])}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.DataFrame(
        {
            cfg.submission.id_column: [r["sample_id"] for r in rows],
            cfg.submission.answer_column: [r["label"] for r in rows],
        }
    )
    out_dir = ensure_dir(Path(cfg.paths.output_dir) / "submissions")
    out_path = out_dir / f"{cfg.experiment_name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[make_submission] {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
