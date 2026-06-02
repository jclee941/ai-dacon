"""raw_outputs.jsonl → outputs/submissions/<exp>.csv (sample_id,label, UTF-8)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from skku_vqa.config import load_config
from skku_vqa.utils.io import ensure_dir, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg.paths.output_dir) / "predictions" / cfg.experiment_name
    rows = list(read_jsonl(pred_dir / "raw_outputs.jsonl"))

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
