"""데이터 점검: 원본 파일 존재 확인 + 샘플 파싱 sanity check."""

from __future__ import annotations

import argparse
from pathlib import Path

from skku_vqa.config import load_config
from skku_vqa.data.dataset import load_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    test_csv = Path(cfg.paths.test_csv)
    if not test_csv.exists():
        print(f"[prepare] test csv not found: {test_csv} (데이터 다운로드 필요)")
        return

    samples = load_samples(test_csv, cfg.paths.image_dir)
    print(f"[prepare] loaded {len(samples)} samples")
    if samples:
        s = samples[0]
        print(f"  id={s.id} options={len(s.options)} has_image={s.image_path is not None}")
        print(f"  context={'Y' if s.context else 'N'} question={s.question[:60]!r}")


if __name__ == "__main__":
    main()
