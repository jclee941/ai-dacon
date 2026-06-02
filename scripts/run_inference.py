"""추론 실행: config → 예측 → outputs/predictions/<exp>/.

규칙 #6: 오프라인 실행 가능해야 함. 외부 API/인터넷 호출 없음.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skku_vqa.config import load_config
from skku_vqa.data.dataset import load_samples
from skku_vqa.inference.predictor import predict_sample
from skku_vqa.models.loader import load_model
from skku_vqa.utils.io import ensure_dir, write_jsonl
from skku_vqa.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="N 샘플만 추론 (스모크 테스트)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    samples = load_samples(cfg.paths.test_csv, cfg.paths.image_dir)
    if args.limit is not None:
        samples = samples[: args.limit]
    model = load_model(cfg.model)

    out_dir = ensure_dir(Path(cfg.paths.output_dir) / "predictions" / cfg.experiment_name)
    rows = []
    t0 = time.time()
    for s in samples:
        pred = predict_sample(model, s, cfg.prompt.template)
        rows.append(
            {
                "sample_id": pred.sample_id,
                "label": pred.label,
                "raw_output": pred.raw_output,
                "template": pred.template,
                "parse_ok": pred.parse_ok,
            }
        )
    elapsed = time.time() - t0

    write_jsonl(out_dir / "raw_outputs.jsonl", rows)
    (out_dir / "config.resolved.json").write_text(
        cfg.model_dump_json(indent=2), encoding="utf-8"
    )
    n = len(samples) or 1
    timing = {"n": len(samples), "elapsed_sec": elapsed, "sec_per_sample": elapsed / n}
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(
        f"[run_inference] {len(samples)} samples in {elapsed:.1f}s "
        f"({elapsed/n:.3f}s/sample) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
