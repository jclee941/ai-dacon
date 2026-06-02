"""CSV → Sample 로딩 (DACON #236722 실제 스키마).

test.csv 컬럼: sample_id, image_path, context, question, answers
  - image_path: './images/test_img_XXXX.jpg' (csv 위치 기준 상대경로)
  - answers: JSON 리스트 문자열, 예: '["A", "B", "Not enough information"]'
train.csv: 위 + label (0-based 정답 인덱스)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import Sample


def _parse_answers(raw: str) -> list[str]:
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x).strip() for x in val]
    except json.JSONDecodeError:
        pass
    return [raw]


def _resolve_image(image_path: str, csv_dir: Path) -> str | None:
    if not isinstance(image_path, str) or not image_path.strip():
        return None
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    # csv가 있는 디렉토리 기준으로 './images/..' 해석
    return str((csv_dir / p).resolve())


def load_samples(csv_path: str | Path, image_dir: str | Path | None = None) -> list[Sample]:
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent
    df = pd.read_csv(csv_path)

    samples: list[Sample] = []
    for _, row in df.iterrows():
        options = _parse_answers(row.get("answers", ""))
        image_path = _resolve_image(row.get("image_path", ""), csv_dir)
        has_ctx = "context" in row and pd.notna(row["context"])
        context = str(row["context"]).strip() if has_ctx else None
        label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else None

        samples.append(
            Sample(
                id=str(row["sample_id"]),
                question=str(row.get("question", "")).strip(),
                options=options,
                context=context,
                image_path=image_path,
                answer_idx=label,  # 0-based
            )
        )
    return samples
