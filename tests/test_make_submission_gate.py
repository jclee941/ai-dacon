"""TDD: make_submission must refuse to submit code-fallback rows (parse_ok=false).

규칙 #5: a submitted label must be parsed from LLM-generated text. The label-0
last-resort fallback (parse_ok=false) is NOT an LLM-decided answer, so it must
never reach a submission CSV.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, exp: str) -> Path:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "defaults: configs/default.yaml\n"
        f"experiment_name: {exp}\n"
        f"paths:\n  output_dir: {tmp_path}\n",
        encoding="utf-8",
    )
    return cfg


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run(cfg: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/make_submission.py", "--config", str(cfg)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_make_submission_ok_when_all_parsed(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "exp_ok")
    _write_jsonl(
        tmp_path / "predictions" / "exp_ok" / "raw_outputs.jsonl",
        [
            {"sample_id": "s1", "label": 0, "parse_ok": True},
            {"sample_id": "s2", "label": 2, "parse_ok": True},
        ],
    )
    r = _run(cfg)
    assert r.returncode == 0, r.stderr + r.stdout
    out_csv = tmp_path / "submissions" / "exp_ok.csv"
    assert out_csv.exists()


def test_make_submission_rejects_parse_fail_rows(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "exp_bad")
    _write_jsonl(
        tmp_path / "predictions" / "exp_bad" / "raw_outputs.jsonl",
        [
            {"sample_id": "s1", "label": 0, "parse_ok": True},
            {"sample_id": "s2", "label": 0, "parse_ok": False},  # code fallback
        ],
    )
    r = _run(cfg)
    assert r.returncode != 0
    assert "s2" in (r.stdout + r.stderr)
    assert "parse_ok" in (r.stdout + r.stderr) or "parse" in (r.stdout + r.stderr).lower()
    assert not (tmp_path / "submissions" / "exp_bad.csv").exists()


def test_make_submission_rejects_missing_parse_ok(tmp_path: Path) -> None:
    # fail-closed: a row without parse_ok (legacy/malformed) must NOT be submitted.
    cfg = _write_config(tmp_path, "exp_missing")
    _write_jsonl(
        tmp_path / "predictions" / "exp_missing" / "raw_outputs.jsonl",
        [
            {"sample_id": "s1", "label": 0, "parse_ok": True},
            {"sample_id": "s2", "label": 1},  # parse_ok absent
        ],
    )
    r = _run(cfg)
    assert r.returncode != 0
    assert "s2" in (r.stdout + r.stderr)
    assert not (tmp_path / "submissions" / "exp_missing.csv").exists()
