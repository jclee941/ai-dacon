"""TDD: scripts/validate_local.py — local stratified train-set eval (analysis only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "validate_local.py"

TRAIN_HEADER = "sample_id,image_path,context,question,answers,label"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _run(train: Path, preds: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--train-csv", str(train), "--predictions", str(preds)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_validate_local_reports_balanced_accuracy_and_parse_fail_rate(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(
        train,
        TRAIN_HEADER
        + ",group\n"
        + 's1,./i1.jpg,ctx,q1,"[""A"",""B"",""C""]",0,ambiguous\n'
        + 's2,./i2.jpg,ctx,q2,"[""A"",""B"",""C""]",1,ambiguous\n'
        + 's3,./i3.jpg,ctx,q3,"[""A"",""B"",""C""]",2,disambiguated\n'
        + 's4,./i4.jpg,ctx,q4,"[""A"",""B"",""C""]",2,disambiguated\n',
    )
    preds = tmp_path / "preds.csv"
    # s1 correct, s2 wrong, s3 correct, s4 wrong; s4 parse failed
    _write(
        preds,
        "sample_id,label,parse_ok\n"
        "s1,0,True\n"
        "s2,2,True\n"
        "s3,2,True\n"
        "s4,0,False\n",
    )
    r = _run(train, preds)
    assert r.returncode == 0, r.stderr + r.stdout
    report = json.loads(r.stdout)
    assert report["ambiguous_acc"] == 0.5
    assert report["disambiguated_acc"] == 0.5
    assert report["balanced_accuracy"] == 0.5
    assert report["parse_fail_rate"] == 0.25


def test_validate_local_without_group_reports_accuracy_only(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(
        train,
        TRAIN_HEADER
        + "\n"
        + 's1,./i1.jpg,ctx,q1,"[""A"",""B""]",0\n'
        + 's2,./i2.jpg,ctx,q2,"[""A"",""B""]",1\n',
    )
    preds = tmp_path / "preds.csv"
    _write(preds, "sample_id,label\ns1,0\ns2,0\n")
    r = _run(train, preds)
    assert r.returncode == 0, r.stderr + r.stdout
    report = json.loads(r.stdout)
    assert report["accuracy"] == 0.5
    assert "parse_fail_rate" in report


def test_validate_local_fails_on_missing_prediction_id(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(
        train,
        TRAIN_HEADER
        + "\n"
        + 's1,./i1.jpg,ctx,q1,"[""A"",""B""]",0\n'
        + 's2,./i2.jpg,ctx,q2,"[""A"",""B""]",1\n',
    )
    preds = tmp_path / "preds.csv"
    _write(preds, "sample_id,label\ns1,0\n")  # missing s2
    r = _run(train, preds)
    assert r.returncode != 0
    assert "s2" in (r.stdout + r.stderr)


def test_validate_local_parses_string_parse_ok_flags(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(
        train,
        TRAIN_HEADER
        + "\n"
        + 's1,./i1.jpg,ctx,q1,"[""A"",""B""]",0\n'
        + 's2,./i2.jpg,ctx,q2,"[""A"",""B""]",1\n'
        + 's3,./i3.jpg,ctx,q3,"[""A"",""B""]",0\n'
        + 's4,./i4.jpg,ctx,q4,"[""A"",""B""]",1\n',
    )
    preds = tmp_path / "preds.csv"
    # parse_ok given as strings (as written by pandas/jsonl round-trips)
    _write(
        preds,
        "sample_id,label,parse_ok\n"
        "s1,0,true\n"
        "s2,1,True\n"
        "s3,0,false\n"
        "s4,1,False\n",
    )
    r = _run(train, preds)
    assert r.returncode == 0, r.stderr + r.stdout
    report = json.loads(r.stdout)
    assert report["parse_fail_rate"] == 0.5


def test_validate_local_fails_on_empty_train(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(train, TRAIN_HEADER + "\n")
    preds = tmp_path / "preds.csv"
    _write(preds, "sample_id,label\n")
    r = _run(train, preds)
    assert r.returncode != 0
    assert "empty" in (r.stdout + r.stderr).lower()


def test_validate_local_fails_on_duplicate_prediction_id(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write(
        train,
        TRAIN_HEADER
        + "\n"
        + 's1,./i1.jpg,ctx,q1,"[""A"",""B""]",0\n',
    )
    preds = tmp_path / "preds.csv"
    _write(preds, "sample_id,label\ns1,0\ns1,1\n")  # duplicate s1
    r = _run(train, preds)
    assert r.returncode != 0
    assert "duplicate" in (r.stdout + r.stderr).lower()
