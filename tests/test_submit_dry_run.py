"""Preflight (--dry-run) for submit_dacon.py: validate token/competition/file/import
WITHOUT calling the real DACON post (so we can verify before midnight that the timer
won't fail on all 5 candidates)."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".dacon_auto_venv" / "bin" / "python"
# dacon_submit_api lives only in the isolated venv; dry-run import check needs it.
PY_EXE = str(VENV_PY) if VENV_PY.exists() else sys.executable
_HAS_API = subprocess.run(
    [PY_EXE, "-c", "import dacon_submit_api"], capture_output=True
).returncode == 0
requires_api = pytest.mark.skipif(not _HAS_API, reason="dacon_submit_api not installed")


def _run(env_text: str, csv: str) -> subprocess.CompletedProcess[str]:
    env_file = REPO / "tests" / "_tmp_dryrun.env"
    env_file.write_text(env_text, encoding="utf-8")
    try:
        return subprocess.run(
            [PY_EXE, "scripts/submit_dacon.py", "--submission", csv,
             "--team", "qws941", "--env", str(env_file), "--dry-run"],
            cwd=REPO, capture_output=True, text=True,
        )
    finally:
        env_file.unlink(missing_ok=True)


@requires_api
def test_dry_run_ok_with_valid_env() -> None:
    r = _run("DACON_TOKEN=tok\nDACON_COMPETITION_ID=236722\n",
             "artifacts/final/qwen3vl_8b.csv")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "DRY-RUN OK" in r.stdout


def test_dry_run_fails_on_missing_token() -> None:
    r = _run("DACON_COMPETITION_ID=236722\n", "artifacts/final/qwen3vl_8b.csv")
    assert r.returncode != 0


def test_dry_run_fails_on_missing_file() -> None:
    r = _run("DACON_TOKEN=tok\nDACON_COMPETITION_ID=236722\n",
             "artifacts/final/does_not_exist.csv")
    assert r.returncode != 0
