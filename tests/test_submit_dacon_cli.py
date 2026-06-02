import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_submit_dacon_help_runs_from_repo_root_without_pythonpath() -> None:
    # Must import skku_vqa even when PYTHONPATH does not include src/.
    env = {"PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, "scripts/submit_dacon.py", "--help"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    assert "--submission" in proc.stdout
    assert "--team" in proc.stdout
    assert "ModuleNotFoundError" not in proc.stderr
