"""DACON 제출 API 호출용 순수 헬퍼.

규칙: 외부 추론 API가 아니라 '제출' API다. 점수는 호출 결과에 포함되지 않고
DACON 웹 리더보드에서만 확인된다. dacon_submit_api는 격리 환경에 설치한다.
"""

from __future__ import annotations

from pathlib import Path


def load_env(env_path: str | Path = ".env") -> dict[str, str]:
    """.env 파일을 단순 KEY=VALUE 로 파싱한다 (주석/빈 줄 무시)."""
    text = Path(env_path).read_text(encoding="utf-8")
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def build_submission_args(
    submission_path: str,
    token: str,
    competition_id: str,
    team: str,
    memo: str = "",
) -> tuple[str, str, str, str, str]:
    """dacon_submit_api.post_submission_file 의 위치 인자 순서대로 구성한다."""
    if not submission_path:
        raise ValueError("submission_path is required")
    if not token:
        raise ValueError("token is required")
    if not competition_id:
        raise ValueError("competition_id is required")
    if not team:
        raise ValueError("team is required")
    return (submission_path, token, competition_id, team, memo)
