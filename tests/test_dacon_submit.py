import pytest

from skku_vqa.submission.dacon import build_submission_args, load_env


def test_load_env_parses_token_and_competition(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nDACON_TOKEN=tok123\nDACON_COMPETITION_ID=236722\n",
        encoding="utf-8",
    )

    env = load_env(env_file)

    assert env["DACON_TOKEN"] == "tok123"
    assert env["DACON_COMPETITION_ID"] == "236722"


def test_build_submission_args_orders_positional_call() -> None:
    args = build_submission_args(
        submission_path="artifacts/final/qwen3vl_8b_hires.csv",
        token="tok123",
        competition_id="236722",
        team="qws941",
        memo="hires retry",
    )

    assert args == (
        "artifacts/final/qwen3vl_8b_hires.csv",
        "tok123",
        "236722",
        "qws941",
        "hires retry",
    )


def test_build_submission_args_requires_team() -> None:
    with pytest.raises(ValueError):
        build_submission_args(
            submission_path="x.csv",
            token="tok123",
            competition_id="236722",
            team="",
            memo="m",
        )
