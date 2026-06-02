import pandas as pd

from skku_vqa.data.validation import validate_submission


def test_valid(tmp_path):
    ref = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "label": [1, 1]}).to_csv(ref, index=False)
    pd.DataFrame({"sample_id": ["a", "b"], "label": [2, 3]}).to_csv(sub, index=False)
    assert validate_submission(sub, ref, "sample_id", "label") == []


def test_missing_column(tmp_path):
    ref = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"sample_id": ["a"], "label": [1]}).to_csv(ref, index=False)
    pd.DataFrame({"sample_id": ["a"]}).to_csv(sub, index=False)
    errs = validate_submission(sub, ref, "sample_id", "label")
    assert any("missing column" in e for e in errs)
