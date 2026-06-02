from skku_vqa.evaluation.metrics import balanced_accuracy


def test_balanced_accuracy():
    y_true = [1, 2, 3, 3]
    y_pred = [1, 1, 3, 2]
    groups = ["ambiguous", "ambiguous", "disambiguated", "disambiguated"]
    out = balanced_accuracy(y_true, y_pred, groups)
    assert out["ambiguous_acc"] == 0.5
    assert out["disambiguated_acc"] == 0.5
    assert out["balanced_accuracy"] == 0.5
