"""평가 지표: Balanced Accuracy (ambiguous / disambiguated 그룹 평균)."""

from __future__ import annotations

from collections.abc import Sequence


def group_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    mask: Sequence[bool],
) -> float | None:
    n = sum(1 for m in mask if m)
    if n == 0:
        return None
    correct = sum(1 for t, p, m in zip(y_true, y_pred, mask, strict=False) if m and t == p)
    return correct / n


def balanced_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    ambiguous_label: str = "ambiguous",
    disambiguated_label: str = "disambiguated",
) -> dict[str, float | None]:
    """그룹별 정확도와 그 평균(balanced accuracy)을 반환."""
    amb_mask = [g == ambiguous_label for g in groups]
    dis_mask = [g == disambiguated_label for g in groups]
    amb = group_accuracy(y_true, y_pred, amb_mask)
    dis = group_accuracy(y_true, y_pred, dis_mask)
    parts = [v for v in (amb, dis) if v is not None]
    bal = sum(parts) / len(parts) if parts else None
    return {"ambiguous_acc": amb, "disambiguated_acc": dis, "balanced_accuracy": bal}
