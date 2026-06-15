"""Evaluation metrics and reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1_binary: float
    f1_macro: float
    roc_auc: float | None
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    y_true: list[int],
    y_pred: list[int],
    y_score: np.ndarray | None = None,
) -> EvalResult:
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = None
    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_binary=float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc=auc,
        confusion=cm,
    )


def text_report(y_true: list[int], y_pred: list[int]) -> str:
    return classification_report(
        y_true, y_pred, target_names=["non-hate", "hate"], digits=4, zero_division=0
    )
