"""Evaluation metrics and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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


@dataclass
class MultiClassEvalResult:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    f1_per_class: list[float]
    precision_per_class: list[float]
    recall_per_class: list[float]
    confusion: list[list[int]]
    class_names: list[str] = field(default_factory=list)

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


def evaluate_multiclass(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> MultiClassEvalResult:
    """3-way (or general multi-class) evaluation.

    Returns macro and weighted F1 plus per-class precision/recall/F1.
    Class indices are assumed to be 0..K-1.
    """
    n_classes = max(max(y_true), max(y_pred)) + 1
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return MultiClassEvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_per_class=[float(x) for x in f1],
        precision_per_class=[float(x) for x in p],
        recall_per_class=[float(x) for x in r],
        confusion=cm,
        class_names=class_names or [str(i) for i in labels],
    )


def text_report(y_true: list[int], y_pred: list[int]) -> str:
    return classification_report(
        y_true, y_pred, target_names=["non-hate", "hate"], digits=4, zero_division=0
    )


def text_report_multiclass(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> str:
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
