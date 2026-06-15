"""Tests for evaluation metrics."""

from hsd.evaluate import evaluate


def test_evaluate_perfect():
    r = evaluate([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert r.accuracy == 1.0
    assert r.f1_binary == 1.0
    assert r.roc_auc == 1.0


def test_evaluate_all_wrong():
    r = evaluate([0, 1, 0, 1], [1, 0, 1, 0])
    assert r.accuracy == 0.0


def test_confusion_shape():
    r = evaluate([0, 0, 1, 1], [0, 1, 0, 1])
    assert len(r.confusion) == 2
    assert len(r.confusion[0]) == 2
