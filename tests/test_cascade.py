"""Smoke tests for the Cascade wrapper."""

import numpy as np
from hsd.cascade import Cascade


class _FakeSklearn:
    """Simulates sklearn Pipeline.predict_proba returning (n, 2)."""

    def __init__(self, p_hate: list[float]):
        self._p = np.asarray(p_hate, dtype=float)

    def predict_proba(self, texts):
        n = len(texts)
        p = self._p[:n]
        return np.stack([1.0 - p, p], axis=1)


class _FakeHsd:
    """Simulates hsd model.predict_proba returning (n,)."""

    def __init__(self, p_hate: list[float]):
        self._p = np.asarray(p_hate, dtype=float)

    def predict_proba(self, texts):
        return self._p[: len(texts)]


def test_cascade_routes_high_confidence_to_stage2():
    stage1 = _FakeSklearn([0.05, 0.60, 0.90, 0.10])
    stage2 = _FakeHsd([0.99, 0.01, 0.99])  # only 3 samples asked
    casc = Cascade(stage1, stage2, "tfidf", "distilbert", stage1_threshold=0.30)
    p = casc.predict_proba(["a", "b", "c", "d"])
    assert casc.stats.n_total == 4
    # samples with p1 >= 0.30 are indices 1, 2 → 2 routed
    assert casc.stats.n_routed_stage2 == 2
    # index 0 keeps stage-1 score (0.05)
    assert p[0] == 0.05
    # index 1 got re-scored by stage 2 → 0.99 (first fake stage-2 value)
    assert p[1] == 0.99


def test_cascade_all_bypass_stage2():
    stage1 = _FakeSklearn([0.01, 0.02, 0.03])
    stage2 = _FakeHsd([0.99, 0.99, 0.99])
    casc = Cascade(stage1, stage2, "tfidf", "distilbert", stage1_threshold=0.50)
    p = casc.predict_proba(["a", "b", "c"])
    assert casc.stats.n_routed_stage2 == 0
    assert list(p) == [0.01, 0.02, 0.03]


def test_cascade_latency_weighting():
    stage1 = _FakeSklearn([0.05, 0.60, 0.90, 0.10])
    stage2 = _FakeHsd([0.99, 0.01])
    casc = Cascade(stage1, stage2, "tfidf", "distilbert", stage1_threshold=0.30)
    casc.predict_proba(["a", "b", "c", "d"])
    # 2 out of 4 routed → rate 0.5. Latency = 1 + 0.5 * 10 = 6
    assert abs(casc.estimated_latency_ms(1.0, 10.0) - 6.0) < 1e-9
