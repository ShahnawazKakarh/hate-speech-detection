"""Two-stage cascade classifier.

Wraps any two already-trained ``hsd`` models as a cheap-prefilter + expensive-
verifier pipeline. The stage-1 model scores every sample; samples whose
stage-1 hate probability meets ``stage1_threshold`` are re-scored by the
stage-2 model and their stage-2 score is used as the final output. Cheap
samples never touch stage 2, giving the throughput-vs-accuracy trade-off
motivated in Section 5.4 of the paper.

Both stages must expose the same probability interface used by the other
``hsd`` models: either a ``predict_proba(texts) -> array-of-p-hate`` method,
or the scikit-learn ``predict_proba`` returning a 2-column matrix (used by
the TF-IDF pipeline).

Example:
    from hsd.cascade import Cascade
    from hsd.models.tfidf import load as load_tfidf
    from hsd.models.distilbert import DistilBertClassifier

    pipe = load_tfidf("artifacts/tfidf_davidson/pipeline.joblib")
    bert = DistilBertClassifier.load("artifacts/distilbert_davidson/final")
    cascade = Cascade(
        stage1=pipe, stage1_type="tfidf",
        stage2=bert, stage2_type="distilbert",
        stage1_threshold=0.20,
    )
    p_hate = cascade.predict_proba(texts)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CascadeStats:
    """Per-inference bookkeeping."""

    n_total: int = 0
    n_routed_stage2: int = 0
    stage1_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stage2_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    routed_to_stage2: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def stage2_rate(self) -> float:
        return float(self.n_routed_stage2 / self.n_total) if self.n_total else 0.0


def _score(model, model_type: str, texts: list[str]) -> np.ndarray:
    """Extract p(hate) as a 1-D float array from any of the hsd model wrappers."""
    if model_type == "tfidf":
        # sklearn Pipeline: predict_proba returns (n, 2), column 1 is p(hate)
        return model.predict_proba(texts)[:, 1]
    # Doc2VecClassifier and DistilBertClassifier both return a 1-D array directly
    return model.predict_proba(texts)


class Cascade:
    """Two-stage cascade.

    Parameters
    ----------
    stage1, stage2
        Trained model objects. Any object with a compatible ``predict_proba``
        (either sklearn 2-column or hsd 1-column) works.
    stage1_type, stage2_type
        One of ``"tfidf"``, ``"doc2vec"``, ``"distilbert"``. Selects the
        probability-extraction shim.
    stage1_threshold
        Samples with ``p_stage1 >= stage1_threshold`` are re-scored by stage 2.
        Lower thresholds route more traffic to stage 2 (higher accuracy,
        lower throughput).
    """

    def __init__(
        self,
        stage1,
        stage2,
        stage1_type: str = "tfidf",
        stage2_type: str = "distilbert",
        stage1_threshold: float = 0.20,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage1_type = stage1_type
        self.stage2_type = stage2_type
        self.stage1_threshold = float(stage1_threshold)
        self.stats: CascadeStats | None = None

    # ------------------------------------------------------------------ #
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return p(hate) for each text, using stage 1 unless routed to stage 2."""
        texts = list(texts)
        n = len(texts)

        s1 = _score(self.stage1, self.stage1_type, texts)
        routed = s1 >= self.stage1_threshold
        final = s1.copy()

        s2_full = np.zeros(n, dtype=float)  # kept for stats; 0 where not scored
        idx = np.where(routed)[0]
        if len(idx):
            s2_texts = [texts[i] for i in idx]
            s2 = _score(self.stage2, self.stage2_type, s2_texts)
            final[idx] = s2
            s2_full[idx] = s2

        self.stats = CascadeStats(
            n_total=n,
            n_routed_stage2=int(len(idx)),
            stage1_scores=s1,
            stage2_scores=s2_full,
            routed_to_stage2=routed,
        )
        return final

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def estimated_latency_ms(
        self,
        stage1_latency_ms: float,
        stage2_latency_ms: float,
    ) -> float:
        """Weighted per-sample latency using the last ``predict_proba`` routing."""
        if self.stats is None or self.stats.n_total == 0:
            raise RuntimeError("Call predict_proba first.")
        rate = self.stats.stage2_rate
        return stage1_latency_ms + rate * stage2_latency_ms

    def estimated_throughput(
        self,
        stage1_throughput: float,
        stage2_throughput: float,
    ) -> float:
        """Weighted samples/sec assuming stages run sequentially in a batch pipeline."""
        if self.stats is None or self.stats.n_total == 0:
            raise RuntimeError("Call predict_proba first.")
        rate = self.stats.stage2_rate
        t_per_sample_s = 1.0 / stage1_throughput + rate / stage2_throughput
        return 1.0 / t_per_sample_s if t_per_sample_s > 0 else float("inf")
