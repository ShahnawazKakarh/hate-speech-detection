"""Paragraph2Vec (Doc2Vec) + Logistic Regression.

Faithful re-implementation of the Djuric et al. (2015) "comment2vec" approach:
jointly learn distributed representations of comments and words with the
PV-DBOW / PV-DM variants of Paragraph2Vec, then train a binary classifier on
the learned comment vectors.

For unseen texts at inference time, gensim's ``infer_vector`` performs the
"folding-in" step described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression


@dataclass
class Doc2VecConfig:
    vector_size: int = 200
    window: int = 5
    min_count: int = 2
    epochs: int = 20
    dm: int = 0  # 0 = PV-DBOW (recommended in the paper), 1 = PV-DM
    dbow_words: int = 1  # also train word vectors in DBOW
    negative: int = 5
    sample: float = 1e-4
    workers: int = 4
    seed: int = 42
    # classifier
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    infer_steps: int = 50


def _tokenize(text: str) -> list[str]:
    return text.split()


def _tagged_corpus(texts: list[str]) -> list[TaggedDocument]:
    return [TaggedDocument(words=_tokenize(t), tags=[i]) for i, t in enumerate(texts)]


class Doc2VecClassifier:
    """End-to-end Doc2Vec + LR classifier."""

    def __init__(self, cfg: Doc2VecConfig):
        self.cfg = cfg
        self.d2v: Doc2Vec | None = None
        self.clf: LogisticRegression | None = None

    # ------------------------------------------------------------------ #
    def fit(self, texts: list[str], labels: list[int]) -> Doc2VecClassifier:
        c = self.cfg
        corpus = _tagged_corpus(texts)

        self.d2v = Doc2Vec(
            vector_size=c.vector_size,
            window=c.window,
            min_count=c.min_count,
            dm=c.dm,
            dbow_words=c.dbow_words,
            negative=c.negative,
            sample=c.sample,
            workers=c.workers,
            seed=c.seed,
            epochs=c.epochs,
        )
        self.d2v.build_vocab(corpus)
        self.d2v.train(corpus, total_examples=self.d2v.corpus_count, epochs=c.epochs)

        # Use the trained doc vectors directly for the training set
        X = np.vstack([self.d2v.dv[i] for i in range(len(texts))])
        self.clf = LogisticRegression(
            C=c.C,
            max_iter=c.max_iter,
            class_weight=c.class_weight,
            solver="lbfgs",
        ).fit(X, labels)
        return self

    # ------------------------------------------------------------------ #
    def _embed(self, texts: list[str]) -> np.ndarray:
        assert self.d2v is not None, "Call fit() first."
        steps = self.cfg.infer_steps
        return np.vstack([self.d2v.infer_vector(_tokenize(t), epochs=steps) for t in texts])

    def predict(self, texts: list[str]) -> np.ndarray:
        assert self.clf is not None
        return self.clf.predict(self._embed(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.clf is not None
        return self.clf.predict_proba(self._embed(texts))[:, 1]

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.d2v.save(str(path / "doc2vec.bin"))
        joblib.dump(self.clf, path / "clf.joblib")

    @classmethod
    def load(cls, path: str | Path, cfg: Doc2VecConfig | None = None) -> Doc2VecClassifier:
        path = Path(path)
        obj = cls(cfg or Doc2VecConfig())
        obj.d2v = Doc2Vec.load(str(path / "doc2vec.bin"))
        obj.clf = joblib.load(path / "clf.joblib")
        return obj
