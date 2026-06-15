"""TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class TfidfConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    max_features: int = 50_000
    min_df: int = 2
    sublinear_tf: bool = True
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"


def build_tfidf_pipeline(cfg: TfidfConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(cfg.ngram_min, cfg.ngram_max),
                    max_features=cfg.max_features,
                    min_df=cfg.min_df,
                    sublinear_tf=cfg.sublinear_tf,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg.C,
                    max_iter=cfg.max_iter,
                    class_weight=cfg.class_weight,
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )


def save(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)


def load(path: str) -> Pipeline:
    return joblib.load(path)


def predict_proba(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    return pipe.predict_proba(texts)[:, 1]
