"""Cross-dataset generalization: train on dataset A, evaluate on B's test split.

Loads already-trained models from ``artifacts/<model>_<train_dataset>/`` and
applies them to the test split of the other dataset.

Usage:
    python scripts/cross_dataset_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hsd.data.loaders import load_dataset
from hsd.evaluate import evaluate
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger

REPO = Path(__file__).resolve().parents[1]

log = get_logger(__name__)

PAIRS = [
    ("davidson", "hatexplain"),
    ("hatexplain", "davidson"),
]


def _load_model(model_type: str, train_ds: str):
    path = ARTIFACTS_DIR / f"{model_type}_{train_ds}"
    if model_type == "tfidf":
        from hsd.models.tfidf import load

        return load(path / "pipeline.joblib")
    if model_type == "doc2vec":
        from hsd.models.doc2vec import Doc2VecClassifier

        return Doc2VecClassifier.load(path)
    if model_type == "distilbert":
        from hsd.models.distilbert import DistilBertClassifier

        return DistilBertClassifier.load(path / "final")
    raise ValueError(model_type)


def _score(model, model_type: str, texts: list[str]):
    if model_type == "tfidf":
        return model.predict_proba(texts)[:, 1]
    return model.predict_proba(texts)


def main() -> None:
    results = {}
    for model_type in ("tfidf", "doc2vec", "distilbert"):
        for train_ds, eval_ds in PAIRS:
            tag = f"{model_type}__train_{train_ds}__eval_{eval_ds}"
            try:
                model = _load_model(model_type, train_ds)
            except FileNotFoundError:
                log.warning("skipping %s: model not trained yet", tag)
                continue
            splits = load_dataset(eval_ds)
            y_score = _score(model, model_type, splits.texts("test"))
            y_pred = (y_score >= 0.5).astype(int)
            res = evaluate(splits.labels("test"), y_pred.tolist(), y_score)
            log.info("%s -> f1_macro=%.4f", tag, res.f1_macro)
            results[tag] = res.to_dict()

    out = ensure_dir(ARTIFACTS_DIR) / "cross_dataset.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
