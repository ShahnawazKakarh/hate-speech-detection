"""Unified config-driven training entrypoint.

Supports both binary (hate vs non-hate) and 3-way (hate/offensive/neither)
classification via the ``task`` field in the YAML config.

Usage:
    python -m hsd.train --config configs/tfidf_davidson.yaml          # binary
    python -m hsd.train --config configs/tfidf_davidson_3way.yaml     # 3-way
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import yaml

from hsd.data.loaders import load_dataset
from hsd.evaluate import (
    evaluate,
    evaluate_multiclass,
    text_report,
    text_report_multiclass,
)
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger, set_seed

log = get_logger(__name__)

# Davidson uses label3: 0=hate, 1=offensive, 2=neither
# HateXplain uses label3: 0=hate, 1=offensive, 2=normal
CLASS_NAMES_3WAY = ["hate", "offensive", "neither/normal"]
CLASS_NAMES_BINARY = ["non-hate", "hate"]


def _labels(splits, split: str, *, three_way: bool):
    return splits.labels(split, three_way=three_way)


# --------------------------------------------------------------------------- #
def _train_tfidf(cfg: dict, splits, three_way: bool):
    from hsd.models.tfidf import TfidfConfig, build_tfidf_pipeline, save

    mcfg = TfidfConfig(**cfg.get("model", {}))
    pipe = build_tfidf_pipeline(mcfg)
    pipe.fit(splits.texts("train"), _labels(splits, "train", three_way=three_way))

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    save(pipe, out / "pipeline.joblib")

    proba = pipe.predict_proba(splits.texts("test"))
    if three_way:
        y_pred = np.argmax(proba, axis=1)
        return y_pred, proba
    y_score = proba[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def _train_doc2vec(cfg: dict, splits, three_way: bool):
    from hsd.models.doc2vec import Doc2VecClassifier, Doc2VecConfig

    mcfg = Doc2VecConfig(**cfg.get("model", {}))
    model = Doc2VecClassifier(mcfg).fit(
        splits.texts("train"), _labels(splits, "train", three_way=three_way)
    )

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    model.save(out)

    if three_way:
        # Doc2VecClassifier wraps sklearn LR; reach in for multi-class proba
        X = model._embed(splits.texts("test"))  # noqa: SLF001
        proba = model.clf.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred, proba
    y_score = model.predict_proba(splits.texts("test"))
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def _train_distilbert(cfg: dict, splits, three_way: bool):
    from hsd.models.distilbert import DistilBertClassifier, DistilBertConfig

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    model_kwargs = dict(cfg.get("model", {}))
    model_kwargs["num_labels"] = 3 if three_way else 2
    mcfg = DistilBertConfig(output_dir=str(out / "hf"), **model_kwargs)
    model = DistilBertClassifier(mcfg).fit(
        splits.texts("train"),
        _labels(splits, "train", three_way=three_way),
        splits.texts("val"),
        _labels(splits, "val", three_way=three_way),
    )
    model.save(out / "final")

    if three_way:
        import torch

        # full softmax for 3 classes; reuse the model's tokenizer + forward
        texts = splits.texts("test")
        device = next(model.model.parameters()).device
        model.model.eval()
        probs_all = []
        with torch.no_grad():
            for i in range(0, len(texts), mcfg.eval_batch_size):
                batch = model._encode(texts[i : i + mcfg.eval_batch_size]).to(device)  # noqa: SLF001
                logits = model.model(**batch).logits
                probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
        proba = np.concatenate(probs_all, axis=0)
        y_pred = np.argmax(proba, axis=1)
        return y_pred, proba

    y_score = model.predict_proba(splits.texts("test"))
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


TRAINERS = {
    "tfidf": _train_tfidf,
    "doc2vec": _train_doc2vec,
    "distilbert": _train_distilbert,
}


# --------------------------------------------------------------------------- #
@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg.get("seed", 42))
    log.info("config: %s", cfg)

    task = cfg.get("task", "binary")
    three_way = task == "multiclass"

    use_wandb = cfg.get("wandb", False)
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.get("wandb_project", "hate-speech-detection"),
            entity=cfg.get("wandb_entity"),
            name=cfg["run_name"],
            config=cfg,
        )

    splits = load_dataset(cfg["dataset"])
    trainer = TRAINERS[cfg["model_type"]]
    y_pred, y_score = trainer(cfg, splits, three_way=three_way)
    y_true = _labels(splits, "test", three_way=three_way)

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])

    if three_way:
        result = evaluate_multiclass(y_true, y_pred.tolist(), class_names=CLASS_NAMES_3WAY)
        log.info("test metrics (3-way):\n%s", json.dumps(result.to_dict(), indent=2))
        log.info("\n%s", text_report_multiclass(y_true, y_pred.tolist(), CLASS_NAMES_3WAY))
    else:
        result = evaluate(y_true, y_pred.tolist(), y_score)
        log.info("test metrics:\n%s", json.dumps(result.to_dict(), indent=2))
        log.info("\n%s", text_report(y_true, y_pred.tolist()))

    (out / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2))

    if use_wandb:
        import wandb

        # wandb.log can't handle lists in nested keys; flatten the per-class arrays
        flat = {k: v for k, v in result.to_dict().items() if not isinstance(v, list)}
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
