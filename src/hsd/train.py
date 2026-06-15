"""Unified config-driven training entrypoint.

Usage:
    python -m hsd.train --config configs/tfidf_davidson.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

from hsd.data.loaders import load_dataset
from hsd.evaluate import evaluate, text_report
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger, set_seed

log = get_logger(__name__)


def _train_tfidf(cfg: dict, splits):
    from hsd.models.tfidf import TfidfConfig, build_tfidf_pipeline, save

    mcfg = TfidfConfig(**cfg.get("model", {}))
    pipe = build_tfidf_pipeline(mcfg)
    pipe.fit(splits.texts("train"), splits.labels("train"))

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    save(pipe, out / "pipeline.joblib")

    y_score = pipe.predict_proba(splits.texts("test"))[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def _train_doc2vec(cfg: dict, splits):
    from hsd.models.doc2vec import Doc2VecClassifier, Doc2VecConfig

    mcfg = Doc2VecConfig(**cfg.get("model", {}))
    model = Doc2VecClassifier(mcfg).fit(splits.texts("train"), splits.labels("train"))

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    model.save(out)

    y_score = model.predict_proba(splits.texts("test"))
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def _train_distilbert(cfg: dict, splits):
    from hsd.models.distilbert import DistilBertClassifier, DistilBertConfig

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    mcfg = DistilBertConfig(output_dir=str(out / "hf"), **cfg.get("model", {}))
    model = DistilBertClassifier(mcfg).fit(
        splits.texts("train"),
        splits.labels("train"),
        splits.texts("val"),
        splits.labels("val"),
    )
    model.save(out / "final")

    y_score = model.predict_proba(splits.texts("test"))
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


TRAINERS = {
    "tfidf": _train_tfidf,
    "doc2vec": _train_doc2vec,
    "distilbert": _train_distilbert,
}


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg.get("seed", 42))
    log.info("config: %s", cfg)

    # Optional W&B
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
    y_pred, y_score = trainer(cfg, splits)
    y_true = splits.labels("test")

    result = evaluate(y_true, y_pred.tolist(), y_score)
    log.info("test metrics:\n%s", json.dumps(result.to_dict(), indent=2))
    log.info("\n%s", text_report(y_true, y_pred.tolist()))

    out = ensure_dir(ARTIFACTS_DIR / cfg["run_name"])
    (out / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2))

    if use_wandb:
        import wandb

        wandb.log(result.to_dict())
        wandb.finish()


if __name__ == "__main__":
    main()
