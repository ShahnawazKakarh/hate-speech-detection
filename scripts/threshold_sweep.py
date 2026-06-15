"""Threshold-tuning ablation.

For each (model, dataset) pair:
  1. Load the trained model from artifacts/.
  2. Score the *validation* set, sweep thresholds in [0.05, 0.95], pick the one
     that maximizes F1 on the hate class.
  3. Apply the chosen threshold to the *test* set and report the full metric
     suite alongside the default-0.5 result.

Outputs:
  - results/threshold_tuning.json   (per-run metrics)
  - results/threshold_tuning.md     (markdown comparison table)
  - paper/tables/threshold_tuning.tex
  - paper/figures/threshold_curves.png  (F1-hate vs threshold per model/dataset)

Usage:
    python scripts/threshold_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hsd.data.loaders import load_dataset  # noqa: E402
from hsd.evaluate import evaluate  # noqa: E402
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger  # noqa: E402

log = get_logger(__name__)

RESULTS_DIR = ensure_dir(REPO / "results")
PAPER_TABLES = ensure_dir(REPO / "paper" / "tables")
PAPER_FIGURES = ensure_dir(REPO / "paper" / "figures")

MODELS = ["tfidf", "doc2vec", "distilbert"]
DATASETS = ["davidson", "hatexplain"]

MODEL_LABEL = {
    "tfidf": "TF-IDF + LR",
    "doc2vec": "Doc2Vec + LR",
    "distilbert": "DistilBERT",
}


# --------------------------------------------------------------------------- #
def _load_model(model_type: str, dataset: str):
    path = ARTIFACTS_DIR / f"{model_type}_{dataset}"
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


def _score(model, model_type: str, texts: list[str]) -> np.ndarray:
    if model_type == "tfidf":
        return model.predict_proba(texts)[:, 1]
    return model.predict_proba(texts)


# --------------------------------------------------------------------------- #
def _sweep(scores: np.ndarray, labels: list[int]) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (best_threshold, best_f1_hate, thresholds, f1_curve)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_curve = np.array(
        [f1_score(labels, (scores >= t).astype(int), zero_division=0) for t in thresholds]
    )
    best_idx = int(np.argmax(f1_curve))
    return float(thresholds[best_idx]), float(f1_curve[best_idx]), thresholds, f1_curve


# --------------------------------------------------------------------------- #
def main() -> None:
    results = {}
    curves = {}

    for model_type in MODELS:
        for dataset in DATASETS:
            tag = f"{model_type}_{dataset}"
            try:
                model = _load_model(model_type, dataset)
            except FileNotFoundError:
                log.warning("skipping %s: model artifact missing", tag)
                continue

            splits = load_dataset(dataset)

            # 1. sweep on validation
            val_scores = _score(model, model_type, splits.texts("val"))
            best_t, best_val_f1, ts, f1_curve = _sweep(val_scores, splits.labels("val"))
            log.info("%s: best val threshold = %.2f (val F1-hate = %.4f)", tag, best_t, best_val_f1)

            # 2. apply to test
            test_scores = _score(model, model_type, splits.texts("test"))
            y_true = splits.labels("test")
            y_pred_default = (test_scores >= 0.5).astype(int)
            y_pred_tuned = (test_scores >= best_t).astype(int)

            default = evaluate(y_true, y_pred_default.tolist(), test_scores)
            tuned = evaluate(y_true, y_pred_tuned.tolist(), test_scores)

            results[tag] = {
                "best_threshold": best_t,
                "best_val_f1_hate": best_val_f1,
                "default_threshold_metrics": default.to_dict(),
                "tuned_threshold_metrics": tuned.to_dict(),
                "delta_f1_hate": tuned.f1_binary - default.f1_binary,
                "delta_f1_macro": tuned.f1_macro - default.f1_macro,
            }
            curves[tag] = {"thresholds": ts.tolist(), "f1_hate": f1_curve.tolist()}

    # write json
    (RESULTS_DIR / "threshold_tuning.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", RESULTS_DIR / "threshold_tuning.json")

    # markdown table
    md = _build_markdown(results)
    (RESULTS_DIR / "threshold_tuning.md").write_text(md)
    print(md)

    # latex table
    tex = _build_latex(results)
    (PAPER_TABLES / "threshold_tuning.tex").write_text(tex)

    # plot
    _plot_curves(curves, PAPER_FIGURES / "threshold_curves.png")
    log.info("wrote %s", PAPER_FIGURES / "threshold_curves.png")


# --------------------------------------------------------------------------- #
def _build_markdown(results: dict) -> str:
    lines = [
        "# Threshold-tuning ablation",
        "",
        "Decision threshold selected by maximizing F1 on the hate class on the *validation* "
        "set, then applied unchanged to the test set. Default = 0.5 baseline shown for "
        "reference.",
        "",
        "| Model | Dataset | t* | F1 (hate) @ 0.5 | F1 (hate) @ t* | Δ F1 (hate) | F1 (macro) @ t* |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — |")
                continue
            t = r["best_threshold"]
            f1d = r["default_threshold_metrics"]["f1_binary"]
            f1t = r["tuned_threshold_metrics"]["f1_binary"]
            f1m = r["tuned_threshold_metrics"]["f1_macro"]
            delta = r["delta_f1_hate"]
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | {t:.2f} | "
                f"{f1d:.4f} | **{f1t:.4f}** | {sign}{delta:.4f} | {f1m:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _build_latex(results: dict) -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- & -- & -- \\\\")
                continue
            t = r["best_threshold"]
            f1d = r["default_threshold_metrics"]["f1_binary"]
            f1t = r["tuned_threshold_metrics"]["f1_binary"]
            f1m = r["tuned_threshold_metrics"]["f1_macro"]
            delta = r["delta_f1_hate"]
            sign = "+" if delta >= 0 else ""
            rows.append(
                f"{MODEL_LABEL[m]} & {d} & {t:.2f} & {f1d:.4f} & "
                f"\\textbf{{{f1t:.4f}}} & {sign}{delta:.4f} & {f1m:.4f} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llccccc}\n"
        "\\toprule\n"
        "Model & Dataset & $t^*$ & F1 (hate) @ 0.5 & F1 (hate) @ $t^*$ "
        "& $\\Delta$ F1 (hate) & F1 (macro) @ $t^*$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _plot_curves(curves: dict, out: Path) -> None:
    """One subplot per dataset, one line per model, F1-hate vs threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=False):
        for m in MODELS:
            tag = f"{m}_{dataset}"
            if tag not in curves:
                continue
            c = curves[tag]
            ax.plot(c["thresholds"], c["f1_hate"], label=MODEL_LABEL[m], linewidth=2)
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.6, label="default (0.5)")
        ax.set_title(f"{dataset}")
        ax.set_xlabel("decision threshold")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F1 (hate)")
    axes[0].legend(loc="lower center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
