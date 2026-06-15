"""Aggregate 3-way classification metrics from artifacts/*_3way/metrics.json.

Emits:
  - results/results_3way.md
  - paper/tables/results_3way.tex

Usage:
    python scripts/make_3way_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hsd.utils import ARTIFACTS_DIR

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
PAPER_TABLES = REPO / "paper" / "tables"

MODELS = ["tfidf", "doc2vec", "distilbert"]
DATASETS = ["davidson", "hatexplain"]
MODEL_LABEL = {
    "tfidf": "TF-IDF + LR",
    "doc2vec": "Doc2Vec + LR",
    "distilbert": "DistilBERT",
}
CLASS_NAMES = ["hate", "offensive", "neither/normal"]


def _load(model: str, dataset: str) -> dict | None:
    p = ARTIFACTS_DIR / f"{model}_{dataset}_3way" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _build_md() -> str:
    lines = [
        "# Three-way classification results",
        "",
        "Hate / offensive / neither-or-normal, default `argmax` decision rule. "
        "Per-class F1 plus macro-F1 and weighted-F1.",
        "",
        "| Model | Dataset | Acc | F1 hate | F1 offensive | F1 neither | F1 macro | F1 weighted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in DATASETS:
            r = _load(m, d)
            if r is None:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — | — |")
                continue
            f1pc = r["f1_per_class"]
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | "
                f"{r['accuracy']:.4f} | "
                f"{f1pc[0]:.4f} | {f1pc[1]:.4f} | {f1pc[2]:.4f} | "
                f"{r['f1_macro']:.4f} | {r['f1_weighted']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _build_tex() -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = _load(m, d)
            if r is None:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- & -- & -- & -- \\\\")
                continue
            f1pc = r["f1_per_class"]
            rows.append(
                f"{MODEL_LABEL[m]} & {d} & {r['accuracy']:.4f} & "
                f"{f1pc[0]:.4f} & {f1pc[1]:.4f} & {f1pc[2]:.4f} & "
                f"{r['f1_macro']:.4f} & {r['f1_weighted']:.4f} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llccccccc}\n"
        "\\toprule\n"
        "Model & Dataset & Acc & "
        "F1\\textsubscript{hate} & F1\\textsubscript{off.} & F1\\textsubscript{neither} & "
        "F1 macro & F1 weighted \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    md = _build_md()
    (RESULTS_DIR / "results_3way.md").write_text(md)
    (PAPER_TABLES / "results_3way.tex").write_text(_build_tex())
    print(md)


if __name__ == "__main__":
    main()
