"""Aggregate metrics from all artifacts into a Markdown + LaTeX results table.

Reads ``artifacts/<run_name>/metrics.json`` and emits:
  - paper/tables/results.md
  - paper/tables/results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

from hsd.utils import ARTIFACTS_DIR

REPO = Path(__file__).resolve().parents[1]
PAPER_TABLES = REPO / "paper" / "tables"

MODELS = ["tfidf", "doc2vec", "distilbert"]
DATASETS = ["davidson", "hatexplain"]

MODEL_LABEL = {
    "tfidf": "TF-IDF + LR",
    "doc2vec": "Doc2Vec + LR",
    "distilbert": "DistilBERT",
}


def load_metrics(model: str, dataset: str) -> dict | None:
    p = ARTIFACTS_DIR / f"{model}_{dataset}" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_markdown() -> str:
    lines = []
    header = "| Model | Dataset | Acc | P | R | F1 (hate) | F1 (macro) | AUC |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for m in MODELS:
        for d in DATASETS:
            r = load_metrics(m, d)
            if r is None:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — | — |")
                continue
            auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "—"
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | "
                f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['f1_binary']:.4f} | {r['f1_macro']:.4f} | {auc} |"
            )
    return "\n".join(lines)


def build_latex() -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = load_metrics(m, d)
            if r is None:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- & -- & -- & -- \\\\")
                continue
            auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "--"
            rows.append(
                f"{MODEL_LABEL[m]} & {d} & "
                f"{r['accuracy']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & "
                f"{r['f1_binary']:.4f} & {r['f1_macro']:.4f} & {auc} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "Model & Dataset & Acc & P & R & F1 (hate) & F1 (macro) & AUC \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> None:
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    md = build_markdown()
    tex = build_latex()
    (PAPER_TABLES / "results.md").write_text(md + "\n")
    (PAPER_TABLES / "results.tex").write_text(tex)
    print(md)


if __name__ == "__main__":
    main()
