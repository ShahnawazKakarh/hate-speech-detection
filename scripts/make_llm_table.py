"""Aggregate Gemini LLM results into a paper-ready table.

Reads ``artifacts/gemini_<dataset>_<regime>/metrics.json`` and emits:

  - results/llm_results.md
  - paper/tables/llm_results.tex

Usage:
    python scripts/make_llm_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hsd.utils import ARTIFACTS_DIR

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
PAPER_TABLES = REPO / "paper" / "tables"

REGIMES = ["zeroshot", "fewshot"]
DATASETS = ["davidson", "hatexplain"]
REGIME_LABEL = {"zeroshot": "zero-shot", "fewshot": "4-shot"}


def _load(regime: str, dataset: str) -> dict | None:
    p = ARTIFACTS_DIR / f"gemini_{dataset}_{regime}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _baseline(dataset: str) -> dict | None:
    """DistilBERT v0.1 baseline on the same test split, for reference."""
    p = ARTIFACTS_DIR / f"distilbert_{dataset}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _row(label: str, r: dict | None) -> str:
    if r is None:
        return f"| {label} | — | — | — | — | — | — |"
    auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "—"
    return (
        f"| {label} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
        f"{r['f1_binary']:.4f} | {r['f1_macro']:.4f} | {auc} |"
    )


def _build_md() -> str:
    lines = [
        "# Gemini LLM-as-classifier results (v0.2.1)",
        "",
        "Zero-shot and 4-shot evaluation of `gemini-2.5-flash` on the same Davidson "
        "and HateXplain test splits used in v0.1. DistilBERT v0.1 numbers included "
        "for direct comparison. Decision threshold 0.5; `p_hate` returned by the "
        "model serves as the score for ROC-AUC.",
        "",
    ]
    for d in DATASETS:
        lines.append(f"## {d}")
        lines.append("")
        lines.append("| Method | Acc | P | R | F1 (hate) | F1 (macro) | AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(_row("DistilBERT (v0.1)", _baseline(d)))
        for reg in REGIMES:
            lines.append(_row(f"Gemini 2.5 Flash, {REGIME_LABEL[reg]}", _load(reg, d)))
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_tex() -> str:
    rows = []
    for d in DATASETS:
        b = _baseline(d)
        if b is not None:
            auc = f"{b['roc_auc']:.4f}" if b.get("roc_auc") is not None else "--"
            rows.append(
                f"DistilBERT (v0.1) & {d} & {b['accuracy']:.4f} & "
                f"{b['precision']:.4f} & {b['recall']:.4f} & "
                f"{b['f1_binary']:.4f} & {b['f1_macro']:.4f} & {auc} \\\\"
            )
        for reg in REGIMES:
            r = _load(reg, d)
            if r is None:
                rows.append(
                    f"Gemini 2.5 Flash ({REGIME_LABEL[reg]}) & {d} & "
                    "-- & -- & -- & -- & -- & -- \\\\"
                )
                continue
            auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "--"
            rows.append(
                f"Gemini 2.5 Flash ({REGIME_LABEL[reg]}) & {d} & "
                f"{r['accuracy']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & "
                f"{r['f1_binary']:.4f} & {r['f1_macro']:.4f} & {auc} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "Method & Dataset & Acc & P & R & F1\\textsubscript{hate} & F1\\textsubscript{macro} & AUC \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    md = _build_md()
    (RESULTS_DIR / "llm_results.md").write_text(md)
    (PAPER_TABLES / "llm_results.tex").write_text(_build_tex())
    print(md)


if __name__ == "__main__":
    main()
