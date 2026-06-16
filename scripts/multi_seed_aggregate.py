"""Aggregate multi-seed metrics into mean ± std tables.

Reads each ``artifacts/<model>_<dataset>{_sN}/metrics.json`` (where the
un-suffixed run is treated as seed 42), computes mean ± std across the
available seeds, and writes:

  - results/multi_seed.md
  - results/multi_seed.json
  - paper/tables/multi_seed.tex

Usage:
    python scripts/multi_seed_aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hsd.utils import ARTIFACTS_DIR

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
PAPER_TABLES = REPO / "paper" / "tables"

MODELS = ["tfidf", "doc2vec", "distilbert"]
DATASETS = ["davidson", "hatexplain"]
SEEDS = [42, 7, 1337]  # 42 lives at the un-suffixed path
MODEL_LABEL = {
    "tfidf": "TF-IDF + LR",
    "doc2vec": "Doc2Vec + LR",
    "distilbert": "DistilBERT",
}
METRICS = ["accuracy", "precision", "recall", "f1_binary", "f1_macro", "roc_auc"]


def _path(model: str, dataset: str, seed: int) -> Path:
    base = ARTIFACTS_DIR / f"{model}_{dataset}"
    if seed == 42:
        return base / "metrics.json"
    return ARTIFACTS_DIR / f"{model}_{dataset}_s{seed}" / "metrics.json"


def _load_seeds(model: str, dataset: str) -> dict[int, dict]:
    out = {}
    for s in SEEDS:
        p = _path(model, dataset, s)
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def _summarize(runs: dict[int, dict]) -> dict[str, dict[str, float] | int]:
    if not runs:
        return {}
    summary: dict[str, dict[str, float] | int] = {"n_seeds": len(runs)}
    for m in METRICS:
        vals = [r.get(m) for r in runs.values() if r.get(m) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        summary[m] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return summary


def _fmt(s: dict | None) -> str:
    if s is None:
        return "—"
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def _build_md(all_summaries: dict) -> str:
    lines = [
        "# Multi-seed results (binary classification)",
        "",
        "Mean ± std across seeds {42, 7, 1337}, identical preprocessing, identical splits, "
        "default 0.5 decision threshold. Seeds were chosen before any results were inspected.",
        "",
        "| Model | Dataset | n | Acc | F1 (hate) | F1 (macro) | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in DATASETS:
            s = all_summaries.get(f"{m}_{d}")
            if not s:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | {s['n_seeds']} | "
                f"{_fmt(s.get('accuracy'))} | "
                f"{_fmt(s.get('f1_binary'))} | "
                f"{_fmt(s.get('f1_macro'))} | "
                f"{_fmt(s.get('roc_auc'))} |"
            )
    return "\n".join(lines) + "\n"


def _build_tex(all_summaries: dict) -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            s = all_summaries.get(f"{m}_{d}")
            if not s:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- & -- \\\\")
                continue

            def _t(key: str, s=s) -> str:
                v = s.get(key)
                if v is None:
                    return "--"
                return f"${v['mean']:.3f} \\pm {v['std']:.3f}$"

            rows.append(
                f"{MODEL_LABEL[m]} & {d} & {_t('accuracy')} & "
                f"{_t('f1_binary')} & {_t('f1_macro')} & {_t('roc_auc')} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        "Model & Dataset & Acc & F1\\textsubscript{hate} & F1\\textsubscript{macro} & AUC \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for m in MODELS:
        for d in DATASETS:
            runs = _load_seeds(m, d)
            summaries[f"{m}_{d}"] = _summarize(runs)

    md = _build_md(summaries)
    print(md)
    (RESULTS_DIR / "multi_seed.md").write_text(md)
    (RESULTS_DIR / "multi_seed.json").write_text(json.dumps(summaries, indent=2))
    (PAPER_TABLES / "multi_seed.tex").write_text(_build_tex(summaries))


if __name__ == "__main__":
    main()
