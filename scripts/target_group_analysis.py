# scripts/target_group_analysis.py
"""Target-group analysis on HateXplain.

HateXplain annotators tag each hateful/offensive post with the *target group*
(African, Women, Islam, Homosexual, …). This script:

  1. Re-parses the raw HateXplain JSON to extract per-post majority target.
  2. Maps fine-grained targets to higher-level categories (Race, Religion,
     Gender, Sexual Orientation, Other, None).
  3. Loads the trained DistilBERT-HateXplain model and predicts on the test
     split.
  4. Computes per-group support, prediction rate, recall on the hate class,
     and F1 on the hate class. Reveals where the model systematically fails.

Outputs:
  - results/target_group_analysis.md
  - results/target_group_analysis.json
  - paper/tables/target_groups.tex

Usage:
    python scripts/target_group_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hsd.utils import ARTIFACTS_DIR, RAW_DIR, ensure_dir, get_logger  # noqa: E402

log = get_logger(__name__)

RESULTS_DIR = ensure_dir(REPO / "results")
PAPER_TABLES = ensure_dir(REPO / "paper" / "tables")

TARGET_CATEGORY = {
    "African": "Race",
    "Arab": "Race",
    "Asian": "Race",
    "Caucasian": "Race",
    "Hispanic": "Race",
    "Indian": "Race",
    "Indigenous": "Race",
    "Buddhism": "Religion",
    "Christian": "Religion",
    "Hindu": "Religion",
    "Islam": "Religion",
    "Jewish": "Religion",
    "Men": "Gender",
    "Women": "Gender",
    "Homosexual": "Sexual Orientation",
    "Heterosexual": "Sexual Orientation",
    "Bisexual": "Sexual Orientation",
    "Disability": "Other",
    "Economic": "Other",
    "Refugee": "Other",
    "Other": "Other",
    "Minority": "Other",
    "None": "None",
}


def _majority_target(annotators):
    votes = []
    for a in annotators:
        tgt = a.get("target", [])
        if not tgt:
            votes.append("None")
            continue
        votes.extend(tgt)
    if not votes:
        return "None"
    counter = Counter(votes)
    non_none = [(t, c) for t, c in counter.items() if t != "None"]
    if non_none:
        return max(non_none, key=lambda x: x[1])[0]
    return "None"


def _test_posts_with_targets():
    raw = json.loads((RAW_DIR / "hatexplain" / "dataset.json").read_text())
    splits = json.loads((RAW_DIR / "hatexplain" / "post_id_divisions.json").read_text())
    test_ids = splits.get("test", [])
    texts, labels, cats = [], [], []
    for pid in test_ids:
        if pid not in raw:
            continue
        post = raw[pid]
        votes = [a["label"] for a in post["annotators"]]
        majority = max(set(votes), key=votes.count)
        if majority not in {"hatespeech", "offensive", "normal"}:
            continue
        texts.append(" ".join(post["post_tokens"]))
        labels.append(int(majority == "hatespeech"))
        fine = _majority_target(post["annotators"])
        cats.append(TARGET_CATEGORY.get(fine, "Other"))
    return texts, labels, cats


def _row(group, y_true, y_pred, y_score):
    n = int(len(y_true))
    n_hate = int(y_true.sum())
    hate_pct = float(n_hate / n) if n else 0.0
    n_flagged = int(y_pred.sum())
    flagged_pct = float(n_flagged / n) if n else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if n_hate else float("nan")
    f1h = float(f1_score(y_true, y_pred, zero_division=0)) if n_hate else float("nan")
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "group": group,
        "n": n,
        "n_hate": n_hate,
        "hate_pct": hate_pct,
        "n_flagged": n_flagged,
        "flagged_pct": flagged_pct,
        "recall_hate": rec,
        "f1_hate": f1h,
        "f1_macro": f1m,
    }


def _fmt(x):
    if isinstance(x, float):
        if np.isnan(x):
            return "—"
        return f"{x:.3f}"
    return str(x)


def _build_md(rows):
    lines = [
        "# Target-group analysis on HateXplain (DistilBERT)",
        "",
        "Per-group breakdown of the DistilBERT-HateXplain test-set predictions, "
        "default decision threshold (0.50). `hate %` is the share of posts in the "
        "group with majority-vote `hatespeech`; `flagged %` is the share predicted "
        "as hate. Large gaps between `recall (hate)` across groups indicate the "
        "model is systematically missing hate aimed at some targets.",
        "",
        "| Group | n | n (hate) | hate % | flagged % | recall (hate) | F1 (hate) | F1 (macro) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['n']} | {r['n_hate']} | "
            f"{r['hate_pct']*100:.1f}% | {r['flagged_pct']*100:.1f}% | "
            f"{_fmt(r['recall_hate'])} | {_fmt(r['f1_hate'])} | {_fmt(r['f1_macro'])} |"
        )
    return "\n".join(lines) + "\n"


def _build_tex(rows):
    body_rows = []
    for r in rows:
        body_rows.append(
            f"{r['group']} & {r['n']} & {r['n_hate']} & "
            f"{r['hate_pct']*100:.1f}\\% & {r['flagged_pct']*100:.1f}\\% & "
            f"{_fmt(r['recall_hate'])} & {_fmt(r['f1_hate'])} & {_fmt(r['f1_macro'])} \\\\"
        )
    body = "\n".join(body_rows)
    return (
        "\\begin{tabular}{lrrrrccc}\n"
        "\\toprule\n"
        "Group & $n$ & $n_\\text{hate}$ & hate \\% & flagged \\% & "
        "Recall (hate) & F1 (hate) & F1 (macro) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main():
    texts, labels, cats = _test_posts_with_targets()
    labels_arr = np.array(labels)
    cats_arr = np.array(cats)
    log.info("loaded %d HateXplain test posts; targets: %s", len(texts), dict(Counter(cats_arr)))

    from hsd.data.preprocess import clean_series

    cleaned = clean_series(texts)

    from hsd.models.distilbert import DistilBertClassifier

    model_path = ARTIFACTS_DIR / "distilbert_hatexplain" / "final"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run: python -m hsd.train --config configs/distilbert_hatexplain.yaml"
        )
    log.info("loading model from %s", model_path)
    model = DistilBertClassifier.load(model_path)

    scores = model.predict_proba(cleaned)
    preds = (scores >= 0.5).astype(int)

    rows = [_row("Overall", labels_arr, preds, scores)]
    for g in ["Race", "Religion", "Gender", "Sexual Orientation", "Other", "None"]:
        mask = cats_arr == g
        if mask.sum() == 0:
            continue
        rows.append(_row(g, labels_arr[mask], preds[mask], scores[mask]))

    md = _build_md(rows)
    print(md)
    (RESULTS_DIR / "target_group_analysis.md").write_text(md)
    (RESULTS_DIR / "target_group_analysis.json").write_text(json.dumps(rows, indent=2))
    (PAPER_TABLES / "target_groups.tex").write_text(_build_tex(rows))
    log.info("wrote %s", RESULTS_DIR / "target_group_analysis.md")


if __name__ == "__main__":
    main()
