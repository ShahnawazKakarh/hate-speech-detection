"""Adversarial obfuscation evaluation.

Applies a fixed, deterministic set of evasion patterns to each dataset's
test split, then re-scores every trained model on the perturbed text.
Compares F1-hate clean vs obfuscated.

Obfuscation patterns (applied per-character with fixed probabilities under
seed 42, so the perturbed test set is reproducible):
  * Char substitution: a→@, e→3, i→!, o→0, s→$, l→1, t→7 (30% per char)
  * Char repetition:   k→kk / kk→kkk on consonants (5% per char)
  * Token-internal spacing: "kill" → "k i l l" (5% of tokens longer than 3 chars)
  * Random punctuation insertion: "fuck" → "f*ck" (10% of tokens with a vowel)

Each rule fires independently. Applied uniformly to every token, not only
hate-related ones, simulating the cat-and-mouse case where a motivated user
obfuscates everything to defeat a keyword filter.

Outputs:
  - results/adversarial_eval.md
  - results/adversarial_eval.json
  - paper/tables/adversarial.tex

Usage:
    python scripts/adversarial_eval.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from hsd.data.loaders import load_dataset
from hsd.evaluate import evaluate
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger

REPO = Path(__file__).resolve().parents[1]

log = get_logger(__name__)

RESULTS_DIR = ensure_dir(REPO / "results")
PAPER_TABLES = ensure_dir(REPO / "paper" / "tables")

MODELS = ["tfidf", "doc2vec", "distilbert"]
DATASETS = ["davidson", "hatexplain"]

MODEL_LABEL = {
    "tfidf": "TF-IDF + LR",
    "doc2vec": "Doc2Vec + LR",
    "distilbert": "DistilBERT",
}

# --------------------------------------------------------------------------- #
# Obfuscation
# --------------------------------------------------------------------------- #
LEET_MAP = {
    "a": "@",
    "e": "3",
    "i": "!",
    "o": "0",
    "s": "$",
    "l": "1",
    "t": "7",
}
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
PUNCT_INSERTS = "*.!"  # used for "f*ck"-style substitution


def _obfuscate_token(token: str, rng: random.Random) -> str:
    """Apply leet substitution + occasional repetition + occasional punct insert."""
    if not token or not token.isalpha():
        return token

    chars = list(token)

    # 1. leet sub (30% per char where applicable)
    for i, c in enumerate(chars):
        if c in LEET_MAP and rng.random() < 0.30:
            chars[i] = LEET_MAP[c]

    # 2. consonant repetition (5% per char)
    out = []
    for c in chars:
        out.append(c)
        if c.lower() in CONSONANTS and rng.random() < 0.05:
            out.append(c)
    chars = out

    # 3. punctuation insertion replacing one vowel (10% of tokens with vowels)
    has_vowel = any(c.lower() in VOWELS for c in chars)
    if has_vowel and rng.random() < 0.10:
        vowel_idxs = [i for i, c in enumerate(chars) if c.lower() in VOWELS]
        if vowel_idxs:
            idx = rng.choice(vowel_idxs)
            chars[idx] = rng.choice(PUNCT_INSERTS)

    s = "".join(chars)

    # 4. token-internal spacing (5% of tokens longer than 3 chars)
    if len(s) > 3 and rng.random() < 0.05:
        s = " ".join(s)

    return s


_WS_SPLIT = re.compile(r"(\s+)")


def obfuscate_text(text: str, seed: int) -> str:
    """Deterministic per-row obfuscation. Seed is row-specific so results are
    reproducible but each row gets a unique perturbation."""
    if not text:
        return text
    rng = random.Random(seed)
    parts = _WS_SPLIT.split(text)  # keeps whitespace runs as separate parts
    obf = []
    for p in parts:
        if p.isspace():
            obf.append(p)
        else:
            obf.append(_obfuscate_token(p, rng))
    return "".join(obf)


# --------------------------------------------------------------------------- #
# Models
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
def main() -> None:
    results = {}
    examples_per_dataset = {}

    for dataset in DATASETS:
        splits = load_dataset(dataset)
        clean_texts = splits.texts("test")
        labels = splits.labels("test")

        # Build obfuscated test set with deterministic per-row seed
        obf_texts = [obfuscate_text(t, seed=42 + i) for i, t in enumerate(clean_texts)]
        examples_per_dataset[dataset] = [
            {"clean": c, "obfuscated": o}
            for c, o in list(zip(clean_texts, obf_texts, strict=False))[:5]
        ]

        for model_type in MODELS:
            tag = f"{model_type}_{dataset}"
            try:
                model = _load_model(model_type, dataset)
            except FileNotFoundError:
                log.warning("skipping %s: artifact missing", tag)
                continue

            scores_clean = _score(model, model_type, clean_texts)
            scores_obf = _score(model, model_type, obf_texts)
            pred_clean = (scores_clean >= 0.5).astype(int)
            pred_obf = (scores_obf >= 0.5).astype(int)

            clean = evaluate(labels, pred_clean.tolist(), scores_clean)
            obf = evaluate(labels, pred_obf.tolist(), scores_obf)

            results[tag] = {
                "clean": clean.to_dict(),
                "obfuscated": obf.to_dict(),
                "delta_f1_hate": obf.f1_binary - clean.f1_binary,
                "delta_f1_macro": obf.f1_macro - clean.f1_macro,
                "delta_auc": (obf.roc_auc or 0) - (clean.roc_auc or 0),
            }
            log.info(
                "%s: F1-hate %.4f -> %.4f (Δ=%+.4f), AUC %.4f -> %.4f (Δ=%+.4f)",
                tag,
                clean.f1_binary,
                obf.f1_binary,
                obf.f1_binary - clean.f1_binary,
                clean.roc_auc or 0,
                obf.roc_auc or 0,
                (obf.roc_auc or 0) - (clean.roc_auc or 0),
            )

    (RESULTS_DIR / "adversarial_eval.json").write_text(
        json.dumps({"metrics": results, "examples": examples_per_dataset}, indent=2)
    )
    md = _build_md(results, examples_per_dataset)
    print(md)
    (RESULTS_DIR / "adversarial_eval.md").write_text(md)
    (PAPER_TABLES / "adversarial.tex").write_text(_build_tex(results))


# --------------------------------------------------------------------------- #
def _build_md(results: dict, examples: dict) -> str:
    lines = [
        "# Adversarial obfuscation evaluation",
        "",
        "Trained models scored on the same test split twice: clean text, and the same "
        "text after a deterministic seed-42 obfuscation pass (leet substitution, "
        "consonant repetition, occasional punctuation insertion, occasional "
        "token-internal spacing). All thresholds at the default 0.5. The Δ column "
        "is `obfuscated − clean` on F1 of the hate class — a negative number means "
        "obfuscation defeated the model.",
        "",
        "| Model | Dataset | F1 (hate) clean | F1 (hate) obfuscated | Δ F1 (hate) | AUC clean | AUC obf. | Δ AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — | — |")
                continue
            c, o = r["clean"], r["obfuscated"]
            df1 = r["delta_f1_hate"]
            dauc = r["delta_auc"]
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | "
                f"{c['f1_binary']:.4f} | {o['f1_binary']:.4f} | "
                f"{'+' if df1 >= 0 else ''}{df1:.4f} | "
                f"{c['roc_auc']:.4f} | {o['roc_auc']:.4f} | "
                f"{'+' if dauc >= 0 else ''}{dauc:.4f} |"
            )

    lines.append("")
    lines.append("## Sample obfuscations")
    lines.append("")
    for d, exs in examples.items():
        lines.append(f"### {d}")
        lines.append("")
        for i, e in enumerate(exs, 1):
            lines.append(f"{i}. **clean:** `{e['clean'][:120]}`")
            lines.append(f"   **obf.:** `{e['obfuscated'][:120]}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_tex(results: dict) -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- & -- & -- & -- \\\\")
                continue
            c, o = r["clean"], r["obfuscated"]
            df1 = r["delta_f1_hate"]
            dauc = r["delta_auc"]
            rows.append(
                f"{MODEL_LABEL[m]} & {d} & "
                f"{c['f1_binary']:.4f} & {o['f1_binary']:.4f} & "
                f"{'+' if df1 >= 0 else ''}{df1:.4f} & "
                f"{c['roc_auc']:.4f} & {o['roc_auc']:.4f} & "
                f"{'+' if dauc >= 0 else ''}{dauc:.4f} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "Model & Dataset & F1\\textsubscript{hate} clean & F1\\textsubscript{hate} obf. "
        "& $\\Delta$ F1\\textsubscript{hate} & AUC clean & AUC obf. & $\\Delta$ AUC \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


if __name__ == "__main__":
    main()
