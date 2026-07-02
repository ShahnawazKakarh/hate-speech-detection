"""Cascade threshold sweep.

For each dataset, load the trained TF-IDF and DistilBERT models, sweep the
stage-1 threshold from ~0.05 to ~0.50, and for each threshold report:
  - system F1-hate, F1-macro, ROC-AUC, accuracy
  - fraction of samples routed to stage 2
  - estimated system latency and throughput (using §5.4 numbers from
    results/inference_cost.json)

Emits:
  - results/cascade_sweep.md
  - results/cascade_sweep.json
  - paper/tables/cascade.tex
  - paper/figures/cascade_frontier.png   (F1 vs. throughput)

Usage:
    python scripts/cascade_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from hsd.cascade import Cascade
from hsd.data.loaders import load_dataset
from hsd.evaluate import evaluate
from hsd.utils import ARTIFACTS_DIR, ensure_dir, get_logger

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = ensure_dir(REPO / "results")
PAPER_TABLES = ensure_dir(REPO / "paper" / "tables")
PAPER_FIGURES = ensure_dir(REPO / "paper" / "figures")

log = get_logger(__name__)

DATASETS = ["davidson", "hatexplain"]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _load_pair(dataset: str):
    """Load trained TF-IDF and DistilBERT for a given dataset."""
    from hsd.models.distilbert import DistilBertClassifier
    from hsd.models.tfidf import load as load_tfidf

    pipe = load_tfidf(ARTIFACTS_DIR / f"tfidf_{dataset}" / "pipeline.joblib")
    bert = DistilBertClassifier.load(ARTIFACTS_DIR / f"distilbert_{dataset}" / "final")
    return pipe, bert


def _load_costs() -> dict:
    """Read §5.4 latency + throughput per model per dataset."""
    p = RESULTS_DIR / "inference_cost.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/inference_cost.py first.")
    return json.loads(p.read_text())


def _baseline_metrics(dataset: str) -> dict | None:
    p = ARTIFACTS_DIR / f"distilbert_{dataset}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
def _sweep_one(dataset: str, costs: dict) -> dict:
    log.info("cascade sweep on %s ...", dataset)
    pipe, bert = _load_pair(dataset)
    splits = load_dataset(dataset)
    texts = splits.texts("test")
    y_true = splits.labels("test")

    cost1 = costs[f"tfidf_{dataset}"]
    cost2 = costs[f"distilbert_{dataset}"]
    s1_lat = cost1["latency_ms_median"]
    s2_lat = cost2["latency_ms_median"]
    s1_thr = cost1["throughput_samples_per_sec"]
    s2_thr = cost2["throughput_samples_per_sec"]

    rows = []
    for t1 in THRESHOLDS:
        casc = Cascade(
            stage1=pipe,
            stage1_type="tfidf",
            stage2=bert,
            stage2_type="distilbert",
            stage1_threshold=t1,
        )
        p_hate = casc.predict_proba(texts)
        y_pred = (p_hate >= 0.5).astype(int)
        m = evaluate(y_true, y_pred.tolist(), p_hate)
        rate = casc.stats.stage2_rate
        lat = casc.estimated_latency_ms(s1_lat, s2_lat)
        thr = casc.estimated_throughput(s1_thr, s2_thr)
        rows.append(
            {
                "stage1_threshold": t1,
                "stage2_rate": rate,
                "accuracy": m.accuracy,
                "f1_hate": m.f1_binary,
                "f1_macro": m.f1_macro,
                "roc_auc": m.roc_auc,
                "latency_ms_per_sample": lat,
                "throughput_samples_per_sec": thr,
            }
        )
        log.info(
            "  t1=%.2f  stage2_rate=%.2f  F1-hate=%.4f  F1-macro=%.4f  thr=%.1f s/s",
            t1,
            rate,
            m.f1_binary,
            m.f1_macro,
            thr,
        )

    # Also stash the DistilBERT-alone baseline for the plot
    baseline = _baseline_metrics(dataset) or {}
    baseline_thr = cost2["throughput_samples_per_sec"]

    return {
        "dataset": dataset,
        "sweep": rows,
        "distilbert_alone": {
            "f1_hate": baseline.get("f1_binary"),
            "f1_macro": baseline.get("f1_macro"),
            "roc_auc": baseline.get("roc_auc"),
            "throughput_samples_per_sec": baseline_thr,
        },
    }


# --------------------------------------------------------------------------- #
def _pick_operating_points(sweep: list[dict], baseline_f1: float | None) -> dict:
    """Highlight two operating points:
    (a) t1 that keeps F1-hate >= baseline (no-loss cascade), highest throughput
    (b) t1 that gives best F1-hate overall
    """
    if not sweep:
        return {}

    best_f1 = max(sweep, key=lambda r: r["f1_hate"])
    no_loss = None
    if baseline_f1 is not None:
        candidates = [r for r in sweep if r["f1_hate"] >= baseline_f1 - 1e-9]
        if candidates:
            no_loss = max(candidates, key=lambda r: r["throughput_samples_per_sec"])
    return {"best_f1_hate": best_f1, "no_loss": no_loss}


def _plot(all_results: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4.5), sharey=False)
    if len(all_results) == 1:
        axes = [axes]
    for ax, res in zip(axes, all_results, strict=False):
        sweep = res["sweep"]
        thrs = [r["throughput_samples_per_sec"] for r in sweep]
        f1s = [r["f1_hate"] for r in sweep]
        ax.plot(thrs, f1s, marker="o", linewidth=2, label="cascade sweep")

        # DistilBERT-alone point
        b = res["distilbert_alone"]
        if b["f1_hate"] is not None:
            ax.scatter(
                [b["throughput_samples_per_sec"]],
                [b["f1_hate"]],
                s=100,
                color="red",
                zorder=5,
                label="DistilBERT alone",
            )

        # Label each point with its threshold
        for r in sweep:
            ax.annotate(
                f"t={r['stage1_threshold']:.2f}",
                (r["throughput_samples_per_sec"], r["f1_hate"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

        ax.set_title(res["dataset"])
        ax.set_xlabel("estimated throughput (samples / sec)")
        ax.set_xscale("log")
        ax.set_ylabel("test F1 (hate)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
def _build_md(all_results: list[dict]) -> str:
    lines = [
        "# Two-stage cascade sweep (v0.2.2)",
        "",
        "Stage 1 = TF-IDF + LR (cheap prefilter). Stage 2 = fine-tuned DistilBERT "
        "(expensive verifier). Samples with stage-1 `p_hate >= t1` are re-scored "
        "by stage 2; the rest use the stage-1 score. Final decision threshold is "
        "the standard 0.5. Latency and throughput are extrapolated from "
        "`results/inference_cost.json`.",
        "",
    ]
    for res in all_results:
        d = res["dataset"]
        lines.append(f"## {d}")
        lines.append("")
        lines.append(
            "| Stage-1 t1 | % → stage 2 | F1 (hate) | F1 (macro) | AUC | "
            "Latency ms / sample | Throughput samples/s |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for r in res["sweep"]:
            auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "—"
            lines.append(
                f"| {r['stage1_threshold']:.2f} | {r['stage2_rate']*100:.1f}% | "
                f"{r['f1_hate']:.4f} | {r['f1_macro']:.4f} | {auc} | "
                f"{r['latency_ms_per_sample']:.2f} | {r['throughput_samples_per_sec']:.1f} |"
            )

        b = res["distilbert_alone"]
        picks = _pick_operating_points(res["sweep"], b["f1_hate"])
        lines.append("")
        lines.append("**Reference points**")
        lines.append("")
        if b["f1_hate"] is not None:
            lines.append(
                f"- DistilBERT alone: F1-hate = {b['f1_hate']:.4f}, "
                f"F1-macro = {b['f1_macro']:.4f}, AUC = {b['roc_auc']:.4f}, "
                f"throughput = {b['throughput_samples_per_sec']:.1f} samples/s."
            )
        if picks.get("no_loss"):
            n = picks["no_loss"]
            speedup = n["throughput_samples_per_sec"] / max(1e-9, b["throughput_samples_per_sec"])
            lines.append(
                f"- **No-loss cascade** (F1-hate ≥ DistilBERT alone): "
                f"t1 = {n['stage1_threshold']:.2f}, "
                f"stage-2 rate = {n['stage2_rate']*100:.1f}%, "
                f"F1-hate = {n['f1_hate']:.4f}, "
                f"throughput = {n['throughput_samples_per_sec']:.1f} samples/s "
                f"(**{speedup:.1f}× DistilBERT alone**)."
            )
        best = picks.get("best_f1_hate")
        if best is not None:
            lines.append(
                f"- Best-F1 operating point: t1 = {best['stage1_threshold']:.2f}, "
                f"F1-hate = {best['f1_hate']:.4f} at "
                f"{best['throughput_samples_per_sec']:.1f} samples/s."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_tex(all_results: list[dict]) -> str:
    rows = []
    for res in all_results:
        d = res["dataset"]
        b = res["distilbert_alone"]
        picks = _pick_operating_points(res["sweep"], b["f1_hate"])

        auc_b = f"{b['roc_auc']:.4f}" if b["roc_auc"] is not None else "--"
        rows.append(
            f"{d} & DistilBERT alone & 100.0\\% & {b['f1_hate']:.4f} & "
            f"{b['f1_macro']:.4f} & {auc_b} & {b['throughput_samples_per_sec']:.1f} \\\\"
        )

        if picks.get("no_loss"):
            n = picks["no_loss"]
            auc_n = f"{n['roc_auc']:.4f}" if n["roc_auc"] is not None else "--"
            speedup = n["throughput_samples_per_sec"] / max(1e-9, b["throughput_samples_per_sec"])
            rows.append(
                f"{d} & Cascade (t1={n['stage1_threshold']:.2f}) & "
                f"{n['stage2_rate']*100:.1f}\\% & {n['f1_hate']:.4f} & "
                f"{n['f1_macro']:.4f} & {auc_n} & "
                f"{n['throughput_samples_per_sec']:.1f} "
                f"($\\times${speedup:.1f}) \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "Dataset & Method & \\% $\\to$ stage 2 & F1\\textsubscript{hate} & "
        "F1\\textsubscript{macro} & AUC & Throughput (samples/s) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    costs = _load_costs()

    all_results = []
    for d in DATASETS:
        all_results.append(_sweep_one(d, costs))

    (RESULTS_DIR / "cascade_sweep.json").write_text(json.dumps(all_results, indent=2))
    md = _build_md(all_results)
    print(md)
    (RESULTS_DIR / "cascade_sweep.md").write_text(md)
    (PAPER_TABLES / "cascade.tex").write_text(_build_tex(all_results))
    _plot(all_results, PAPER_FIGURES / "cascade_frontier.png")
    log.info("wrote %s", PAPER_FIGURES / "cascade_frontier.png")


if __name__ == "__main__":
    main()
