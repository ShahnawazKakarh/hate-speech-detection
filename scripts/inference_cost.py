"""Inference-cost benchmark.

For each trained model:
  - Model size on disk (MB)
  - Peak resident memory while loaded (MB)
  - Wall-clock latency, single-sample (ms / sample)
  - Wall-clock throughput, batched-32 (samples / sec)

Reads the *test* split of each model's training dataset. Repeats each
measurement N times, reports median and IQR. Output:

  - results/inference_cost.md
  - results/inference_cost.json
  - paper/tables/inference_cost.tex

Usage:
    python scripts/inference_cost.py
    python scripts/inference_cost.py --n-warmup 5 --n-samples 200
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path

import click
import psutil
from hsd.data.loaders import load_dataset
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
def _artifact_dir(model_type: str, dataset: str) -> Path:
    return ARTIFACTS_DIR / f"{model_type}_{dataset}"


def _model_size_mb(model_type: str, dataset: str) -> float:
    """Total bytes of model artifacts on disk."""
    d = _artifact_dir(model_type, dataset)
    if not d.exists():
        return 0.0
    total = 0
    if model_type == "tfidf":
        f = d / "pipeline.joblib"
        if f.exists():
            total = f.stat().st_size
    elif model_type == "doc2vec":
        # gensim writes doc2vec.bin plus possibly auxiliary .npy files
        for f in d.glob("doc2vec.bin*"):
            total += f.stat().st_size
        clf = d / "clf.joblib"
        if clf.exists():
            total += clf.stat().st_size
    elif model_type == "distilbert":
        final = d / "final"
        if final.exists():
            for f in final.rglob("*"):
                if f.is_file() and f.suffix in {".safetensors", ".bin", ".json", ".txt"}:
                    total += f.stat().st_size
    return total / (1024 * 1024)


def _load_model(model_type: str, dataset: str):
    d = _artifact_dir(model_type, dataset)
    if model_type == "tfidf":
        from hsd.models.tfidf import load

        return load(d / "pipeline.joblib")
    if model_type == "doc2vec":
        from hsd.models.doc2vec import Doc2VecClassifier

        return Doc2VecClassifier.load(d)
    if model_type == "distilbert":
        from hsd.models.distilbert import DistilBertClassifier

        return DistilBertClassifier.load(d / "final")
    raise ValueError(model_type)


def _score_one(model, model_type: str, text: str) -> float:
    if model_type == "tfidf":
        return float(model.predict_proba([text])[0, 1])
    return float(model.predict_proba([text])[0])


def _score_batch(model, model_type: str, texts: list[str]) -> None:
    if model_type == "tfidf":
        model.predict_proba(texts)
    else:
        model.predict_proba(texts)


# --------------------------------------------------------------------------- #
def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _benchmark(
    model_type: str,
    dataset: str,
    samples: list[str],
    n_warmup: int,
    n_iters: int,
    batch_size: int,
) -> dict | None:
    if not _artifact_dir(model_type, dataset).exists():
        log.warning("skipping %s_%s: artifact missing", model_type, dataset)
        return None

    log.info("benchmarking %s_%s ...", model_type, dataset)
    gc.collect()
    rss_before = _rss_mb()

    model = _load_model(model_type, dataset)
    gc.collect()
    rss_loaded = _rss_mb()
    rss_load_delta = rss_loaded - rss_before

    # warmup
    for s in samples[:n_warmup]:
        _score_one(model, model_type, s)
    _score_batch(model, model_type, samples[:batch_size])

    # single-sample latency
    single_times_ms = []
    for s in samples[:n_iters]:
        t0 = time.perf_counter()
        _score_one(model, model_type, s)
        single_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # batched throughput
    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
    batches = [b for b in batches if len(b) == batch_size]  # drop ragged tail
    # cap at ~ n_iters / batch_size full batches
    batches = batches[: max(1, n_iters // batch_size)]

    batch_times_s = []
    for b in batches:
        t0 = time.perf_counter()
        _score_batch(model, model_type, b)
        batch_times_s.append(time.perf_counter() - t0)
    total_samples = sum(len(b) for b in batches)
    total_time = sum(batch_times_s)
    throughput = total_samples / total_time if total_time > 0 else float("nan")

    return {
        "size_mb": _model_size_mb(model_type, dataset),
        "rss_load_delta_mb": rss_load_delta,
        "latency_ms_median": statistics.median(single_times_ms),
        "latency_ms_p05": (
            statistics.quantiles(single_times_ms, n=20)[0]
            if len(single_times_ms) >= 20
            else min(single_times_ms)
        ),
        "latency_ms_p95": (
            statistics.quantiles(single_times_ms, n=20)[-1]
            if len(single_times_ms) >= 20
            else max(single_times_ms)
        ),
        "throughput_samples_per_sec": throughput,
        "batch_size": batch_size,
        "n_iters": n_iters,
        "n_warmup": n_warmup,
    }


# --------------------------------------------------------------------------- #
def _build_md(results: dict) -> str:
    lines = [
        "# Inference-cost benchmark",
        "",
        "Measured on the *test* split of each model's training dataset, single thread, "
        "no GPU (CPU/MPS only). Latency is per single sample (median + 5/95 percentile). "
        "Throughput is samples per second at batch size 32. RAM delta is resident set "
        "size increase when the model is loaded into memory.",
        "",
        "| Model | Dataset | Size on disk (MB) | RAM load Δ (MB) | Latency p50 (ms) | Latency p05/p95 (ms) | Throughput @ bs=32 (samples/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                lines.append(f"| {MODEL_LABEL[m]} | {d} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {MODEL_LABEL[m]} | {d} | "
                f"{r['size_mb']:.1f} | {r['rss_load_delta_mb']:.1f} | "
                f"{r['latency_ms_median']:.2f} | "
                f"{r['latency_ms_p05']:.2f} / {r['latency_ms_p95']:.2f} | "
                f"{r['throughput_samples_per_sec']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def _build_tex(results: dict) -> str:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            tag = f"{m}_{d}"
            r = results.get(tag)
            if r is None:
                rows.append(f"{MODEL_LABEL[m]} & {d} & -- & -- & -- \\\\")
                continue
            rows.append(
                f"{MODEL_LABEL[m]} & {d} & "
                f"{r['size_mb']:.1f} & "
                f"{r['latency_ms_median']:.2f} & {r['throughput_samples_per_sec']:.1f} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "Model & Dataset & Size (MB) & Latency p50 (ms) & Throughput @ bs=32 (s$^{-1}$) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# --------------------------------------------------------------------------- #
@click.command()
@click.option("--n-warmup", type=int, default=5, help="warm-up iterations before timing")
@click.option("--n-iters", type=int, default=200, help="single-sample timed iterations")
@click.option("--batch-size", type=int, default=32, help="batch size for throughput")
def main(n_warmup: int, n_iters: int, batch_size: int) -> None:
    results = {}

    for dataset in DATASETS:
        splits = load_dataset(dataset)
        samples = splits.texts("test")
        if len(samples) < n_iters:
            n_iters_local = len(samples)
            log.warning(
                "%s test set has only %d samples, using n_iters=%d",
                dataset,
                len(samples),
                n_iters_local,
            )
        else:
            n_iters_local = n_iters

        for model_type in MODELS:
            tag = f"{model_type}_{dataset}"
            try:
                r = _benchmark(
                    model_type,
                    dataset,
                    samples,
                    n_warmup=n_warmup,
                    n_iters=n_iters_local,
                    batch_size=batch_size,
                )
            except Exception as e:  # noqa: BLE001
                log.error("benchmark failed for %s: %s", tag, e)
                continue
            if r is None:
                continue
            results[tag] = r
            log.info(
                "%s: size=%.1fMB ram_d=%.1fMB lat_p50=%.2fms thru=%.1f s/s",
                tag,
                r["size_mb"],
                r["rss_load_delta_mb"],
                r["latency_ms_median"],
                r["throughput_samples_per_sec"],
            )

    (RESULTS_DIR / "inference_cost.json").write_text(json.dumps(results, indent=2))
    md = _build_md(results)
    (RESULTS_DIR / "inference_cost.md").write_text(md)
    (PAPER_TABLES / "inference_cost.tex").write_text(_build_tex(results))
    print(md)


if __name__ == "__main__":
    main()
