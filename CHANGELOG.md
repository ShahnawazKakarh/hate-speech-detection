# Changelog

All notable changes to this project will be documented in this file. Format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-16

Initial public release. A reproducible head-to-head benchmark of three eras of NLP for hate-speech classification, on two public datasets, with a shared protocol.

### Added — pipeline

- Unified, configuration-driven training entrypoint (`hsd.train`) supporting both binary and 3-way classification via the `task` field in YAML.
- Datasets: automated download + preprocessing for [Davidson 2017](https://github.com/t-davidson/hate-speech-and-offensive-language) and [HateXplain](https://github.com/hate-alert/HateXplain), with stratified or official splits and shared cleaning (URL/mention strip, hashtag-word retention, emoji-to-text, lowercasing, whitespace normalisation).
- Models:
  - **TF-IDF + Logistic Regression** (unigram + bigram, class-balanced).
  - **Paragraph2Vec (Doc2Vec) + Logistic Regression** — faithful PV-DBOW re-implementation of Djuric et al. (2015).
  - **DistilBERT fine-tuning** (binary or 3-way, threshold-agnostic AUC, F1-macro early stopping).
- Evaluator with binary + multi-class metrics, classification reports, confusion matrices.

### Added — experiments

- **In-domain results** on both corpora (see [`results/results.md`](results/results.md)).
- **Cross-dataset generalisation** — every model trained on one corpus, evaluated on the other ([`scripts/cross_dataset_eval.py`](scripts/cross_dataset_eval.py)).
- **Threshold tuning** — validation-set F1-hate maximiser per model, with F1-vs-threshold curves ([`scripts/threshold_sweep.py`](scripts/threshold_sweep.py)).
- **Inference cost** — size, latency, throughput on the same hardware ([`scripts/inference_cost.py`](scripts/inference_cost.py)).
- **Target-group bias audit** on HateXplain (Race / Religion / Gender / Sexual Orientation / Other), using the dataset's per-annotator target tags ([`scripts/target_group_analysis.py`](scripts/target_group_analysis.py)).
- **Adversarial obfuscation** — deterministic leet + repetition + spacing perturbation, zero-shot evaluation ([`scripts/adversarial_eval.py`](scripts/adversarial_eval.py)).
- **3-way classification appendix** — same models, same protocol, on the original hate / offensive / neither labels.
- **Multi-seed (mean ± std)** across seeds {42, 7, 1337} for all binary configurations.

### Added — engineering

- Editable pip install with `pyproject.toml`; one dependency file for everything.
- GitHub Actions CI: ruff lint + pytest matrix on Python 3.11 and 3.12.
- Pre-commit: ruff (lint + format), basic file hygiene.
- Unit tests for preprocessing, evaluator, and a model smoke test.
- W&B integration (opt-in via YAML).

### Added — documentation

- Banner-style README with badges, results tables, datasets table, ASCII pipeline, roadmap, citations, disclaimer.
- `docs/research_notes.md` — design rationale, prior-art positioning, v0.1 → v1.0 plan.
- `paper/main.tex` — arXiv-style LaTeX with abstract, intro, related work, methods, results, cross-dataset eval, threshold tuning, inference cost, target-group analysis, adversarial obfuscation, discussion, two appendices.
- `CITATION.cff`, `CONTRIBUTING.md`, MIT `LICENSE`.

### Headline findings

- DistilBERT is the best model on every cell — in-domain and out-of-domain — but the in-domain advantage on the heavily-imbalanced Davidson corpus only materialises after threshold tuning.
- The hate-vs-offensive boundary is consistently the hardest cell in the 3-way setup, regardless of model.
- The best model's recall on hate varies by ~37 points across protected attributes (Race 0.84, Gender 0.47) — a deployment-relevant bias finding.
- Subword tokenisation does **not** confer adversarial robustness: all three models lose 11–16 AUC points under realistic character-level obfuscation, with DistilBERT losing *most* on F1-hate.
- Inference cost gap: TF-IDF is ~180× smaller on disk, ~55× faster per single sample, and ~115× higher throughput than DistilBERT — relevant for two-stage moderation architectures.

[0.1.0]: https://github.com/ShahnawazKakarh/hate-speech-detection/releases/tag/v0.1.0
