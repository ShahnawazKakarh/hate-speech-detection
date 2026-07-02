# Changelog

All notable changes to this project will be documented in this file. Format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-16

Adds a two-stage cascade architecture and an LLM baseline comparison, closing the v0.2 roadmap.

### Added — v0.2 experiments

- **Two-stage cascade** (§5.7). `src/hsd/cascade.py` wraps TF-IDF (prefilter) + DistilBERT (verifier). `scripts/cascade_sweep.py` sweeps the stage-1 threshold and emits an F1-vs-throughput plot, table, and JSON of the frontier. On Davidson the best cascade gains +0.010 F1-hate over DistilBERT-alone at **9.1× the throughput**; on HateXplain the gain is smaller (+0.003 F1-hate at 1.6× throughput) as expected from the more balanced prior.
- **LLM baselines via OpenRouter** (Appendix C). `src/hsd/models/llm.py` is a cost-managed wrapper around any OpenRouter model (Gemini, Claude, Llama, DeepSeek). Four Gemini 2.5 Flash configs — zero-shot and 4-shot on Davidson and HateXplain — completed for a total spend of $0.03.
- **Cost-safety in the LLM path**: pre-flight cost estimate before any API call, live spend counter on the progress bar, hard `cost_ceiling_usd` per config, and cache-and-resume on quota errors. Errored responses are stored but treated as cache-misses on the next run so transient failures retry naturally.

### Changed

- Paper Appendix C rewritten with the actual LLM numbers.
- `pyproject.toml`: added `openai>=1.40` and `google-genai>=2.7.0` as core dependencies.
- Sklearn LR solver switched from `liblinear` to `lbfgs` so the same code paths handle binary and 3-way classification.
- `.gitignore`: hardened `.env` handling to `.env.*` + `*.bak`.

### v0.2 headline findings (in addition to v0.1)

- On both datasets, fine-tuned DistilBERT still wins F1-hate at the default threshold. Zero-shot Gemini 2.5 Flash ties on Davidson (0.390 vs 0.368) and loses on HateXplain (0.675 vs 0.757).
- Gemini's AUC on Davidson is 10 points below DistilBERT (0.797 vs 0.897) — the LLM's probability scores are less well-ranked on the imbalanced corpus.
- Few-shot exemplars help on HateXplain (+0.048 F1-hate) but not on Davidson (+0.0002) — the balanced 2-hate + 2-non-hate exemplar mix aligns with HateXplain's 31% prior but not with Davidson's 5.8% prior.
- The two-stage cascade lets a moderation system have both the accuracy of DistilBERT and (most of) the throughput of TF-IDF simultaneously.

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
