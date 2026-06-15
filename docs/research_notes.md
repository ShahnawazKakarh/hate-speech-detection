# Research notes

> Design decisions, prior-art positioning, and the v0.1 → v1.0 plan. Counterpart to the formal `paper/main.tex`.

## Scientific framing

Hate speech detection has been studied for over a decade, but the literature is fragmented across three eras of NLP — bag-of-words / hand-crafted features, distributed word/document embeddings, and contextual transformers. Most published comparisons compare one or two of these on a single dataset, with idiosyncratic preprocessing. The lay reader is left guessing whether the headline number reflects the model, the data, or the train-time hygiene.

**This repo's contribution is not a new model.** It is a clean, reproducible head-to-head of three representative methods on two public datasets, with identical preprocessing, splits, seed, and metrics. The point is to give anyone — a reviewer, a student, an engineer triaging a moderation queue — a defensible baseline they can copy and modify.

## Why these three methods

| Method | Why included |
|---|---|
| **TF-IDF + LR** | The honest baseline. Often more competitive than papers admit, especially on small labelled corpora. If you can't beat it, you can't claim a contribution. |
| **Doc2Vec / Paragraph2Vec** | Faithful re-implementation of [Djuric et al. 2015](https://dl.acm.org/doi/10.1145/2740908.2742760), the canonical "distributed comment embeddings" approach. Tests whether their conclusion holds on smaller modern corpora. |
| **DistilBERT** | A modern transformer baseline that is fast enough to train on a laptop in 15 min. Tests whether contextual pretraining beats both of the above. |

GPT-class generative models are deliberately excluded from v0.1 — they raise separate questions about cost, licensing, and inference latency that deserve their own paper.

## Why these two datasets

| Dataset | Why included |
|---|---|
| **Davidson 2017** | The most-cited hate-speech benchmark. Heavily class-imbalanced (5.8% hate), which exposes the threshold-tuning problem most papers gloss over. |
| **HateXplain** | More balanced (31% hate), comes with rationale annotations, and represents a different platform mix (Twitter + Gab). Provides the contrast that lets the "imbalance vs. model" question be teased apart. |

OLID/OffensEval was considered for v0.1 but excluded. Two datasets is enough to demonstrate the in-domain vs. cross-domain story; a third risks turning the paper into a survey.

## Design decisions called out

- **Binary labels by default.** Both datasets are originally 3-way (hate / offensive / neither). We binarize for the headline table because the hate-vs-everything-else boundary is what platforms actually moderate. Three-way breakdowns live in the appendix.
- **`class_weight="balanced"` for classical models.** Standard practice on imbalanced data. The trade-off (higher recall, lower precision) is acknowledged in the discussion.
- **Default decision threshold (0.5).** Un-tuned, because the cleanest cross-model comparison is at a fixed threshold. The Davidson + DistilBERT result is the canonical case for why this is sometimes misleading; threshold tuning is the obvious ablation.
- **Identical preprocessing.** URL/mention stripping, hashtag-word retention, emoji-to-text, lowercasing, whitespace normalization. Applied to every model so any performance difference is attributable to the model, not the cleaning.
- **Seed 42, single run.** Mean + std across seeds is the obvious v1.0 upgrade. For v0.1, a single deterministic run keeps the wall-clock cost manageable and the results unambiguous.
- **Cross-dataset evaluation as a first-class result.** Most hate-speech papers report in-domain F1 and stop. Generalization is the question that matters for deployment, so we run it explicitly and report the degradation.

## Limitations honestly

- English only.
- No annotator-disagreement modelling (we collapse HateXplain's 3 annotators to majority vote).
- No adversarial / obfuscation evaluation. A user-substituted `n!gger` or `f*ck` will defeat the TF-IDF and Doc2Vec models in ways DistilBERT mostly handles via subword tokenization, but we don't quantify it here.
- Single seed, single hyperparameter setting per model. No claim that ours are optimal.

## v0.1 → v1.0 roadmap

The current code + paper is v0.1: a working benchmark that publishes a defensible table. v1.0 adds:

1. **Threshold-tuning ablation** on the validation set, with the chosen operating point reported alongside the default-threshold result.
2. **3-seed mean ± std** for all six (model, dataset) cells.
3. **3-way classification** appendix (hate / offensive / neither), not just binary.
4. **Target-group analysis** on HateXplain (race, religion, gender, etc.) using the dataset's group annotations.
5. **Adversarial obfuscation** test set built from a small set of substitution rules, evaluated zero-shot.
6. **Inference-cost table** (latency, RAM, model size). Important for the deployment story.

v2.0, if it ever ships, would extend to multilingual (the SER project's transfer-learning playbook applies cleanly).
