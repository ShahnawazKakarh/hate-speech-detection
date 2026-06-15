# Hate Speech Detection in Online Comments

A reproducible benchmark of three approaches to hate-speech classification:

1. **TF-IDF + Logistic Regression** — classical baseline.
2. **Paragraph2Vec (Doc2Vec) + Logistic Regression** — distributed comment embeddings, following Djuric et al. (2015).
3. **DistilBERT fine-tuning** — modern transformer baseline.

Evaluated on two public datasets — **Davidson et al. (2017)** and **HateXplain (Mathew et al., 2021)** — plus a cross-dataset generalization experiment.

[![CI](https://github.com/ShahnawazKakarh/hate-speech-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ShahnawazKakarh/hate-speech-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Why this repo

The hate-speech detection literature is fragmented across decades and methodologies — bag-of-words, paragraph embeddings, transformers — but few works compare them head-to-head on shared corpora with a shared evaluation protocol. This repo does exactly that, in a single command-driven pipeline, so anyone can reproduce the table in the paper or swap in their own model.

## Setup

```bash
pyenv local 3.11.1            # any Python >= 3.11 works
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Optional W&B logging:

```bash
export WANDB_ENTITY=qapulsebysk
export WANDB_PROJECT=hate-speech-detection
wandb login
# then flip `wandb: true` and `report_to: wandb` in the YAML configs
```

## Quickstart

```bash
# 1. Download + prepare datasets (cached to data/processed/)
python -m hsd.data.prepare --dataset all

# 2. Train all three models on Davidson
python -m hsd.train --config configs/tfidf_davidson.yaml
python -m hsd.train --config configs/doc2vec_davidson.yaml
python -m hsd.train --config configs/distilbert_davidson.yaml

# 3. Repeat on HateXplain
python -m hsd.train --config configs/tfidf_hatexplain.yaml
python -m hsd.train --config configs/doc2vec_hatexplain.yaml
python -m hsd.train --config configs/distilbert_hatexplain.yaml

# 4. Cross-dataset generalization
python scripts/cross_dataset_eval.py

# 5. Aggregate into the paper table
python scripts/make_results_table.py
```

## Repo layout

```
src/hsd/
  data/       dataset loaders + preprocessing
  models/     tfidf, doc2vec, distilbert
  train.py    unified training entrypoint (config-driven)
  evaluate.py metrics + cross-dataset eval
configs/      one YAML per (model, dataset)
paper/        arXiv-style LaTeX source
tests/        pytest unit tests
```

## Datasets

| Dataset | Size | Classes | Source |
|---|---|---|---|
| Davidson 2017 | ~24,783 tweets | hate / offensive / neither | [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| HateXplain | ~20,148 posts | hate / offensive / normal | [hate-alert/HateXplain](https://github.com/hate-alert/HateXplain) |

We binarize to `hate vs non-hate` for the primary results table and report 3-way macro-F1 in the appendix.

## Results

Primary test-set metrics, hate as positive class. Numbers are auto-regenerated into `paper/tables/results.{md,tex}` by `scripts/make_results_table.py`.

| Model | Dataset | Acc | F1 (hate) | F1 (macro) | AUC |
|---|---|---|---|---|---|
| TF-IDF + LR  | Davidson   | 0.904 | 0.403 | 0.676 | 0.857 |
| TF-IDF + LR  | HateXplain | 0.828 | 0.731 | 0.802 | 0.892 |
| Doc2Vec + LR | Davidson   | 0.555 | 0.167 | 0.432 | 0.722 |
| Doc2Vec + LR | HateXplain | 0.700 | 0.613 | 0.684 | 0.795 |
| DistilBERT   | Davidson   | 0.939 | 0.368 | 0.668 | 0.897 |
| DistilBERT   | HateXplain | _running_ | — | — | — |

**Observations**

- On the heavily imbalanced Davidson corpus (5.8% hate), DistilBERT achieves the highest AUC but the default 0.5 threshold is too conservative for the minority class — a known artefact of class imbalance, fixable with threshold tuning (covered in the paper's discussion).
- HateXplain (~31% hate) is more balanced and gives all three models a fairer footing.
- Doc2Vec underperforms on Davidson because ~1,150 hate-class training examples isn't enough signal for a 200-dim distributed representation; the original Djuric et al. paper used ~50× more data.

## Citation

If you use this work, please cite:

```bibtex
@misc{khan2026hsd,
  title  = {Benchmarking Classical, Embedding-Based, and Transformer Approaches for Hate Speech Detection in Online Comments},
  author = {Khan, Shahnawaz},
  year   = {2026},
  note   = {Code: https://github.com/ShahnawazKakarh/hate-speech-detection}
}
```

A `CITATION.cff` is included so GitHub renders the "Cite this repository" button.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome for new datasets, model baselines, or paper improvements.

## License

MIT — see [LICENSE](LICENSE).

---

Part of the [QA Pulse by SK](https://skakarh.com) research ecosystem.
