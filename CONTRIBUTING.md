# Contributing

Thanks for your interest in improving this project. Contributions are welcome — bug fixes, new datasets, additional model baselines, or improvements to the paper are all useful.

## Development setup

```bash
git clone https://github.com/ShahnawazKakarh/hate-speech-detection.git
cd hate-speech-detection
pyenv local 3.11.1            # or any Python >= 3.11
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
pytest -m "not slow"
ruff check src tests
```

## Adding a new model

1. Create `src/hsd/models/<name>.py` with a class exposing `fit(texts, labels)`, `predict_proba(texts)`, `save(path)`, and `load(path)`.
2. Register it in `TRAINERS` inside `src/hsd/train.py`.
3. Add a YAML config in `configs/`.
4. Add a smoke test in `tests/test_models.py`.
5. Add a row to the model labels in `scripts/make_results_table.py`.

## Adding a new dataset

1. Add a `prepare_<name>()` function in `src/hsd/data/prepare.py` that writes `train/val/test.parquet` to `data/processed/<name>/` with columns `text`, `label` (binary), `label3` (3-way).
2. Extend the CLI choices.
3. Add `<model>_<name>.yaml` configs.

## Code style

- Ruff is the single source of truth for lint + formatting (configured in `pyproject.toml`). Run `ruff format` and `ruff check --fix` before committing.
- Type hints encouraged on public functions.
- Tests required for new utilities; smoke tests at minimum for new models.

## Commit messages

Conventional commits, e.g.

- `feat: add OLID dataset loader`
- `fix: handle empty input in clean_text`
- `docs: clarify cross-dataset eval in README`
- `paper: update results table after distilbert run`

## Reporting issues

Include Python version, OS, the command you ran, and the full traceback. For training issues, the relevant section of `artifacts/<run_name>/metrics.json` if it exists.

## License

By contributing you agree that your contributions are licensed under the MIT License of this project.
