# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Testing without the real dataset or a live MLflow server

`data/processed/bank_marketing_processed.parquet` (built by `datathon.data_loader` from
the Kaggle CSV) is not versioned, so tests that need `training.train.CLIENTS_PATH` or
`evaluation.evaluator.CLIENTS_PATH` must monkeypatch that attribute to a small synthetic
parquet fixture instead (see `tests/test_training_train.py`,
`tests/test_evaluate_pipeline.py`) — both names are plain module attributes, patchable
even though `evaluator` imports its copy from `training.train`.

For code that calls `mlflow.*` (`training.train.train()`, `evaluation.evaluator.main()`),
point `mlflow.set_tracking_uri()` at a local SQLite file
(`sqlite:///{tmp_path}/mlflow.db`) — real MLflow code path, no server needed. The legacy
file-based `mlruns/` backend is blocked by installed MLflow ("filesystem tracking backend
is in maintenance mode"); use SQLite, not `file:`.

Run `uv run pytest --cov=src/datathon --cov-report=term-missing` to check coverage; see
`[tool.pytest.ini_options]` in `pyproject.toml` for the configured `testpaths`/`pythonpath`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
