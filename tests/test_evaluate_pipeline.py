"""
Testes de `evaluate()` e `train_snapshot_policy()` (`datathon.evaluation.evaluator`).

As duas funções leem `CLIENTS_PATH` (o parquet processado do Bank Marketing, gerado por
`datathon.data_loader` a partir do CSV do Kaggle — não versionado no repositório) e usam o
catálogo real de ofertas. Aqui apontamos `CLIENTS_PATH` para um parquet sintético pequeno
via monkeypatch — o nome é importado direto no módulo do avaliador (`from
..training.train import CLIENTS_PATH`), então repatchear o atributo no módulo do avaliador
é suficiente. O catálogo usado é o real (`CATALOG_PATH`), porque é só dados de configuração
e já está no repositório.

`main()` (a CLI completa) também é exercitada, com um tracking store MLflow local baseado
em SQLite (não exige servidor) e todos os caminhos de saída (`REPORT_PATH`,
`GOLDEN_SET_PATH`, etc.) redirecionados para `tmp_path`.
"""

import json

import pandas as pd
import pytest

from datathon.evaluation import evaluator
from datathon.evaluation.evaluator import EXPERIMENT_NAME

SYNTHETIC_COLUMNS = [
    "poutcome", "month", "job", "contact", "education", "housing", "loan",
    "default", "had_previous_contact", "contacted_before",
]


def _synthetic_client(**overrides):
    row = {
        "poutcome": "nonexistent", "month": "may", "job": "admin.",
        "contact": "telephone", "education": "high.school", "housing": "no",
        "loan": "no", "default": "no", "had_previous_contact": 0,
        "contacted_before": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def synthetic_clients_parquet(tmp_path, monkeypatch):
    """Substitui o parquet real por um pequeno e sintético, mantendo o catálogo real."""
    rows = [
        _synthetic_client(),
        _synthetic_client(job="student", contact="cellular", month="mar", poutcome="success"),
        _synthetic_client(job="retired", month="oct"),
        _synthetic_client(job="student", education="university.degree"),
        _synthetic_client(housing="yes", loan="yes"),
        _synthetic_client(had_previous_contact=1, contacted_before=1),
    ]
    path = tmp_path / "clients.parquet"
    pd.DataFrame(rows, columns=SYNTHETIC_COLUMNS).to_parquet(path)
    monkeypatch.setattr(evaluator, "CLIENTS_PATH", path)
    return path


def test_evaluate_returns_ranked_metrics_and_meta_for_every_built_policy(
    synthetic_clients_parquet,
):
    metrics, meta = evaluator.evaluate(n_clients=6)

    policy_names = {m.policy for m in metrics}
    assert policy_names == {
        "baseline_fixo", "baseline_aleatorio", "thompson_sampling", "ucb1", "epsilon_greedy",
    }
    assert meta["n_clients"] == 6
    assert meta["sampling_seed"] == evaluator.SAMPLING_SEED
    # ranqueado por conversão, do maior para o menor
    assert [m.conversion_rate for m in metrics] == sorted(
        (m.conversion_rate for m in metrics), reverse=True
    )


def test_evaluate_is_deterministic_for_the_same_n_clients(synthetic_clients_parquet):
    first, _ = evaluator.evaluate(n_clients=6)
    second, _ = evaluator.evaluate(n_clients=6)

    assert [m.conversion_rate for m in first] == [m.conversion_rate for m in second]


def test_train_snapshot_policy_returns_a_serialized_thompson_state(synthetic_clients_parquet):

    catalog = json.loads(evaluator.CATALOG_PATH.read_text(encoding="utf-8"))

    snapshot = evaluator.train_snapshot_policy(catalog, n_clients=6)

    assert snapshot["algorithm"] == "thompson_sampling"
    assert snapshot["trained_on_n_clients"] == 6
    assert len(snapshot["alpha"]) == len(snapshot["arm_ids"])


def test_train_snapshot_policy_is_reproducible(synthetic_clients_parquet):
    catalog = json.loads(evaluator.CATALOG_PATH.read_text(encoding="utf-8"))

    first = evaluator.train_snapshot_policy(catalog, n_clients=6)
    second = evaluator.train_snapshot_policy(catalog, n_clients=6)

    assert first == second


@pytest.fixture
def main_cli_environment(tmp_path, monkeypatch):
    """
    Ambiente completo para `main()`: parquet sintético com clientes suficientes para o
    `LEARNING_HORIZON` (500, hardcoded no módulo — não configurável pela CLI), política
    publicada válida e todos os caminhos de saída redirecionados para `tmp_path`.
    """
    rows = [
        _synthetic_client(
            job="student" if i % 3 == 0 else "admin.",
            month="mar" if i % 5 == 0 else "may",
        )
        for i in range(520)
    ]
    clients_path = tmp_path / "clients.parquet"
    pd.DataFrame(rows, columns=SYNTHETIC_COLUMNS).to_parquet(clients_path)
    monkeypatch.setattr(evaluator, "CLIENTS_PATH", clients_path)

    catalog = json.loads(evaluator.CATALOG_PATH.read_text(encoding="utf-8"))
    arm_ids = [a["arm_id"] for a in catalog["arms"]]
    published_policy = {
        "algorithm": "thompson_sampling",
        "arm_ids": arm_ids,
        "alpha": [2.0] * len(arm_ids),
        "beta": [1.0] * len(arm_ids),
    }
    local_policy_path = tmp_path / "policy_state.json"
    local_policy_path.write_text(json.dumps(published_policy), encoding="utf-8")
    monkeypatch.setattr(evaluator, "LOCAL_POLICY_PATH", local_policy_path)

    monkeypatch.setattr(evaluator, "REPORT_PATH", tmp_path / "reports" / "etapa4.md")
    monkeypatch.setattr(evaluator, "GOLDEN_SET_PATH", tmp_path / "golden" / "golden_set.json")
    monkeypatch.setattr(
        evaluator, "LEARNING_GOLDEN_PATH", tmp_path / "golden" / "golden_set_em_aprendizado.json"
    )
    monkeypatch.setattr(
        evaluator, "LEARNING_POLICY_PATH", tmp_path / "policy_state_em_aprendizado.json"
    )

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    return tmp_path


def test_main_writes_the_report_and_registers_metrics_in_mlflow(
    main_cli_environment, monkeypatch
):
    monkeypatch.setattr("sys.argv", ["datathon-evaluate", "--n-clients", "20"])

    evaluator.main()

    report = evaluator.REPORT_PATH.read_text(encoding="utf-8")
    assert "thompson_sampling" in report
    assert not evaluator.GOLDEN_SET_PATH.exists(), "sem --write-golden, nada deve ser gravado"

    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], filter_string="tags.mlflow.runName = 'avaliacao-etapa4'")
    assert len(runs) == 1
    run = runs[0]
    assert any(name.startswith("conversao_") for name in run.data.metrics)
    assert any(name.startswith("uplift_vs_fixo_") for name in run.data.metrics)


def test_main_with_write_golden_regenerates_both_golden_sets(main_cli_environment, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["datathon-evaluate", "--n-clients", "20", "--write-golden", "--no-mlflow"]
    )

    evaluator.main()

    golden = json.loads(evaluator.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    learning_golden = json.loads(evaluator.LEARNING_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(golden["cases"]) == 5
    assert len(learning_golden["cases"]) == 5
    assert evaluator.LEARNING_POLICY_PATH.exists()
