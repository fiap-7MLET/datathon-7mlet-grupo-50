"""
Testes de `datathon.api.policy_store` — de onde a API tira a política treinada.

O caminho feliz (MLflow disponível, artefato registrado) exige um tracking server de
verdade; aqui exercitamos o contrato observável — o fallback para o arquivo local quando o
MLflow falha ou não tem run marcado, e os erros acionáveis quando nada está disponível —
via monkeypatch dos pontos de entrada (`_from_mlflow`, os caminhos de `policy_artifact`).
"""

import json

import pytest

from datathon.api import policy_store
from datathon.policy_artifact import CATALOG_PATH


def test_load_catalog_reads_the_real_catalog_file():
    catalog = policy_store.load_catalog()

    assert "arms" in catalog
    assert catalog["arms"], "catálogo real não deveria estar vazio"


def test_load_policy_state_prefers_mlflow_when_available(monkeypatch):
    fake_state = {"algorithm": "thompson_sampling", "arm_ids": ["arm_a"]}
    monkeypatch.setattr(policy_store, "_from_mlflow", lambda: fake_state)

    state, source = policy_store.load_policy_state()

    assert state == fake_state
    assert source == "mlflow"


def test_load_policy_state_falls_back_to_the_local_file_when_mlflow_is_unavailable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(policy_store, "_from_mlflow", lambda: None)
    local_path = tmp_path / "policy_state.json"
    local_path.write_text(json.dumps({"algorithm": "ucb1", "arm_ids": ["arm_a"]}))
    monkeypatch.setattr(policy_store, "LOCAL_POLICY_PATH", local_path)

    state, source = policy_store.load_policy_state()

    assert state["algorithm"] == "ucb1"
    assert source == f"local:{local_path.name}"


def test_from_local_file_raises_an_actionable_error_when_nothing_was_trained(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(policy_store, "_from_mlflow", lambda: None)
    monkeypatch.setattr(policy_store, "LOCAL_POLICY_PATH", tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="datathon-train"):
        policy_store.load_policy_state()


def test_load_learning_policy_state_returns_none_when_the_snapshot_was_never_generated(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(policy_store, "LEARNING_POLICY_PATH", tmp_path / "missing.json")

    assert policy_store.load_learning_policy_state() is None


def test_load_learning_policy_state_reads_the_snapshot_when_present(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "learning.json"
    snapshot_path.write_text(json.dumps({"algorithm": "thompson_sampling"}))
    monkeypatch.setattr(policy_store, "LEARNING_POLICY_PATH", snapshot_path)

    state = policy_store.load_learning_policy_state()

    assert state == {"algorithm": "thompson_sampling"}


def test_from_mlflow_returns_none_when_the_experiment_does_not_exist(monkeypatch):
    """
    Sem exigir um tracking server de verdade: simulamos um `MlflowClient` cujo experimento
    nunca foi criado, que é exatamente o estado de uma máquina nova sem `mlruns/`.
    `_from_mlflow` importa `MlflowClient` de `mlflow.tracking` a cada chamada, então
    substituir o atributo no módulo real é suficiente para interceptar a instanciação.
    """
    import mlflow.tracking

    class FakeClient:
        def __init__(self):
            pass

        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)

    assert policy_store._from_mlflow() is None


def test_from_mlflow_returns_none_when_no_run_is_tagged_as_production(monkeypatch):
    import mlflow.tracking

    class FakeExperiment:
        experiment_id = "0"

    class FakeClient:
        def __init__(self):
            pass

        def get_experiment_by_name(self, name):
            return FakeExperiment()

        def search_runs(self, *args, **kwargs):
            return []

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)

    assert policy_store._from_mlflow() is None


def test_from_mlflow_swallows_any_failure_and_returns_none(monkeypatch):
    """Qualquer falha no caminho do MLflow (rede, credencial, o que for) não pode derrubar a API."""
    import mlflow.tracking

    class ExplodingClient:
        def __init__(self):
            raise RuntimeError("mlflow indisponível")

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", ExplodingClient)

    assert policy_store._from_mlflow() is None


def test_from_mlflow_honours_the_tracking_uri_env_var(monkeypatch):
    """`MLFLOW_TRACKING_URI`, quando setado, deve ser repassado a `mlflow.set_tracking_uri`."""
    import mlflow
    import mlflow.tracking

    seen_uris = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", seen_uris.append)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-tracking:5000")

    class FakeClient:
        def __init__(self):
            pass

        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)

    policy_store._from_mlflow()

    assert seen_uris == ["http://fake-tracking:5000"]


def test_from_mlflow_downloads_and_parses_the_artifact_of_the_tagged_run(monkeypatch, tmp_path):
    """Caminho feliz completo: experimento existe, tem um run marcado, artefato baixa e parseia."""
    import mlflow.tracking

    artifact_file = tmp_path / "policy_state.json"
    artifact_file.write_text(json.dumps({"algorithm": "thompson_sampling", "arm_ids": ["a"]}))

    class FakeExperiment:
        experiment_id = "0"

    class FakeRunInfo:
        run_id = "run-123"

    class FakeRun:
        info = FakeRunInfo()

    class FakeClient:
        def __init__(self):
            pass

        def get_experiment_by_name(self, name):
            return FakeExperiment()

        def search_runs(self, *args, **kwargs):
            return [FakeRun()]

        def download_artifacts(self, run_id, path):
            assert run_id == "run-123"
            return str(artifact_file)

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)

    state = policy_store._from_mlflow()

    assert state == {"algorithm": "thompson_sampling", "arm_ids": ["a"]}


def test_catalog_path_points_at_a_real_file_on_disk():
    assert CATALOG_PATH.exists()
