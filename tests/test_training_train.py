"""
Testes de `datathon.training.train`.

`build_policies()` é pura e testável isoladamente: monta o dicionário de políticas
comparadas na Etapa 3 a partir de `arm_ids`/`arms`, sem tocar disco, MLflow ou o dataset
real.

`train()` é exercitado contra um parquet sintético pequeno e um tracking store MLflow
local baseado em SQLite (`sqlite:///...`) — não exige nenhum servidor, só disco, então não
é um mock: é o código real de treino rodando de ponta a ponta. `main()` (parsing da CLI) é
testado isoladamente, com `train()` mockada.
"""

import json

import pandas as pd

from datathon.bandits.baseline import FixedArmPolicy, RandomPolicy
from datathon.bandits.epsilon_greedy import EpsilonGreedyPolicy
from datathon.bandits.thompson import ThompsonSamplingPolicy
from datathon.bandits.ucb import UCB1Policy
from datathon.training import train as train_module
from datathon.training.train import EPSILON, N_CLIENTS, build_policies

SYNTHETIC_COLUMNS = [
    "poutcome", "month", "job", "contact", "education", "housing", "loan",
    "default", "had_previous_contact", "contacted_before",
]


def _synthetic_client(i):
    return {
        "poutcome": "success" if i % 7 == 0 else "nonexistent",
        "month": "mar" if i % 5 == 0 else "may",
        "job": "student" if i % 3 == 0 else "admin.",
        "contact": "cellular" if i % 2 == 0 else "telephone",
        "education": "university.degree" if i % 4 == 0 else "high.school",
        "housing": "no" if i % 2 == 0 else "yes",
        "loan": "no",
        "default": "no",
        "had_previous_contact": int(i % 6 == 0),
        "contacted_before": int(i % 6 == 0),
    }

ARMS = [
    {"arm_id": "arm_a", "base_conversion_rate": 0.10},
    {"arm_id": "arm_b", "base_conversion_rate": 0.30},
]
ARM_IDS = [a["arm_id"] for a in ARMS]


def test_build_policies_returns_the_five_compared_algorithms():
    policies = build_policies(ARM_IDS, ARMS)

    assert set(policies) == {
        "baseline_fixo",
        "baseline_aleatorio",
        "thompson_sampling",
        "ucb1",
        "epsilon_greedy",
    }


def test_build_policies_uses_the_correct_class_for_each_name():
    policies = build_policies(ARM_IDS, ARMS)

    assert isinstance(policies["baseline_fixo"], FixedArmPolicy)
    assert isinstance(policies["baseline_aleatorio"], RandomPolicy)
    assert isinstance(policies["thompson_sampling"], ThompsonSamplingPolicy)
    assert isinstance(policies["ucb1"], UCB1Policy)
    assert isinstance(policies["epsilon_greedy"], EpsilonGreedyPolicy)


def test_build_policies_fixed_baseline_picks_the_best_base_rate_from_the_catalog():
    policies = build_policies(ARM_IDS, ARMS)

    assert policies["baseline_fixo"].fixed_arm == "arm_b"


def test_build_policies_epsilon_greedy_uses_the_module_constant():
    policies = build_policies(ARM_IDS, ARMS)

    assert policies["epsilon_greedy"].epsilon == EPSILON


def test_build_policies_start_from_the_same_neutral_state_every_call():
    """Cada chamada deve devolver políticas "zeradas" — comparação justa entre algoritmos."""
    first = build_policies(ARM_IDS, ARMS)
    second = build_policies(ARM_IDS, ARMS)

    assert first["thompson_sampling"].to_dict() == second["thompson_sampling"].to_dict()
    assert first["ucb1"].to_dict() == second["ucb1"].to_dict()


def test_main_parses_n_clients_and_forwards_it_to_train(monkeypatch):
    """
    `main()` só faz o parsing da CLI e repassa para `train()`; a chamada real de `train()`
    precisa do parquet processado e do MLflow, então aqui isolamos o parsing mockando-a.
    """
    calls = []
    monkeypatch.setattr(train_module, "train", lambda n_clients: calls.append(n_clients))
    monkeypatch.setattr("sys.argv", ["datathon-train", "--n-clients", "500"])

    train_module.main()

    assert calls == [500]


def test_main_defaults_to_the_module_constant_when_no_flag_is_given(monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "train", lambda n_clients: calls.append(n_clients))
    monkeypatch.setattr("sys.argv", ["datathon-train"])

    train_module.main()

    assert calls == [N_CLIENTS]


def test_train_runs_end_to_end_against_a_synthetic_dataset_and_a_local_mlflow_store(
    tmp_path, monkeypatch
):
    """
    `train()` normalmente lê o parquet real (Kaggle, não versionado) e registra no MLflow.
    Aqui substituímos os dois por um dataset sintético pequeno e um tracking store local
    em SQLite (`sqlite:///`), que não exige nenhum servidor — só disco. O backend legado
    baseado em arquivo (`file:`) não serve: a versão instalada do MLflow o bloqueia. É um
    teste de integração de verdade, não um mock do MLflow.
    """
    import mlflow

    clients_path = tmp_path / "clients.parquet"
    pd.DataFrame(
        [_synthetic_client(i) for i in range(30)], columns=SYNTHETIC_COLUMNS
    ).to_parquet(clients_path)
    monkeypatch.setattr(train_module, "CLIENTS_PATH", clients_path)

    local_policy_path = tmp_path / "policy_state.json"
    monkeypatch.setattr(train_module, "LOCAL_POLICY_PATH", local_policy_path)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    results = train_module.train(n_clients=20)

    assert set(results) == {
        "baseline_fixo", "baseline_aleatorio", "thompson_sampling", "ucb1", "epsilon_greedy",
    }
    assert local_policy_path.exists()
    saved = json.loads(local_policy_path.read_text(encoding="utf-8"))
    assert saved["algorithm"] == "thompson_sampling"
    assert saved["trained_on_n_clients"] == 20
