"""
Testes de contrato: os schemas Pydantic da API (`datathon.api.main`) e o registro de
algoritmos de `datathon.load_policy` (`datathon/__init__.py`).

Ambos são a fronteira entre "dado externo" e "código interno" — validação de entrada e
reconstrução de política a partir de estado serializado — e por isso merecem testes
diretos, sem precisar subir a API nem treinar nada.
"""

import pytest
from pydantic import ValidationError

from datathon import load_policy
from datathon.api.main import ClientFeatures, UNKNOWN
from datathon.bandits.thompson import ThompsonSamplingPolicy


# ---------------------------------------------------------------------------
# datathon.load_policy — registro de algoritmos
# ---------------------------------------------------------------------------


def test_load_policy_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="Algoritmo desconhecido"):
        load_policy({"algorithm": "algoritmo_que_nao_existe"})


def test_load_policy_rejects_a_state_missing_the_algorithm_field():
    with pytest.raises(ValueError, match="Algoritmo desconhecido"):
        load_policy({"arm_ids": ["arm_a"]})


def test_load_policy_reconstructs_the_right_class_from_the_algorithm_field():
    state = {
        "algorithm": "thompson_sampling",
        "arm_ids": ["arm_a", "arm_b"],
        "alpha": [2.0, 1.0],
        "beta": [1.0, 3.0],
    }

    policy = load_policy(state)

    assert isinstance(policy, ThompsonSamplingPolicy)
    assert policy.posterior_mean("arm_a") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ClientFeatures — contrato de entrada de POST /recommend
# ---------------------------------------------------------------------------


def test_client_features_fills_in_documented_defaults():
    client = ClientFeatures(age=35)

    assert client.job == UNKNOWN
    assert client.campaign == 1
    assert client.previous == 0


def test_client_features_rejects_age_outside_the_documented_bounds():
    with pytest.raises(ValidationError):
        ClientFeatures(age=17)
    with pytest.raises(ValidationError):
        ClientFeatures(age=121)


def test_client_features_accepts_the_documented_age_boundaries():
    assert ClientFeatures(age=18).age == 18
    assert ClientFeatures(age=120).age == 120


def test_client_features_rejects_negative_campaign_or_previous_counts():
    with pytest.raises(ValidationError):
        ClientFeatures(age=35, campaign=0)
    with pytest.raises(ValidationError):
        ClientFeatures(age=35, previous=-1)
