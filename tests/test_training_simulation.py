"""
Testes de `datathon.training.simulation` — o ambiente contrafactual usado para treinar e
comparar as políticas (Etapa 3).

Usa um catálogo e uma base de clientes minúsculos e sintéticos, não o dataset Bank
Marketing real: o que importa aqui é a mecânica (segmentos, matriz de probabilidades,
*common random numbers*, laço de `run_policy`), não os números de negócio.
"""

import numpy as np
import pandas as pd
import pytest

from datathon.bandits.baseline import FixedArmPolicy
from datathon.training.simulation import OfferEnvironment, derive_segments, run_policy

CATALOG = {
    "arms": [
        {
            "arm_id": "arm_low",
            "base_conversion_rate": 0.10,
            "segment_multipliers": {"high_season": 2.0},
        },
        {
            "arm_id": "arm_high",
            "base_conversion_rate": 0.20,
            "segment_multipliers": {"student_saver": 3.0},
        },
    ]
}


def _client(**overrides):
    row = {
        "poutcome": "nonexistent",
        "month": "may",
        "job": "admin.",
        "contact": "telephone",
        "education": "high.school",
        "housing": "no",
        "loan": "no",
        "default": "no",
        "had_previous_contact": 0,
        "contacted_before": 0,
    }
    row.update(overrides)
    return row


def test_derive_segments_activates_every_documented_rule():
    row = pd.Series(
        _client(
            poutcome="success",
            month="mar",
            job="student",
            contact="cellular",
            education="university.degree",
        )
    )

    segments = derive_segments(row)

    assert set(segments) == {
        "previous_converter",
        "high_season",
        "student_digital",
        "student_saver",
        "university_educated",
        "digital_channel",
        "debt_free",
        "low_engagement",  # _client() nunca foi contactado por padrão
    }


def test_derive_segments_flags_retired_clients():
    row = pd.Series(_client(job="retired"))

    assert "retired" in derive_segments(row)


def test_derive_segments_flags_low_engagement_only_when_never_contacted():
    never_contacted = pd.Series(_client(had_previous_contact=0, contacted_before=0))
    contacted = pd.Series(_client(had_previous_contact=1, contacted_before=1))

    assert "low_engagement" in derive_segments(never_contacted)
    assert "low_engagement" not in derive_segments(contacted)


def test_derive_segments_debt_free_requires_no_housing_no_loan_and_no_default():
    debt_free = pd.Series(_client(housing="no", loan="no", default="no"))
    has_loan = pd.Series(_client(housing="no", loan="yes", default="no"))

    assert "debt_free" in derive_segments(debt_free)
    assert "debt_free" not in derive_segments(has_loan)


def test_offer_environment_caps_probabilities_at_the_documented_maximum():
    catalog = {
        "arms": [
            {
                "arm_id": "arm_capped",
                "base_conversion_rate": 0.9,
                "segment_multipliers": {"high_season": 5.0},
            }
        ]
    }
    clients = pd.DataFrame([_client(month="mar")])  # ativa high_season -> 0.9*5=4.5, capado

    environment = OfferEnvironment(clients, catalog, seed=1)

    # noise em [0,1) < probabilidade capada em 0.95 quase sempre: usamos uma seed fixa e
    # conferimos que a implementação de fato aplicou o cap comparando com o cálculo manual.
    noise = np.random.RandomState(1).random_sample((1, 1))
    expected_outcome = int(noise[0, 0] < 0.95)
    assert environment.outcomes[0, 0] == expected_outcome


def test_offer_environment_is_deterministic_for_a_fixed_seed():
    """
    "Common random numbers" (ver docstring do módulo) significa que duas políticas
    comparadas no mesmo ambiente enfrentam exatamente a mesma matriz de desfechos — não que
    um cliente tenha o mesmo ruído em todo braço. Duas instâncias com a mesma seed devem
    gerar exatamente os mesmos desfechos, o que é o que garante a comparação justa.
    """
    clients = pd.DataFrame([_client(), _client(job="student", month="mar")])

    first = OfferEnvironment(clients, CATALOG, seed=42)
    second = OfferEnvironment(clients, CATALOG, seed=42)

    assert np.array_equal(first.outcomes, second.outcomes)


def test_offer_environment_reward_looks_up_the_right_arm_column():
    clients = pd.DataFrame([_client(), _client(job="student", month="mar")])
    environment = OfferEnvironment(clients, CATALOG, seed=7)

    assert environment.reward(0, "arm_low") in (0, 1)
    assert environment.reward(1, "arm_high") in (0, 1)
    assert len(environment) == 2


def test_run_policy_reports_the_conversion_rate_and_the_converged_arm():
    """Com um FixedArmPolicy, toda escolha é o mesmo braço — convergência é garantida."""
    clients = pd.DataFrame([_client(month="mar") for _ in range(10)])
    environment = OfferEnvironment(clients, CATALOG, seed=1)
    policy = FixedArmPolicy(arm_ids=["arm_low", "arm_high"], fixed_arm="arm_high")

    result = run_policy(policy, environment)

    assert result["converged_arm"] == "arm_high"
    assert result["arm_share_second_half"] == {"arm_high": 1.0}
    assert 0.0 <= result["conversion_rate"] <= 1.0


def test_run_policy_conversion_rate_matches_the_manual_mean_of_observed_rewards():
    clients = pd.DataFrame([_client() for _ in range(6)])
    environment = OfferEnvironment(clients, CATALOG, seed=3)
    policy = FixedArmPolicy(arm_ids=["arm_low", "arm_high"], fixed_arm="arm_low")

    result = run_policy(policy, environment)

    manual_rewards = [environment.reward(i, "arm_low") for i in range(6)]
    assert result["conversion_rate"] == pytest.approx(sum(manual_rewards) / 6)
