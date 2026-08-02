"""
Testes do seam de recomendação: `OfferRecommender.recommend()`.

Este é o ponto onde a política treinada (estado serializado) encontra o catálogo de
ofertas e produz a recomendação que a API devolve. Os testes observam só esse contrato
público — nunca o estado interno da política.
"""

import pytest

from datathon.api.recommender import OfferRecommender

CATALOG = {
    "arms": [
        {
            "arm_id": "arm_a",
            "name": "Oferta A",
            "channel": "app",
            "base_conversion_rate": 0.10,
        },
        {
            "arm_id": "arm_b",
            "name": "Oferta B",
            "channel": ["email", "sms"],
            "base_conversion_rate": 0.05,
        },
    ]
}


def policy_state(alpha, beta):
    return {
        "algorithm": "thompson_sampling",
        "arm_ids": ["arm_a", "arm_b"],
        "alpha": alpha,
        "beta": beta,
    }


def uninformed():
    """Crença neutra Beta(1,1) nos dois braços: a escolha fica por conta do sorteio."""
    return policy_state(alpha=[1.0, 1.0], beta=[1.0, 1.0])


def test_recommends_the_arm_the_posterior_favours():
    """Com uma crença esmagadoramente a favor de arm_a, a recomendação é arm_a."""
    recommender = OfferRecommender(
        policy_state=policy_state(alpha=[500.0, 1.0], beta=[1.0, 500.0]),
        catalog=CATALOG,
    )

    recommendation = recommender.recommend(seed=42)

    assert recommendation.arm_id == "arm_a"


def test_same_seed_gives_the_same_recommendation():
    """Seed fixa torna a recomendação reproduzível — é o que o golden set e a demo usam."""
    recommender = OfferRecommender(policy_state=uninformed(), catalog=CATALOG)

    first = [recommender.recommend(seed=7).arm_id for _ in range(5)]

    assert len(set(first)) == 1, "mesma seed deve sempre dar o mesmo braço"


def test_recommendation_carries_the_offer_details_and_belief():
    """A resposta traz os dados de negócio do braço e a crença atual sobre a conversão dele."""
    recommender = OfferRecommender(
        policy_state=policy_state(alpha=[500.0, 1.0], beta=[1.0, 500.0]),
        catalog=CATALOG,
    )

    recommendation = recommender.recommend(seed=42)

    assert recommendation.arm_name == "Oferta A"
    assert recommendation.channel == "app"
    # Média da Beta(500, 1) = 500 / 501 ≈ 0.998
    assert recommendation.score == pytest.approx(0.998, abs=1e-3)


def test_policy_trained_on_arms_missing_from_the_catalog_fails_on_load():
    """
    Política e catálogo têm de estar em sincronia. Se a política conhece um braço que o
    catálogo não tem, o erro aparece ao carregar (startup) e não no meio de uma chamada.
    """
    stale = {
        "algorithm": "thompson_sampling",
        "arm_ids": ["arm_a", "arm_removido"],
        "alpha": [1.0, 1.0],
        "beta": [1.0, 1.0],
    }

    with pytest.raises(ValueError, match="arm_removido"):
        OfferRecommender(policy_state=stale, catalog=CATALOG)


def test_without_a_seed_the_policy_explores():
    """Sem seed, o sorteio Thompson varia — é a exploração exigida pela spec."""
    recommender = OfferRecommender(policy_state=uninformed(), catalog=CATALOG)

    drawn = {recommender.recommend().arm_id for _ in range(60)}

    assert drawn == {"arm_a", "arm_b"}, "sem seed os dois braços devem aparecer"


def test_beliefs_reports_the_posterior_mean_and_observations_sorted_best_first():
    """`/policy` (GET) mostra isso braço a braço, do mais promissor ao menos."""
    recommender = OfferRecommender(
        policy_state=policy_state(alpha=[500.0, 1.0], beta=[1.0, 500.0]),
        catalog=CATALOG,
    )

    beliefs = recommender.beliefs()

    assert [b.arm_id for b in beliefs] == ["arm_a", "arm_b"]
    assert beliefs[0].belief == pytest.approx(500 / 501)
    # alpha+beta começa em 2 (prior Beta(1,1) sem evidência): 501 observações reais.
    assert beliefs[0].observations == pytest.approx(499.0)


def test_beliefs_are_none_for_a_policy_without_a_posterior():
    """FixedArmPolicy não mantém crença nem contagem — ambos devem vir `None`, não um erro."""
    stale = {
        "algorithm": "fixed_arm",
        "arm_ids": ["arm_a", "arm_b"],
        "fixed_arm": "arm_a",
    }
    recommender = OfferRecommender(policy_state=stale, catalog=CATALOG)

    beliefs = recommender.beliefs()

    assert all(b.belief is None and b.observations is None for b in beliefs)
