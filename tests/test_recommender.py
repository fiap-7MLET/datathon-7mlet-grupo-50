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


def test_non_contextual_policy_has_no_segment():
    """Thompson não-contextual não segmenta — o campo existe, mas vem vazio."""
    recommender = OfferRecommender(policy_state=uninformed(), catalog=CATALOG)

    recommendation = recommender.recommend(seed=1)

    assert recommendation.segment is None
    assert recommender.segment_for({"job": "retired"}) is None


def test_contextual_policy_exposes_the_segment_the_context_activated():
    """`segment_for` é a mesma regra que decide o braço — auditável antes de recomendar."""
    contextual_state = {
        "algorithm": "contextual_thompson_sampling",
        "arm_ids": ["arm_a", "arm_b"],
        "posteriors": {
            segment: {"arm_a": [1.0, 1.0], "arm_b": [1.0, 1.0]}
            for segment in (
                "previous_converter", "student_digital", "retired",
                "low_engagement", "digital_channel", "general",
            )
        },
    }
    recommender = OfferRecommender(policy_state=contextual_state, catalog=CATALOG)

    assert recommender.segment_for({"job": "retired"}) == "retired"
    assert recommender.recommend(context={"job": "retired"}, seed=1).segment == "retired"


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


def test_record_outcome_updates_the_belief_for_the_recommended_arm():
    recommender = OfferRecommender(policy_state=uninformed(), catalog=CATALOG)

    recommender.record_outcome("arm_a", segment=None, reward=1.0)

    belief = recommender.beliefs()
    arm_a = next(b for b in belief if b.arm_id == "arm_a")
    assert arm_a.belief == pytest.approx(2 / 3)  # Beta(1,1) + 1 sucesso -> alpha=2, beta=1


def test_record_outcome_targets_the_segment_it_was_given_not_the_last_selected_one():
    """
    O mesmo contrato que corrige o bug de `_last_segment`: `record_outcome` recebe o
    segmento de onde a recomendação original veio (ex: de um log de decisão), em vez de
    confiar em qual foi o último `select_arm` desta instância.
    """
    contextual_state = {
        "algorithm": "contextual_thompson_sampling",
        "arm_ids": ["arm_a", "arm_b"],
        "posteriors": {
            segment: {"arm_a": [1.0, 1.0], "arm_b": [1.0, 1.0]}
            for segment in (
                "previous_converter", "student_digital", "retired",
                "low_engagement", "digital_channel", "general",
            )
        },
    }
    recommender = OfferRecommender(policy_state=contextual_state, catalog=CATALOG)
    recommender.recommend(context={"job": "retired"}, seed=1)  # muda o último segmento visto
    recommender.recommend(context={"job": "student", "contact": "cellular"}, seed=1)

    recommender.record_outcome("arm_a", segment="retired", reward=1.0)

    retired_belief = recommender.beliefs({"segment_override": "retired"})
    student_belief = recommender.beliefs({"segment_override": "student_digital"})
    assert next(b for b in retired_belief if b.arm_id == "arm_a").belief == pytest.approx(2 / 3)
    assert next(b for b in student_belief if b.arm_id == "arm_a").belief == pytest.approx(0.5)


def test_without_a_seed_the_policy_explores():
    """Sem seed, o sorteio Thompson varia — é a exploração exigida pela spec."""
    recommender = OfferRecommender(policy_state=uninformed(), catalog=CATALOG)

    drawn = {recommender.recommend().arm_id for _ in range(60)}

    assert drawn == {"arm_a", "arm_b"}, "sem seed os dois braços devem aparecer"
