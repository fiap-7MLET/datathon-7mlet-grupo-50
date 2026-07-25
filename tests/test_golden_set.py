"""
Golden sets da Etapa 4 — cinco casos congelados, em dois momentos da vida da política.

O que estes testes protegem: dada uma política e uma seed fixa, a recomendação é sempre a
mesma. É o contrato de reprodutibilidade do ADR 0002, e é o que permite gravar a demo da
Etapa 8 sabendo de antemão o que a API vai responder.

São dois arquivos, e o contraste entre eles é o argumento da Etapa 4:

- `golden_set.json` — política publicada, treinada em 20.000 clientes. Os cinco casos
  recebem **a mesma** oferta: o posterior convergiu e a exploração cessou.
- `golden_set_em_aprendizado.json` — a mesma política com 500 clientes de treino, com a
  crença ainda difusa. Os casos recebem ofertas **diferentes**: exploração viva.

Se a política publicada for retreinada e a crença mudar, o fingerprint deixa de bater e o
teste falha pedindo `uv run datathon-evaluate --write-golden`. Isso é intencional: mudou o
modelo, mudou a recomendação — o time precisa ver, não descobrir no vídeo.
"""

import json

import pytest

from datathon.api.recommender import OfferRecommender
from datathon.evaluation.evaluator import (
    GOLDEN_SET_PATH,
    LEARNING_GOLDEN_PATH,
    policy_fingerprint,
)
from datathon.policy_artifact import CATALOG_PATH, LOCAL_POLICY_PATH

REGENERATE = "Rode `uv run datathon-evaluate --write-golden` para regravar o golden set."


def _load(path):
    if not path.exists():
        pytest.skip(f"Golden set ausente em {path}. {REGENERATE}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden():
    return _load(GOLDEN_SET_PATH)


@pytest.fixture(scope="module")
def learning_golden():
    return _load(LEARNING_GOLDEN_PATH)


@pytest.fixture(params=["publicado", "em_aprendizado"])
def any_golden(request, golden, learning_golden):
    """Os dois arquivos obedecem ao mesmo contrato — os testes estruturais valem para ambos."""
    return golden if request.param == "publicado" else learning_golden


@pytest.fixture(scope="module")
def policy_state():
    if not LOCAL_POLICY_PATH.exists():
        pytest.skip(f"Política ausente em {LOCAL_POLICY_PATH}. Rode `uv run datathon-train`.")
    return json.loads(LOCAL_POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_golden_set_matches_the_published_policy(golden, policy_state):
    """
    O golden set descreve uma crença específica. Comparar contra outra não diria nada.
    """
    assert golden["policy_fingerprint"] == policy_fingerprint(policy_state), (
        "A política publicada mudou desde que o golden set foi gravado. " + REGENERATE
    )


def test_golden_set_has_the_five_cases_the_spec_asks_for(any_golden):
    cases = any_golden["cases"]

    assert len(cases) == 5
    assert len({case["case_id"] for case in cases}) == 5, "case_id duplicado"
    assert len({case["seed"] for case in cases}) == 5, "seeds repetidas escondem casos"


def test_each_golden_case_reproduces_its_frozen_recommendation(golden, policy_state, catalog):
    """
    O teste que importa: refazer as cinco chamadas e obter exatamente o que está congelado.
    """
    recommender = OfferRecommender(policy_state=policy_state, catalog=catalog)

    obtained = {
        case["case_id"]: recommender.recommend(seed=case["seed"]).arm_id for case in golden["cases"]
    }
    expected = {case["case_id"]: case["expected"]["arm_id"] for case in golden["cases"]}

    assert obtained == expected, REGENERATE


def test_the_learning_snapshot_reproduces_from_its_own_embedded_policy(learning_golden, catalog):
    """
    O golden "em aprendizado" carrega a própria política dentro do arquivo — a crença dele
    não é a publicada. Sem isso não seria reproduzível fora do script que o gerou.
    """
    recommender = OfferRecommender(
        policy_state=learning_golden["policy_state"], catalog=catalog
    )

    obtained = [
        recommender.recommend(seed=case["seed"]).arm_id for case in learning_golden["cases"]
    ]

    assert obtained == [case["expected"]["arm_id"] for case in learning_golden["cases"]]


def test_the_published_policy_has_stopped_exploring(golden):
    """
    Documenta o fato incômodo em vez de escondê-lo: com 20.000 clientes de treino os cinco
    casos recebem a mesma oferta. É convergência, não bug — e quem for gravar a demo
    precisa saber antes, não durante.
    """
    arms = {case["expected"]["arm_id"] for case in golden["cases"]}

    assert len(arms) == 1, (
        "A política publicada voltou a explorar. Bom sinal — mas o roteiro da demo e o "
        "relatório da Etapa 4 falam em convergência para um único braço. " + REGENERATE
    )


def test_the_learning_snapshot_still_explores(learning_golden):
    """A contraparte: cedo no treino, seeds diferentes levam a ofertas diferentes."""
    arms = {case["expected"]["arm_id"] for case in learning_golden["cases"]}

    assert len(arms) > 1, (
        "O snapshot de horizonte curto deveria mostrar exploração; se colapsou, ele perdeu "
        "a função de contraste. " + REGENERATE
    )


def test_golden_recommendations_carry_valid_catalog_offers(any_golden, catalog):
    """Toda oferta congelada tem de existir no catálogo, com o nome de negócio certo."""
    arms = {arm["arm_id"]: arm for arm in catalog["arms"]}

    for case in any_golden["cases"]:
        arm_id = case["expected"]["arm_id"]
        assert arm_id in arms, f"{case['case_id']} aponta para braço inexistente: {arm_id}"
        assert case["expected"]["arm_name"] == arms[arm_id]["name"]


def test_golden_set_states_that_the_policy_is_not_contextual(any_golden):
    """
    Documental, e de propósito: o arquivo será lido por gente do time e pela banca. Se os
    cinco casos vierem com a mesma oferta, tem de estar escrito ali por quê (ADR 0001).
    """
    assert any_golden["contextual"] is False
    assert "seed" in any_golden["nota"]
