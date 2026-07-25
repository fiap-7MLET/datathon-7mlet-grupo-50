"""
Testes HTTP da API (Etapa 5) e da página de demo (Etapa 8).

Sobem a aplicação de verdade com `TestClient`, incluindo o `lifespan` — é lá que a política
é carregada, e uma quebra na carga precisa aparecer aqui e não na hora da apresentação.
Dependem da política publicada existir; sem ela, os testes pulam com a instrução de gerá-la.
"""

import pytest
from fastapi.testclient import TestClient

from datathon.api.main import DEMO_PAGE, app
from datathon.policy_artifact import LEARNING_POLICY_PATH, LOCAL_POLICY_PATH

CLIENT_PAYLOAD = {
    "age": 35,
    "job": "student",
    "contact": "cellular",
    "month": "mar",
    "education": "university.degree",
    "poutcome": "success",
    "previous": 2,
}


@pytest.fixture(scope="module")
def client():
    if not LOCAL_POLICY_PATH.exists():
        pytest.skip(f"Política ausente em {LOCAL_POLICY_PATH}. Rode `uv run datathon-train`.")
    with TestClient(app) as client:
        yield client


def test_health_reports_the_policy_being_served(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["algorithm"] == "thompson_sampling"
    assert body["n_arms"] > 1
    assert "publicada" in body["policies_available"]


def test_recommend_returns_an_offer_for_a_client(client):
    body = client.post("/recommend?seed=42", json=CLIENT_PAYLOAD).json()

    assert body["arm_id"].startswith("arm_")
    assert body["arm_name"]
    assert body["policy"] == "publicada"


def test_the_same_seed_always_returns_the_same_offer(client):
    """O contrato que a demo e o golden set dependem (ver README, “Reprodutibilidade”)."""
    offers = {
        client.post("/recommend?seed=7", json=CLIENT_PAYLOAD).json()["arm_id"] for _ in range(5)
    }

    assert len(offers) == 1


def test_the_response_says_the_policy_is_not_contextual(client):
    """
    A API recebe os dados do cliente por exigência da Etapa 5, mas não os usa para decidir.
    O campo é o que impede que a demo prometa personalização (ver README, “Escolhas de design”).
    """
    body = client.post("/recommend?seed=42", json=CLIENT_PAYLOAD).json()

    assert body["contextual"] is False


def test_an_invalid_client_is_rejected_before_reaching_the_policy(client):
    """Idade fora do domínio é erro de validação (422), não uma recomendação silenciosa."""
    response = client.post("/recommend?seed=1", json={**CLIENT_PAYLOAD, "age": 7})

    assert response.status_code == 422


def test_policy_exposes_the_belief_and_evidence_for_every_arm(client):
    """`/policy` é o que torna a decisão auditável — e o que a demo desenha em barras."""
    body = client.get("/policy").json()

    assert body["algorithm"] == "thompson_sampling"
    assert len(body["arms"]) > 1

    beliefs = [arm["belief"] for arm in body["arms"]]
    assert beliefs == sorted(beliefs, reverse=True), "braços vêm do mais promissor ao menos"
    assert all(arm["observations"] >= 0 for arm in body["arms"]), "prior não conta como evidência"


def test_the_chosen_arm_is_the_one_the_policy_believes_most_in(client):
    """
    Coerência entre as duas telas da demo: a oferta recomendada é o braço no topo de
    `/policy`. Só vale com o posterior convergido — que é o estado da política publicada.
    """
    recommended = client.post("/recommend?seed=42", json=CLIENT_PAYLOAD).json()["arm_id"]
    top_arm = client.get("/policy").json()["arms"][0]

    if top_arm["observations"] < 1000:
        pytest.skip("Política ainda explorando; a coerência só é exigível após convergir.")
    assert recommended == top_arm["arm_id"]


def test_personas_match_the_golden_cases(client):
    """A demo só oferece casos que o golden set protege."""
    from datathon.evaluation.evaluator import GOLDEN_CLIENTS

    personas = client.get("/personas").json()

    assert [p["case_id"] for p in personas] == [c["case_id"] for c in GOLDEN_CLIENTS]
    assert [p["seed"] for p in personas] == [c["seed"] for c in GOLDEN_CLIENTS]


def test_asking_for_a_policy_that_was_never_generated_explains_how_to_generate_it(client):
    """Erro acionável em vez de 500 opaco — a demo é montada sob pressão de tempo."""
    if LEARNING_POLICY_PATH.exists():
        pytest.skip("Snapshot da demo presente; este teste cobre a ausência dele.")

    response = client.get("/policy?policy=em_aprendizado")

    assert response.status_code == 404
    assert "datathon-evaluate" in response.json()["detail"]


def test_the_learning_snapshot_is_served_when_present(client):
    """Quando o snapshot existe, ele responde — é o contraste que a demo mostra."""
    if not LEARNING_POLICY_PATH.exists():
        pytest.skip("Snapshot da demo ausente. Rode `uv run datathon-evaluate --write-golden`.")

    body = client.get("/policy?policy=em_aprendizado").json()
    published = client.get("/policy").json()

    assert body["trained_on_n_clients"] < published["trained_on_n_clients"]
    assert sum(arm["observations"] for arm in body["arms"]) < sum(
        arm["observations"] for arm in published["arms"]
    )


def test_the_demo_page_is_served_and_needs_no_external_resources(client):
    """
    A apresentação pode acontecer sem internet: a página não pode depender de CDN. Este
    teste falha se alguém acrescentar um <script src> ou <link> apontando para fora.
    """
    response = client.get("/demo")
    page = response.text

    assert response.status_code == 200
    assert "Ofertas Adaptativas" in page
    assert "http://" not in page.replace("http://127.0.0.1", "")
    assert "https://" not in page
    assert page == DEMO_PAGE.read_text(encoding="utf-8")
