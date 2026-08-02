"""
Testes dos caminhos de erro de `datathon.api.main` que não dependem do estado dos dados
locais (política publicada / snapshot presentes ou não) — ao contrário de `test_api.py`,
que sobe a aplicação de verdade via `TestClient` e depende de `LOCAL_POLICY_PATH` existir.

Aqui exercitamos `_recommender_for`/`health` diretamente contra um `app_state` controlado
pelo teste, para cobrir os 503 (API ainda não subiu) e o 404 por nome de crença
desconhecido — este último inatingível via HTTP porque `PolicyChoice` é um `Literal` que a
validação do Pydantic já barra antes de chegar à função.
"""

import pytest
from fastapi import HTTPException

from datathon.api import main


@pytest.fixture(autouse=True)
def clean_app_state():
    """Isola cada teste: `app_state` é um dict de módulo, compartilhado entre chamadas."""
    original = dict(main.app_state)
    main.app_state.clear()
    yield
    main.app_state.clear()
    main.app_state.update(original)


def test_recommender_for_raises_503_before_the_app_has_started():
    with pytest.raises(HTTPException) as exc_info:
        main._recommender_for(main.PUBLISHED)

    assert exc_info.value.status_code == 503


def test_health_raises_503_before_the_app_has_started():
    with pytest.raises(HTTPException) as exc_info:
        main.health()

    assert exc_info.value.status_code == 503


def test_recommender_for_raises_404_for_a_belief_that_was_never_registered():
    """
    Via HTTP isso é impossível (o `Literal` do Pydantic barra antes), mas a função
    continua sendo o que protege contra um nome inválido vindo de dentro do processo.
    """
    main.app_state["recommenders"] = {main.PUBLISHED: object()}

    with pytest.raises(HTTPException) as exc_info:
        main._recommender_for("crenca_que_nao_existe")

    assert exc_info.value.status_code == 404
    assert "datathon-evaluate" in exc_info.value.detail


def test_run_starts_uvicorn_with_the_documented_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda target, **kwargs: calls.append((target, kwargs))
    )

    main.run()

    assert len(calls) == 1
    target, kwargs = calls[0]
    assert target == "datathon.api.main:app"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False
