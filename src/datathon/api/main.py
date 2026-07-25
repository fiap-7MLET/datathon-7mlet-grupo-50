"""
Etapa 5 — serviço de recomendação de ofertas.

Recebe os dados de um cliente e devolve a oferta recomendada pela política treinada na
Etapa 3 e registrada no MLflow na Etapa 7.

Uma ressalva importante e deliberada: a política de produção é **não-contextual**
(ADR 0001). Os dados do cliente são validados e registrados no log, mas **não** alteram a
escolha do braço — a recomendação vem da crença populacional aprendida na simulação. Dois
clientes diferentes podem receber a mesma oferta, e isso é o comportamento esperado.

Uso:
    uv run datathon-serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .policy_store import load_catalog, load_policy_state
from .recommender import OfferRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UNKNOWN = "unknown"


class ClientFeatures(BaseModel):
    """
    Um cliente elegível, no mesmo vocabulário do dataset Bank Marketing.

    Campos categóricos aceitam "unknown" — é como o próprio dataset representa ausência,
    e evita que um cadastro incompleto impeça uma recomendação.
    """

    age: int = Field(ge=18, le=120, description="Idade do cliente.")
    job: str = Field(default=UNKNOWN, description="Ocupação (ex: student, retired, admin.).")
    marital: str = Field(default=UNKNOWN, description="Estado civil.")
    education: str = Field(default=UNKNOWN, description="Nível educacional.")
    default: str = Field(default=UNKNOWN, description="Possui crédito em default? yes/no.")
    housing: str = Field(default=UNKNOWN, description="Possui financiamento imobiliário?")
    loan: str = Field(default=UNKNOWN, description="Possui empréstimo pessoal?")
    contact: str = Field(default=UNKNOWN, description="Canal de contato: cellular/telephone.")
    month: str = Field(default=UNKNOWN, description="Mês do último contato (jan-dec).")
    day_of_week: str = Field(default=UNKNOWN, description="Dia da semana do último contato.")
    campaign: int = Field(default=1, ge=1, description="Nº de contatos na campanha atual.")
    previous: int = Field(default=0, ge=0, description="Nº de contatos em campanhas anteriores.")
    poutcome: str = Field(default=UNKNOWN, description="Resultado da campanha anterior.")


class RecommendationResponse(BaseModel):
    arm_id: str = Field(description="Identificador do braço recomendado no catálogo.")
    arm_name: str = Field(description="Nome de negócio da oferta.")
    channel: str | list[str] | None = Field(
        description="Canal ou canais em que a oferta é entregue; nulo no braço de controle."
    )
    score: Optional[float] = Field(
        description="Crença atual da política na taxa de conversão do braço (média posterior)."
    )
    algorithm: str = Field(description="Algoritmo que produziu a recomendação.")
    contextual: Literal[False] = Field(
        default=False,
        description=(
            "A política de produção é não-contextual (ADR 0001): os dados do cliente são "
            "registrados mas não influenciam a escolha do braço."
        ),
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    algorithm: str
    policy_source: str = Field(description="De onde a política foi carregada: mlflow ou local.")
    n_arms: int


app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega política e catálogo uma única vez, no startup — nunca no caminho do request."""
    policy_state, source = load_policy_state()
    app_state["recommender"] = OfferRecommender(policy_state=policy_state, catalog=load_catalog())
    app_state["algorithm"] = policy_state["algorithm"]
    app_state["policy_source"] = source
    app_state["n_arms"] = len(policy_state["arm_ids"])
    logger.info(
        "API pronta: política '%s' (%s braços) carregada de %s.",
        app_state["algorithm"],
        app_state["n_arms"],
        source,
    )
    yield
    app_state.clear()


app = FastAPI(
    title="Plataforma de Ofertas Adaptativas",
    description=__doc__ or "",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Diz se a API está de pé e qual política ela está servindo."""
    if "recommender" not in app_state:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")
    return HealthResponse(
        status="ok",
        algorithm=app_state["algorithm"],
        policy_source=app_state["policy_source"],
        n_arms=app_state["n_arms"],
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(
    client: ClientFeatures,
    seed: Optional[int] = Query(
        default=None,
        description=(
            "Fixa o sorteio Thompson, tornando a resposta reproduzível (ADR 0002). "
            "Sem seed, a política explora e chamadas sucessivas podem variar."
        ),
    ),
) -> RecommendationResponse:
    """Recebe os dados de um cliente e devolve a oferta recomendada."""
    recommender = app_state.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")

    recommendation = recommender.recommend(seed=seed)
    logger.info(
        "Recomendação para cliente (job=%s, contact=%s): %s",
        client.job,
        client.contact,
        recommendation.arm_id,
    )
    return RecommendationResponse(
        arm_id=recommendation.arm_id,
        arm_name=recommendation.arm_name,
        channel=recommendation.channel,
        score=recommendation.score,
        algorithm=app_state["algorithm"],
    )


def run() -> None:
    import uvicorn

    uvicorn.run("datathon.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
