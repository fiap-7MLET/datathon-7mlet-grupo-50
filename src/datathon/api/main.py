"""
Etapa 5 — serviço de recomendação de ofertas.

Recebe os dados de um cliente e devolve a oferta recomendada pela política treinada na
Etapa 3 e registrada no MLflow na Etapa 7.

A política de produção é contextual: os atributos do cliente determinam um segmento
auditável e a decisão usa o posterior aprendido para esse segmento.

Uso:
    uv run datathon-serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..client_personas import CLIENT_PERSONAS
from .policy_store import load_catalog, load_learning_policy_state, load_policy_state
from .recommender import OfferRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UNKNOWN = "unknown"

PUBLISHED = "publicada"
LEARNING = "em_aprendizado"

PolicyChoice = Literal["publicada", "em_aprendizado"]
SegmentChoice = Literal[
    "previous_converter", "student_digital", "retired", "low_engagement",
    "digital_channel", "general"
]
"""
Qual crença atender.

`publicada` é a política de produção — a treinada em 20.000 clientes e registrada no
MLflow; é o default e é o que qualquer integração real usa. `em_aprendizado` é um snapshot
de horizonte curto que existe **só para a demo**, para mostrar a mesma política antes de a
exploração decair. Não é uma segunda política de produção.
"""

DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"


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
    policy: PolicyChoice = Field(
        default=PUBLISHED, description="Qual crença respondeu: a publicada ou o snapshot da demo."
    )
    contextual: Literal[True] = Field(
        default=True, description="Os atributos do cliente influenciam a escolha da oferta."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    algorithm: str
    policy_source: str = Field(description="De onde a política foi carregada: mlflow ou local.")
    n_arms: int
    policies_available: list[str] = Field(
        description="Crenças que a API pode atender. `em_aprendizado` só existe se o "
        "snapshot da demo tiver sido gerado."
    )


class ArmBeliefResponse(BaseModel):
    arm_id: str
    arm_name: str
    belief: Optional[float] = Field(
        description="Crença atual na taxa de conversão do braço (média posterior)."
    )
    observations: Optional[float] = Field(
        description="Rodadas de evidência acumuladas no braço, descontado o prior."
    )


class PolicyResponse(BaseModel):
    """Estado da crença, braço a braço — a decisão auditável por trás de `/recommend`."""

    policy: PolicyChoice
    algorithm: str
    trained_on_n_clients: Optional[int] = Field(
        description="Tamanho do horizonte de treino desta crença."
    )
    segment: str
    arms: list[ArmBeliefResponse]


class PersonaResponse(BaseModel):
    case_id: str
    descricao: str
    seed: int = Field(description="Seed que torna a recomendação deste caso reprodutível.")
    client: dict[str, Any]


app_state: dict[str, Any] = {}


def _register(name: str, policy_state: dict[str, Any], catalog: dict[str, Any], source: str) -> None:
    app_state.setdefault("recommenders", {})[name] = OfferRecommender(
        policy_state=policy_state, catalog=catalog
    )
    app_state.setdefault("policy_meta", {})[name] = {
        "algorithm": policy_state["algorithm"],
        "source": source,
        "n_arms": len(policy_state["arm_ids"]),
        "trained_on_n_clients": policy_state.get("trained_on_n_clients"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega política e catálogo uma única vez, no startup — nunca no caminho do request."""
    catalog = load_catalog()
    policy_state, source = load_policy_state()
    _register(PUBLISHED, policy_state, catalog, source)

    learning_state = load_learning_policy_state()
    if learning_state is not None:
        _register(LEARNING, learning_state, catalog, "local:snapshot-demo")

    logger.info(
        "API pronta: política '%s' (%s braços) carregada de %s. Crenças disponíveis: %s.",
        policy_state["algorithm"],
        len(policy_state["arm_ids"]),
        source,
        ", ".join(app_state["recommenders"]),
    )
    yield
    app_state.clear()


def _recommender_for(policy: str) -> OfferRecommender:
    """Recupera a crença pedida, ou explica como gerá-la — em vez de um 500 opaco."""
    recommenders = app_state.get("recommenders") or {}
    if PUBLISHED not in recommenders:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")
    if policy not in recommenders:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Crença '{policy}' não disponível. Gere o snapshot da demo com "
                "`uv run datathon-evaluate --write-golden` e reinicie a API."
            ),
        )
    return recommenders[policy]


app = FastAPI(
    title="Plataforma de Ofertas Adaptativas",
    description=__doc__ or "",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Diz se a API está de pé e qual política ela está servindo."""
    meta = (app_state.get("policy_meta") or {}).get(PUBLISHED)
    if meta is None:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")
    return HealthResponse(
        status="ok",
        algorithm=meta["algorithm"],
        policy_source=meta["source"],
        n_arms=meta["n_arms"],
        policies_available=sorted(app_state["recommenders"]),
    )


@app.get("/policy", response_model=PolicyResponse)
def policy(
    policy: PolicyChoice = Query(
        default=PUBLISHED, description="Qual crença inspecionar."
    ),
    segment: SegmentChoice = Query(default="general", description="Segmento a auditar."),
) -> PolicyResponse:
    """
    Abre a caixa-preta: o que a política acredita sobre **cada** braço, e com quanta
    evidência. É o que sustenta a afirmação "ela aprendeu qual oferta converte melhor".
    """
    recommender = _recommender_for(policy)
    meta = app_state["policy_meta"][policy]
    return PolicyResponse(
        policy=policy,
        algorithm=meta["algorithm"],
        trained_on_n_clients=meta["trained_on_n_clients"],
        segment=segment,
        arms=[
            ArmBeliefResponse(**vars(belief))
            for belief in recommender.beliefs({"segment_override": segment})
        ],
    )


@app.get("/personas", response_model=list[PersonaResponse])
def personas() -> list[PersonaResponse]:
    """
    Os cinco clientes fixos da Etapa 4, com as seeds que os tornam reprodutíveis.

    Mesma fonte que o golden set: a demo não pode oferecer um caso que nenhum teste cobre.
    """
    return [PersonaResponse(**persona) for persona in CLIENT_PERSONAS]


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    """
    Página de apresentação (Etapa 8), servida pela própria API.

    Servida daqui, e não como arquivo aberto do disco, porque a página chama a API por
    `fetch`: mesma origem, sem CORS e sem configuração na hora da demo. O Swagger em
    `/docs` continua sendo a interface de teste técnica; esta é a de narrativa.
    """
    return HTMLResponse(DEMO_PAGE.read_text(encoding="utf-8"))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(
    client: ClientFeatures,
    seed: Optional[int] = Query(
        default=None,
        description=(
            "Fixa o sorteio Thompson, tornando a resposta reproduzível (ver README, “Reprodutibilidade”). "
            "Sem seed, a política explora e chamadas sucessivas podem variar."
        ),
    ),
    policy: PolicyChoice = Query(
        default=PUBLISHED,
        description=(
            "Qual crença atende a chamada. `em_aprendizado` é o snapshot de horizonte "
            "curto usado na demo; produção usa sempre a publicada."
        ),
    ),
) -> RecommendationResponse:
    """Recebe os dados de um cliente e devolve a oferta recomendada."""
    recommender = _recommender_for(policy)

    recommendation = recommender.recommend(context=client.model_dump(), seed=seed)
    logger.info(
        "Recomendação para cliente (job=%s, contact=%s) via política %s: %s",
        client.job,
        client.contact,
        policy,
        recommendation.arm_id,
    )
    return RecommendationResponse(
        arm_id=recommendation.arm_id,
        arm_name=recommendation.arm_name,
        channel=recommendation.channel,
        score=recommendation.score,
        algorithm=app_state["policy_meta"][policy]["algorithm"],
        policy=policy,
    )


def run() -> None:
    import uvicorn

    uvicorn.run("datathon.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
