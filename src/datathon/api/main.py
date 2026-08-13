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
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from ..client_personas import CLIENT_PERSONAS
from .policy_store import load_catalog, load_policy_state
from .recommender import OfferRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UNKNOWN = "unknown"

SegmentChoice = Literal[
    "previous_converter", "student_digital", "retired", "low_engagement",
    "digital_channel", "general"
]
"""Segmento que os atributos do cliente ativam — a base auditável da personalização."""

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
    default: str = Field(
        default=UNKNOWN, description="Está inadimplente em algum crédito? yes/no."
    )
    housing: str = Field(default=UNKNOWN, description="Possui financiamento imobiliário?")
    loan: str = Field(default=UNKNOWN, description="Possui empréstimo pessoal?")
    contact: str = Field(default=UNKNOWN, description="Canal de contato: cellular/telephone.")
    month: str = Field(default=UNKNOWN, description="Mês do último contato (jan-dec).")
    day_of_week: str = Field(default=UNKNOWN, description="Dia da semana do último contato.")
    campaign: int = Field(default=1, ge=1, description="Nº de contatos na campanha atual.")
    previous: int = Field(default=0, ge=0, description="Nº de contatos em campanhas anteriores.")
    poutcome: str = Field(default=UNKNOWN, description="Resultado da campanha anterior.")


class RecommendationResponse(BaseModel):
    recommendation_id: str = Field(
        description=(
            "Identifica esta decisão. Envie de volta em POST /outcome quando souber o "
            "desfecho — é o que liga o feedback ao braço e segmento certos, sem depender de "
            "estado implícito da política."
        )
    )
    arm_id: str = Field(description="Identificador do braço recomendado no catálogo.")
    arm_name: str = Field(description="Nome de negócio da oferta.")
    channel: str | list[str] | None = Field(
        description="Canal ou canais em que a oferta é entregue; nulo no braço de controle."
    )
    score: Optional[float] = Field(
        description="Crença atual da política na taxa de conversão do braço (média posterior)."
    )
    segment: Optional[str] = Field(
        description="Segmento que os atributos do cliente ativaram — a base da personalização."
    )
    algorithm: str = Field(description="Algoritmo que produziu a recomendação.")
    contextual: Literal[True] = Field(
        default=True, description="Os atributos do cliente influenciam a escolha da oferta."
    )


class OutcomeRequest(BaseModel):
    recommendation_id: str = Field(description="O id devolvido por POST /recommend.")
    accepted: bool = Field(description="O cliente aceitou a oferta recomendada?")


class OutcomeResponse(BaseModel):
    recommendation_id: str
    arm_id: str = Field(description="Braço cujo posterior foi atualizado.")
    segment: Optional[str] = Field(description="Segmento cujo posterior foi atualizado.")
    reward: float = Field(description="Recompensa aplicada (1.0 aceitou, 0.0 recusou).")


class SegmentResponse(BaseModel):
    """Resultado isolado de `context_segment()` — sem sortear, sem gastar estado da política."""

    segment: str = Field(description="Segmento que estes atributos de cliente ativam.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    algorithm: str
    policy_source: str = Field(description="De onde a política foi carregada: mlflow ou local.")
    n_arms: int


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega política e catálogo uma única vez, no startup — nunca no caminho do request."""
    catalog = load_catalog()
    policy_state, source = load_policy_state()
    app_state["recommender"] = OfferRecommender(policy_state=policy_state, catalog=catalog)
    app_state["policy_meta"] = {
        "algorithm": policy_state["algorithm"],
        "source": source,
        "n_arms": len(policy_state["arm_ids"]),
        "trained_on_n_clients": policy_state.get("trained_on_n_clients"),
    }
    app_state["decisions"] = {}
    """
    Recomendações à espera de desfecho: recommendation_id -> {arm_id, segment}.

    Em memória, de propósito — este processo é o único a servir a política, e o objetivo é
    fechar o loop exploração/explotação ao vivo na demo, não sobreviver a um restart. Uma
    entrada some assim que POST /outcome a resolve (pop), então o dicionário só cresce com
    recomendações ainda sem resposta.
    """

    logger.info(
        "API pronta: política '%s' (%s braços) carregada de %s.",
        policy_state["algorithm"],
        len(policy_state["arm_ids"]),
        source,
    )
    yield
    app_state.clear()


def _recommender() -> OfferRecommender:
    """Recupera a política carregada, ou explica que a API ainda não subiu — em vez de um 500 opaco."""
    recommender = app_state.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")
    return recommender


app = FastAPI(
    title="Plataforma de Ofertas Adaptativas",
    description=__doc__ or "",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Diz se a API está de pé e qual política ela está servindo."""
    meta = app_state.get("policy_meta")
    if meta is None:
        raise HTTPException(status_code=503, detail="Política ainda não carregada.")
    return HealthResponse(
        status="ok",
        algorithm=meta["algorithm"],
        policy_source=meta["source"],
        n_arms=meta["n_arms"],
    )


@app.get("/policy", response_model=PolicyResponse)
def policy(
    segment: SegmentChoice = Query(default="general", description="Segmento a auditar."),
) -> PolicyResponse:
    """
    Abre a caixa-preta: o que a política acredita sobre **cada** braço, e com quanta
    evidência. É o que sustenta a afirmação "ela aprendeu qual oferta converte melhor".
    """
    recommender = _recommender()
    meta = app_state["policy_meta"]
    return PolicyResponse(
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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Quem abre a raiz do serviço provavelmente quer a demo, não um 404."""
    return RedirectResponse(url="/demo")


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
) -> RecommendationResponse:
    """Recebe os dados de um cliente e devolve a oferta recomendada."""
    recommender = _recommender()

    recommendation = recommender.recommend(context=client.model_dump(), seed=seed)
    recommendation_id = uuid4().hex
    app_state["decisions"][recommendation_id] = {
        "arm_id": recommendation.arm_id,
        "segment": recommendation.segment,
    }
    logger.info(
        "Recomendação %s para cliente (job=%s, contact=%s): %s",
        recommendation_id,
        client.job,
        client.contact,
        recommendation.arm_id,
    )
    return RecommendationResponse(
        recommendation_id=recommendation_id,
        arm_id=recommendation.arm_id,
        arm_name=recommendation.arm_name,
        channel=recommendation.channel,
        score=recommendation.score,
        segment=recommendation.segment,
        algorithm=app_state["policy_meta"]["algorithm"],
    )


@app.post("/outcome", response_model=OutcomeResponse)
def outcome(body: OutcomeRequest) -> OutcomeResponse:
    """
    Fecha o loop: registra se o cliente aceitou ou não a oferta e atualiza o posterior do
    braço/segmento servidos — é isso que faz a próxima chamada explorar/explotar de novo em
    cima de evidência real, e não só da política congelada no treino.

    `recommendation_id` só pode ser resolvido uma vez: a segunda tentativa (ou um id que
    nunca existiu) devolve 404, em vez de contar o mesmo desfecho duas vezes.
    """
    recommender = _recommender()
    decision = app_state["decisions"].pop(body.recommendation_id, None)
    if decision is None:
        raise HTTPException(
            status_code=404,
            detail="recommendation_id desconhecido, ou o desfecho dele já foi registrado.",
        )

    reward = 1.0 if body.accepted else 0.0
    recommender.record_outcome(decision["arm_id"], decision["segment"], reward)
    logger.info(
        "Desfecho %s: braço %s (segmento %s) — %s",
        body.recommendation_id,
        decision["arm_id"],
        decision["segment"],
        "aceitou" if body.accepted else "recusou",
    )
    return OutcomeResponse(
        recommendation_id=body.recommendation_id,
        arm_id=decision["arm_id"],
        segment=decision["segment"],
        reward=reward,
    )


@app.post("/segment", response_model=SegmentResponse)
def segment(client: ClientFeatures) -> SegmentResponse:
    """
    Só o passo "atributos → segmento", sem sortear nem gastar o estado da política.

    Existe para a tela de simulação de cliente mostrar o segmento assim que o usuário
    edita um campo, sem forçar um sorteio Thompson a cada tecla — `POST /recommend` é o
    único caminho que decide de fato e consome aleatoriedade.
    """
    recommender = _recommender()
    result = recommender.segment_for(client.model_dump())
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="A política carregada não é contextual — não há segmento a calcular.",
        )
    return SegmentResponse(segment=result)


def run() -> None:
    import uvicorn

    uvicorn.run("datathon.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
