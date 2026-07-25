"""
Etapa 4 — avaliação das políticas e golden set de casos reprodutíveis.

Duas entregas distintas no mesmo módulo, porque respondem à mesma pergunta ("a política
adaptativa é melhor, e o que exatamente ela faz?") — uma no agregado, outra caso a caso:

1. **Métricas agregadas.** Roda todas as políticas no mesmo ambiente contrafactual da
   Etapa 3 (`training.simulation`) e mede conversão, *uplift* contra os dois baselines e
   convergência (para qual braço a política migrou depois de aprender, e com que
   concentração). Como o ambiente usa *common random numbers*, a diferença entre políticas
   é decisão, não sorte: todas enfrentam o mesmo mundo.

2. **Golden set.** Cinco clientes fixos com seed fixa, cuja recomendação é congelada em
   `tests/golden/golden_set.json` e verificada por teste. É regressão do contrato de
   serving e roteiro para a demo da Etapa 8.

Ressalva que precisa ficar visível: a política de produção é **não-contextual** (ADR 0001).
No golden set, o que faz dois casos receberem ofertas diferentes é a **seed**, não o perfil
do cliente. Os cinco perfis existem para mostrar a variedade da base, não para sugerir uma
personalização que o modelo não entrega.

Uso:
    uv run datathon-evaluate                  # métricas + relatório
    uv run datathon-evaluate --write-golden   # regrava o golden set a partir da política atual
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd

from ..api.recommender import OfferRecommender
from ..policy_artifact import (
    CATALOG_PATH,
    EXPERIMENT_NAME,
    LOCAL_POLICY_PATH,
    PROJECT_ROOT,
)
from ..training.simulation import OfferEnvironment, run_policy
from ..training.train import (
    CLIENTS_PATH,
    N_CLIENTS,
    OUTCOME_SEED,
    SAMPLING_SEED,
    build_policies,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_BASELINE = "baseline_fixo"
"""
Baseline principal do uplift: sempre a melhor oferta segundo o catálogo.

É o que o time faria sem bandit nenhum, e por isso é a régua honesta. O
`baseline_aleatorio` entra como piso, não como alvo.
"""

RANDOM_BASELINE = "baseline_aleatorio"

GOLDEN_SET_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_set.json"
LEARNING_GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_set_em_aprendizado.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "etapa4_avaliacao.md"

LEARNING_HORIZON = 500
"""
Horizonte curto do segundo golden set.

A política publicada foi treinada em 20.000 clientes e o posterior degenerou: `arm_005`
acumulou milhares de observações e os cinco casos golden recebem todos a mesma oferta. Isso
é o Thompson Sampling funcionando — a exploração decai com a evidência — mas mostra só
metade da história. Um snapshot com 500 clientes captura a outra metade, com a política
ainda explorando, e é o que torna a demo da Etapa 8 legível.
"""


@dataclass(frozen=True)
class PolicyMetrics:
    """Desempenho de uma política num horizonte simulado."""

    policy: str
    conversion_rate: float
    uplift_vs_fixed_pct: float
    """Ganho percentual sobre o baseline fixo. Negativo = pior do que não fazer nada."""
    uplift_vs_random_pct: float
    converged_arm: str
    """Braço mais escolhido na segunda metade do horizonte — depois de a política aprender."""
    converged_arm_share: float
    """Fração das escolhas da segunda metade nesse braço. Perto de 1,0 = exploração cessou."""
    arms_explored_second_half: int


def summarize(results: Dict[str, Dict[str, Any]]) -> List[PolicyMetrics]:
    """
    Converte a saída crua de `run_policy` em métricas comparáveis, ordenadas por conversão.

    Pura de propósito: é o que os testes exercitam, sem precisar do dataset nem do MLflow.
    """
    missing = [name for name in (REFERENCE_BASELINE, RANDOM_BASELINE) if name not in results]
    if missing:
        raise ValueError(
            f"Os baselines {missing} precisam estar entre as políticas avaliadas — "
            "sem eles não há uplift a reportar."
        )

    fixed_rate = results[REFERENCE_BASELINE]["conversion_rate"]
    random_rate = results[RANDOM_BASELINE]["conversion_rate"]

    metrics = [
        PolicyMetrics(
            policy=name,
            conversion_rate=result["conversion_rate"],
            uplift_vs_fixed_pct=_uplift_pct(result["conversion_rate"], fixed_rate),
            uplift_vs_random_pct=_uplift_pct(result["conversion_rate"], random_rate),
            converged_arm=result["converged_arm"],
            converged_arm_share=result["arm_share_second_half"][result["converged_arm"]],
            arms_explored_second_half=len(result["arm_share_second_half"]),
        )
        for name, result in results.items()
    ]
    return sorted(metrics, key=lambda m: m.conversion_rate, reverse=True)


def _uplift_pct(rate: float, reference: float) -> float:
    """Ganho percentual sobre uma referência. Referência zerada não define uplift."""
    if reference == 0:
        return float("nan")
    return 100.0 * (rate / reference - 1.0)


def to_markdown(metrics: List[PolicyMetrics], meta: Dict[str, Any]) -> str:
    """Relatório legível — é o que sustenta a Etapa 4 no vídeo e na defesa."""
    header = f"Horizonte simulado: **{meta['n_clients']:,} clientes**".replace(",", ".")
    lines = [
        "# Etapa 4 — Avaliação das políticas",
        "",
        header,
        f"(amostragem seed={meta['sampling_seed']}, desfechos seed={meta['outcome_seed']}, "
        f"catálogo v{meta['catalog_version']}).",
        "",
        "Todas as políticas enfrentam a **mesma** matriz de desfechos (*common random "
        "numbers*), então a diferença entre elas é decisão, não sorte.",
        "",
        "| política | conversão | uplift vs fixo | uplift vs aleatório | convergiu para "
        "| concentração | braços na 2ª metade |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    lines += [
        f"| {m.policy} | {m.conversion_rate:.4f} | {m.uplift_vs_fixed_pct:+.1f}% | "
        f"{m.uplift_vs_random_pct:+.1f}% | {m.converged_arm} | "
        f"{m.converged_arm_share:.1%} | {m.arms_explored_second_half} |"
        for m in metrics
    ]
    lines += [
        "",
        f"Baseline de referência do uplift: `{REFERENCE_BASELINE}` — sempre a melhor oferta "
        "do catálogo, o que se faria sem bandit.",
        "",
        "Leitura da concentração: quanto mais perto de 100%, menos a política ainda explora. "
        "Concentração alta no fim do horizonte é o comportamento esperado do Thompson "
        "Sampling — a exploração decai à medida que a evidência se acumula.",
        "",
        "> Gerado por `uv run datathon-evaluate`. Não editar à mão.",
        "",
    ]
    return "\n".join(lines)


def evaluate(n_clients: int = N_CLIENTS) -> Tuple[List[PolicyMetrics], Dict[str, Any]]:
    """Roda todas as políticas no ambiente de simulação e devolve métricas + metadados."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    arms = catalog["arms"]
    arm_ids = [a["arm_id"] for a in arms]

    clients = (
        pd.read_parquet(CLIENTS_PATH, engine="pyarrow")
        .sample(n=n_clients, random_state=SAMPLING_SEED)
        .reset_index(drop=True)
    )
    environment = OfferEnvironment(clients, catalog, seed=OUTCOME_SEED)

    results = {
        name: run_policy(policy, environment)
        for name, policy in build_policies(arm_ids, arms).items()
    }
    meta = {
        "n_clients": n_clients,
        "n_arms": len(arm_ids),
        "sampling_seed": SAMPLING_SEED,
        "outcome_seed": OUTCOME_SEED,
        "catalog_version": catalog.get("version"),
    }
    return summarize(results), meta


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

GOLDEN_CLIENTS: List[Dict[str, Any]] = [
    {
        "case_id": "previous_converter",
        "descricao": "Já converteu em campanha anterior — segmento de maior conversão real (65,1%).",
        "seed": 101,
        "client": {
            "age": 47,
            "job": "management",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "campaign": 1,
            "previous": 2,
            "poutcome": "success",
        },
    },
    {
        "case_id": "student_digital",
        "descricao": "Estudante em canal digital, em alta temporada (março).",
        "seed": 202,
        "client": {
            "age": 22,
            "job": "student",
            "marital": "single",
            "education": "university.degree",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "mar",
            "day_of_week": "tue",
            "campaign": 1,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "retired",
        "descricao": "Aposentado — perfil conservador, conversão real de 17,6%.",
        "seed": 303,
        "client": {
            "age": 68,
            "job": "retired",
            "marital": "married",
            "education": "basic.4y",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "oct",
            "day_of_week": "wed",
            "campaign": 2,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "digital_channel_massa",
        "descricao": "Cliente típico da maior fatia da base (54% em canal digital).",
        "seed": 404,
        "client": {
            "age": 38,
            "job": "admin.",
            "marital": "married",
            "education": "high.school",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "jul",
            "day_of_week": "thu",
            "campaign": 3,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "low_engagement",
        "descricao": "Nunca contactado antes — o caso mais frio, conversão real de 3,7%.",
        "seed": 505,
        "client": {
            "age": 31,
            "job": "blue-collar",
            "marital": "single",
            "education": "basic.9y",
            "default": "unknown",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "jun",
            "day_of_week": "fri",
            "campaign": 1,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
]

FINGERPRINT_FIELDS = frozenset({"arm_ids", "alpha", "beta", "counts", "values", "arm_id"})
"""
Campos do estado serializado que efetivamente mudam a decisão.

Cobre as quatro políticas serializáveis: Beta-Bernoulli (`alpha`/`beta`), contadores
(`counts`/`values`) e braço fixo (`arm_id`). Metadados de treino ficam de fora de propósito
— retreinar com o mesmo resultado não deve invalidar o golden set.
"""


def policy_fingerprint(policy_state: Dict[str, Any]) -> str:
    """
    Impressão digital da crença aprendida.

    Serve para o teste do golden set saber se está comparando contra a mesma política que
    gerou os valores congelados: se o modelo for retreinado, o fingerprint muda e o teste
    manda regravar, em vez de acusar uma falha que não existe.
    """
    core = {k: v for k, v in policy_state.items() if k in FINGERPRINT_FIELDS}
    payload = json.dumps(
        {"algorithm": policy_state.get("algorithm"), **core}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_NOTE = (
    "Política não-contextual (ADR 0001): o que distingue os casos é a seed, não o perfil "
    "do cliente. Regravar com `uv run datathon-evaluate --write-golden` sempre que a "
    "política for retreinada."
)


def build_golden_set(
    policy_state: Dict[str, Any],
    catalog: Dict[str, Any],
    nota: str = DEFAULT_NOTE,
    embed_policy_state: bool = False,
) -> Dict[str, Any]:
    """
    Congela a recomendação atual para os cinco clientes fixos.

    `embed_policy_state` guarda a própria política dentro do arquivo. É o que o golden "em
    aprendizado" usa: a crença dele não é a publicada em `data/processed/`, é um snapshot
    de horizonte curto que só existe aqui — sem embutir, não haveria como reproduzi-lo.
    """
    recommender = OfferRecommender(policy_state=policy_state, catalog=catalog)
    cases = [
        {
            **case,
            "expected": _as_expected(recommender.recommend(seed=case["seed"])),
        }
        for case in GOLDEN_CLIENTS
    ]
    golden = {
        "algorithm": policy_state.get("algorithm"),
        "policy_fingerprint": policy_fingerprint(policy_state),
        "catalog_version": catalog.get("version"),
        "contextual": False,
        "nota": nota,
        "cases": cases,
    }
    if embed_policy_state:
        golden["policy_state"] = policy_state
    return golden


def train_snapshot_policy(catalog: Dict[str, Any], n_clients: int = LEARNING_HORIZON) -> Dict[str, Any]:
    """
    Treina um Thompson Sampling em horizonte curto e devolve o estado serializado.

    Mesmas seeds e mesmo ambiente do treino de produção — só o horizonte muda. Assim o
    snapshot é literalmente "a política de produção mais cedo na vida dela", e não um
    segundo modelo com outra configuração.
    """
    clients = (
        pd.read_parquet(CLIENTS_PATH, engine="pyarrow")
        .sample(n=n_clients, random_state=SAMPLING_SEED)
        .reset_index(drop=True)
    )
    environment = OfferEnvironment(clients, catalog, seed=OUTCOME_SEED)
    policy = build_policies([a["arm_id"] for a in catalog["arms"]], catalog["arms"])[
        "thompson_sampling"
    ]
    run_policy(policy, environment)
    return {**policy.to_dict(), "trained_on_n_clients": n_clients}


def _as_expected(recommendation: Any) -> Dict[str, Any]:
    return {
        "arm_id": recommendation.arm_id,
        "arm_name": recommendation.arm_name,
        "score": recommendation.score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etapa 4: avalia as políticas e mantém o golden set."
    )
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="Regrava tests/golden/golden_set.json a partir da política publicada.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Não registrar run no MLflow.")
    args = parser.parse_args()

    metrics, meta = evaluate(n_clients=args.n_clients)
    for m in metrics:
        logger.info(
            "%-20s conversão=%.4f  uplift vs fixo=%+.1f%%  convergiu para=%s (%.0f%%)",
            m.policy,
            m.conversion_rate,
            m.uplift_vs_fixed_pct,
            m.converged_arm,
            100 * m.converged_arm_share,
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(to_markdown(metrics, meta), encoding="utf-8")
    logger.info("Relatório escrito em %s", REPORT_PATH)

    if not args.no_mlflow:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="avaliacao-etapa4"):
            mlflow.log_params(meta)
            for m in metrics:
                mlflow.log_metric(f"conversao_{m.policy}", m.conversion_rate)
                mlflow.log_metric(f"uplift_vs_fixo_{m.policy}", m.uplift_vs_fixed_pct)
                mlflow.log_metric(f"concentracao_{m.policy}", m.converged_arm_share)
            mlflow.log_artifact(str(REPORT_PATH))

    if args.write_golden:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

        published = json.loads(LOCAL_POLICY_PATH.read_text(encoding="utf-8"))
        _write_golden(GOLDEN_SET_PATH, build_golden_set(published, catalog))

        snapshot = train_snapshot_policy(catalog)
        _write_golden(
            LEARNING_GOLDEN_PATH,
            build_golden_set(
                snapshot,
                catalog,
                nota=(
                    f"Snapshot da mesma política com apenas {LEARNING_HORIZON} clientes de "
                    "treino, quando a exploração ainda está ativa. Contraste com "
                    "`golden_set.json` (20.000 clientes), onde o posterior já convergiu para "
                    "um único braço. A política continua não-contextual (ADR 0001): o que "
                    "varia entre os casos é a seed."
                ),
                embed_policy_state=True,
            ),
        )


def _write_golden(path, golden: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    arms = sorted({case["expected"]["arm_id"] for case in golden["cases"]})
    logger.info(
        "Golden set gravado em %s (fingerprint %s, braços distintos: %s).",
        path,
        golden["policy_fingerprint"],
        ", ".join(arms),
    )


if __name__ == "__main__":
    main()
