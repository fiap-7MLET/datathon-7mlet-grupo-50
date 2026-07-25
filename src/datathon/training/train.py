"""
Etapa 7 — treino das políticas com rastreamento de experimentos no MLflow.

Roda o baseline e os algoritmos adaptativos no mesmo ambiente de simulação, registra
parâmetros e métricas no MLflow, e publica o estado da política de produção (Thompson
Sampling — ADR 0001) como artefato. É esse artefato que a API carrega no startup.

Uso:
    uv run datathon-train
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import mlflow
import pandas as pd

from ..bandits.baseline import FixedArmPolicy, RandomPolicy
from ..bandits.epsilon_greedy import EpsilonGreedyPolicy
from ..bandits.thompson import ThompsonSamplingPolicy
from ..bandits.ucb import UCB1Policy
from ..policy_artifact import (
    CATALOG_PATH,
    EXPERIMENT_NAME,
    LOCAL_POLICY_PATH,
    POLICY_ARTIFACT_PATH,
    PRODUCTION_POLICY_TAG,
    PROJECT_ROOT,
)
from .simulation import OfferEnvironment, run_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLIENTS_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing_processed.parquet"

N_CLIENTS = 20_000
SAMPLING_SEED = 42
OUTCOME_SEED = 2026
EPSILON = 0.10


def build_policies(arm_ids, arms):
    """As políticas comparadas na Etapa 3, todas partindo do mesmo estado inicial neutro."""
    return {
        "baseline_fixo": FixedArmPolicy.from_catalog_best_base_rate(arms),
        "baseline_aleatorio": RandomPolicy(arm_ids, seed=123),
        "thompson_sampling": ThompsonSamplingPolicy(arm_ids, seed=7),
        "ucb1": UCB1Policy(arm_ids),
        "epsilon_greedy": EpsilonGreedyPolicy(arm_ids, epsilon=EPSILON, seed=99),
    }


def train(n_clients: int = N_CLIENTS) -> Dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    arms = catalog["arms"]
    arm_ids = [a["arm_id"] for a in arms]

    clients = (
        pd.read_parquet(CLIENTS_PATH, engine="pyarrow")
        .sample(n=n_clients, random_state=SAMPLING_SEED)
        .reset_index(drop=True)
    )
    logger.info("Clientes amostrados: %s", clients.shape)

    environment = OfferEnvironment(clients, catalog, seed=OUTCOME_SEED)
    policies = build_policies(arm_ids, arms)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline-vs-adaptativos"):
        mlflow.log_params(
            {
                "n_clients": n_clients,
                "n_arms": len(arm_ids),
                "catalog_version": catalog.get("version"),
                "sampling_seed": SAMPLING_SEED,
                "outcome_seed": OUTCOME_SEED,
                "epsilon": EPSILON,
                "thompson_prior_alpha": 1.0,
                "thompson_prior_beta": 1.0,
                "production_policy": "thompson_sampling",
            }
        )

        results = {name: run_policy(policy, environment) for name, policy in policies.items()}

        baseline_rate = results["baseline_fixo"]["conversion_rate"]
        for name, result in results.items():
            mlflow.log_metric(f"conversao_{name}", result["conversion_rate"])
            mlflow.log_metric(
                f"uplift_vs_baseline_{name}",
                100 * (result["conversion_rate"] / baseline_rate - 1),
            )
            logger.info(
                "%-20s conversão=%.4f  convergiu para=%s",
                name,
                result["conversion_rate"],
                result["converged_arm"],
            )

        production = policies["thompson_sampling"]
        policy_state = {
            **production.to_dict(),
            "trained_on_n_clients": n_clients,
            "random_seed": SAMPLING_SEED,
            "catalog_version": catalog.get("version"),
        }
        LOCAL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_POLICY_PATH.write_text(
            json.dumps(policy_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        mlflow.log_artifact(str(LOCAL_POLICY_PATH), artifact_path=POLICY_ARTIFACT_PATH)
        # Marca este run como o que produziu a política de produção. É por esta tag que a
        # API encontra o artefato a servir, sem depender do Model Registry.
        mlflow.set_tag(PRODUCTION_POLICY_TAG, "true")

        logger.info("Estado da política salvo em %s e logado no MLflow.", LOCAL_POLICY_PATH)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina as políticas e registra no MLflow.")
    parser.add_argument(
        "--n-clients",
        type=int,
        default=N_CLIENTS,
        help=f"Tamanho do horizonte simulado (default: {N_CLIENTS}).",
    )
    args = parser.parse_args()
    train(n_clients=args.n_clients)


if __name__ == "__main__":
    main()
