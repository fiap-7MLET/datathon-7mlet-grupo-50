"""
De onde a API tira a política treinada.

Fonte primária é o MLflow (ADR: o artefato registrado na Etapa 7 é o que vai para produção).
Se o tracking store não estiver disponível — máquina sem `mlruns/`, container sem volume —
cai para o JSON local gravado pelo treino, para que a demo não dependa da infraestrutura de
tracking estar de pé.

O carregamento acontece uma vez, no startup da API. Nada de MLflow no caminho do request.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from ..policy_artifact import (
    CATALOG_PATH,
    EXPERIMENT_NAME,
    LEARNING_POLICY_PATH,
    LOCAL_POLICY_PATH,
    POLICY_ARTIFACT_PATH,
    POLICY_STATE_FILENAME,
    PRODUCTION_POLICY_TAG,
)

logger = logging.getLogger(__name__)


def load_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def load_policy_state() -> Tuple[Dict[str, Any], str]:
    """
    Devolve o estado da política e uma string dizendo de onde ele veio.

    A origem é devolvida junto de propósito: `/health` a expõe, para que na demo se saiba
    se a API está servindo o artefato do MLflow ou o fallback local.
    """
    state = _from_mlflow()
    if state is not None:
        return state, "mlflow"

    logger.warning("MLflow indisponível; carregando política local de %s", LOCAL_POLICY_PATH)
    return _from_local_file(), f"local:{LOCAL_POLICY_PATH.name}"


def load_learning_policy_state() -> Dict[str, Any] | None:
    """
    Snapshot de horizonte curto usado só pela demo, ou `None` se ele não foi gerado.

    Deliberadamente opcional e só local: não é candidato a produção, não vem do MLflow, e a
    API tem de subir normalmente sem ele. Quem quiser o contraste na apresentação roda
    `uv run datathon-evaluate --write-golden`.
    """
    if not LEARNING_POLICY_PATH.exists():
        logger.info(
            "Snapshot em aprendizado ausente (%s); a demo mostrará só a política publicada.",
            LEARNING_POLICY_PATH.name,
        )
        return None
    return json.loads(LEARNING_POLICY_PATH.read_text(encoding="utf-8"))


def _from_mlflow() -> Dict[str, Any] | None:
    """Artefato do run de treino mais recente marcado como política de produção."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        if "MLFLOW_TRACKING_URI" in os.environ:
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None

        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.{PRODUCTION_POLICY_TAG} = 'true'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run_id = runs[0].info.run_id
        artifact = client.download_artifacts(
            run_id, f"{POLICY_ARTIFACT_PATH}/{POLICY_STATE_FILENAME}"
        )
        logger.info("Política carregada do MLflow (run %s).", run_id)
        return json.loads(Path(artifact).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - qualquer falha do MLflow cai para o arquivo local
        logger.warning("Falha ao carregar a política do MLflow: %s", exc)
        return None


def _from_local_file() -> Dict[str, Any]:
    if not LOCAL_POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Nenhuma política encontrada em {LOCAL_POLICY_PATH}. "
            "Rode `uv run datathon-train` antes de subir a API."
        )
    return json.loads(LOCAL_POLICY_PATH.read_text(encoding="utf-8"))
