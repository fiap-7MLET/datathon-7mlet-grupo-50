"""
O contrato entre o treino e o serviço: onde a política treinada é publicada e como
encontrá-la de novo.

Treino (`datathon.training.train`) e serviço (`datathon.api.policy_store`) precisam
concordar exatamente sobre nome do experimento, caminho do artefato e tag. Manter isso num
lugar só evita que renomear o artefato quebre a carga da API em silêncio.
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "datathon-ofertas-mab"
"""Experimento MLflow onde os runs de treino são registrados."""

POLICY_ARTIFACT_PATH = "policy"
"""Diretório do artefato dentro do run MLflow."""

POLICY_STATE_FILENAME = "policy_state.json"
"""Estado serializado da política de produção (`BanditPolicy.to_dict()` + metadados)."""

PRODUCTION_POLICY_TAG = "production_policy"
"""Tag que marca o run cujo artefato deve ir para produção."""

LEARNING_POLICY_FILENAME = "policy_state_em_aprendizado.json"
"""
Snapshot da mesma política num horizonte curto, quando a exploração ainda está ativa.

Não é candidata a produção: existe para a demo (Etapa 8) poder contrastar, lado a lado, uma
crença já convergida com uma ainda difusa. Gerado por `datathon-evaluate --write-golden`.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "data" / "synthetic_enrichment" / "offer_catalog.json"
LOCAL_POLICY_PATH = PROJECT_ROOT / "data" / "processed" / POLICY_STATE_FILENAME
"""
Cópia local do artefato, escrita pelo treino e usada pela API se o MLflow não estiver
disponível. Não confundir com `thompson_policy_state.json`, que é a saída do notebook 03
(Etapa 3) e não é gerenciada por este código.
"""

LEARNING_POLICY_PATH = LOCAL_POLICY_PATH.parent / LEARNING_POLICY_FILENAME
