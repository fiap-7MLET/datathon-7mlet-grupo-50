"""
Ponte entre a política treinada e a resposta da API.

Recebe o estado serializado da política (o mesmo `to_dict()` que a Etapa 3 grava e que o
MLflow registra) e o catálogo de ofertas, e devolve a recomendação já enriquecida com os
dados de negócio do braço (nome, canal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .. import load_policy


@dataclass(frozen=True)
class Recommendation:
    """Oferta recomendada para uma chamada, com a crença atual sobre ela."""

    arm_id: str
    arm_name: str
    channel: str | list[str] | None
    """Canal de entrega. O catálogo usa um canal, uma lista, ou `null` no braço de controle."""
    score: float | None
    """Crença atual na taxa de conversão do braço. `None` se a política não expõe uma."""


class OfferRecommender:
    def __init__(self, policy_state: Dict[str, Any], catalog: Dict[str, Any]):
        self._policy = load_policy(policy_state)
        self._arms = {arm["arm_id"]: arm for arm in catalog["arms"]}

        unknown = [aid for aid in self._policy.arm_ids if aid not in self._arms]
        if unknown:
            raise ValueError(
                f"A política conhece braços ausentes do catálogo: {unknown}. "
                "Política e catálogo precisam ser da mesma versão."
            )

    def recommend(self, seed: int | None = None) -> Recommendation:
        if seed is not None:
            self._policy.reseed(seed)
        arm_id = self._policy.select_arm()
        arm = self._arms[arm_id]
        return Recommendation(
            arm_id=arm_id,
            arm_name=arm["name"],
            channel=arm["channel"],
            score=self._belief_in(arm_id),
        )

    def _belief_in(self, arm_id: str) -> float | None:
        """Média posterior do braço, quando a política mantém uma (ex: Thompson Sampling)."""
        posterior_mean = getattr(self._policy, "posterior_mean", None)
        return posterior_mean(arm_id) if posterior_mean else None
