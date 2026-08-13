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
class ArmBelief:
    """O que a política acredita sobre um braço, e sobre quanta evidência."""

    arm_id: str
    arm_name: str
    belief: float | None
    """Média posterior da taxa de conversão. `None` se a política não mantém uma."""
    observations: float | None
    """
    Quantas rodadas de evidência sustentam essa crença.

    É o número que explica a demo: um braço com 18.000 observações e outro com 20 podem ter
    médias parecidas, mas a política tem confiança muito diferente nas duas — e é daí que
    vem a decisão de parar (ou não) de explorar.
    """


@dataclass(frozen=True)
class Recommendation:
    """Oferta recomendada para uma chamada, com a crença atual sobre ela."""

    arm_id: str
    arm_name: str
    channel: str | list[str] | None
    """Canal de entrega. O catálogo usa um canal, uma lista, ou `null` no braço de controle."""
    score: float | None
    """Crença atual na taxa de conversão do braço. `None` se a política não expõe uma."""
    segment: str | None
    """Segmento que os atributos do cliente ativaram. `None` se a política não é contextual."""


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

    def recommend(self, context: Dict[str, Any] | None = None, seed: int | None = None) -> Recommendation:
        if seed is not None:
            self._policy.reseed(seed)
        arm_id = self._policy.select_arm(context)
        arm = self._arms[arm_id]
        return Recommendation(
            arm_id=arm_id,
            arm_name=arm["name"],
            channel=arm["channel"],
            score=self._belief_in(arm_id, context),
            segment=self.segment_for(context),
        )

    def record_outcome(self, arm_id: str, segment: str | None, reward: float) -> None:
        """
        Atualiza o posterior do braço servido com o desfecho real (aceitou/recusou).

        `segment` vem de onde a recomendação foi decidida (o log da chamada original),
        nunca recalculado do estado atual da instância — é o que evita o bug de depender de
        `_last_segment`, que uma chamada concorrente pode ter sobrescrito nesse meio-tempo.
        Passado como `segment_override` porque é isso que `context_segment()` já sabe
        respeitar sem duplicar a lógica de segmentação aqui.
        """
        context = {"segment_override": segment} if segment is not None else None
        self._policy.update(arm_id, reward, context=context)

    def segment_for(self, context: Dict[str, Any] | None = None) -> str | None:
        """
        Segmento que este contexto ativa, sem sortear nem alterar estado.

        Pura de propósito: permite à API mostrar "atributos → segmento" como um passo
        auditável antes de gastar o sorteio Thompson, e alimenta a pré-visualização ao
        vivo da tela de simulação de cliente. `None` se a política não é contextual.
        """
        segment_for = getattr(self._policy, "segment_for", None)
        return segment_for(context) if segment_for else None

    def beliefs(self, context: Dict[str, Any] | None = None) -> list[ArmBelief]:
        """
        A crença atual sobre todos os braços, do mais promissor ao menos, com a evidência
        que a sustenta.

        É o que torna a decisão auditável: em vez de "a API respondeu arm_005", dá para
        mostrar *por que* — e mostrar que os braços preteridos foram avaliados, não
        ignorados. Consumido pelo `GET /policy` e pela página de demo.
        """
        beliefs = [
            ArmBelief(
                arm_id=arm_id,
                arm_name=self._arms[arm_id]["name"],
                belief=self._belief_in(arm_id, context),
                observations=self._observations_for(arm_id, context),
            )
            for arm_id in self._policy.arm_ids
        ]
        return sorted(beliefs, key=lambda b: (b.belief is not None, b.belief), reverse=True)

    def _belief_in(self, arm_id: str, context: Dict[str, Any] | None = None) -> float | None:
        """Média posterior do braço, quando a política mantém uma (ex: Thompson Sampling)."""
        posterior_mean = getattr(self._policy, "posterior_mean", None)
        if not posterior_mean:
            return None
        try:
            return posterior_mean(arm_id, context)
        except TypeError:
            return posterior_mean(arm_id)

    def _observations_for(self, arm_id: str, context: Dict[str, Any] | None = None) -> float | None:
        """
        Rodadas observadas no braço, descontando o prior.

        Beta(1,1) é a crença inicial *sem* nenhuma evidência: alpha+beta começa em 2 e cada
        rodada soma 1. Por isso o −2 — sem ele, uma política recém-criada alegaria duas
        observações que nunca aconteceram.
        """
        observations = getattr(self._policy, "observations", None)
        if observations:
            return observations(arm_id, context)
        alpha, beta = getattr(self._policy, "alpha", None), getattr(self._policy, "beta", None)
        if alpha is None or beta is None:
            return None
        return alpha[arm_id] + beta[arm_id] - 2.0
