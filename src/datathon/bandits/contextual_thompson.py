"""Thompson Sampling contextual simples, com posteriors Beta-Bernoulli por segmento."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import BanditPolicy

SEGMENTS = (
    "previous_converter",
    "student_digital",
    "retired",
    "low_engagement",
    "digital_channel",
    "general",
)


def context_segment(context: Optional[Dict[str, Any]]) -> str:
    """Converte atributos do Bank Marketing em um segmento único e auditável."""
    context = context or {}
    if context.get("segment_override") in SEGMENTS:
        return context["segment_override"]
    if context.get("poutcome") == "success":
        return "previous_converter"
    if context.get("job") == "student" and context.get("contact") == "cellular":
        return "student_digital"
    if context.get("job") == "retired":
        return "retired"
    never_contacted = context.get("previous", 0) == 0 and (
        context.get("contacted_before", 0) == 0
    )
    if never_contacted:
        return "low_engagement"
    if context.get("contact") == "cellular":
        return "digital_channel"
    return "general"


class ContextualThompsonSamplingPolicy(BanditPolicy):
    """Um posterior por (segmento, braço), com prior Beta(1,1)."""

    def __init__(
        self,
        arm_ids: List[str],
        posteriors: Optional[Dict[str, Dict[str, List[float]]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(arm_ids)
        self.posteriors = posteriors or {
            segment: {arm_id: [1.0, 1.0] for arm_id in arm_ids} for segment in SEGMENTS
        }
        self._rng = random.Random(seed)
        self._last_segment = "general"

    def segment_for(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Segmento que este contexto ativaria, sem sortear nem mudar estado."""
        return context_segment(context)

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        self._last_segment = context_segment(context)
        segment_state = self.posteriors[self._last_segment]
        samples = {
            arm_id: self._rng.betavariate(*segment_state[arm_id]) for arm_id in self.arm_ids
        }
        return max(samples, key=samples.get)

    def update(self, arm_id: str, reward: float) -> None:
        self._check_arm(arm_id)
        posterior = self.posteriors[self._last_segment][arm_id]
        posterior[0] += reward
        posterior[1] += 1 - reward

    def reseed(self, seed: Optional[int]) -> None:
        self._rng.seed(seed)

    def posterior_mean(self, arm_id: str, context: Optional[Dict[str, Any]] = None) -> float:
        self._check_arm(arm_id)
        alpha, beta = self.posteriors[context_segment(context)][arm_id]
        return alpha / (alpha + beta)

    def observations(self, arm_id: str, context: Optional[Dict[str, Any]] = None) -> float:
        alpha, beta = self.posteriors[context_segment(context)][arm_id]
        return alpha + beta - 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "contextual_thompson_sampling",
            "arm_ids": self.arm_ids,
            "segments": list(SEGMENTS),
            "posteriors": self.posteriors,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "ContextualThompsonSamplingPolicy":
        return cls(arm_ids=state["arm_ids"], posteriors=state["posteriors"])
