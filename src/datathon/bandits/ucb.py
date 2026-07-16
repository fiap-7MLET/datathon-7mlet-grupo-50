"""
UCB1 (Upper Confidence Bound).

A cada rodada, escolhe o braço que maximiza:
    média_observada(braço) + sqrt(2 * ln(t) / n_vezes_escolhido(braço))

Determinístico dado o histórico (sem amostragem aleatória, diferente do Thompson Sampling).
Cada braço é testado uma vez no início (fase de warm-up) antes da fórmula de UCB entrar em
vigor, para evitar divisão por zero e log(0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .base import BanditPolicy


class UCB1Policy(BanditPolicy):
    def __init__(
        self,
        arm_ids: List[str],
        counts: Optional[List[int]] = None,
        sums: Optional[List[float]] = None,
    ):
        super().__init__(arm_ids)
        n = len(arm_ids)
        self.counts: Dict[str, int] = dict(zip(arm_ids, counts or [0] * n))
        self.sums: Dict[str, float] = dict(zip(arm_ids, sums or [0.0] * n))
        self._t = sum(self.counts.values())

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        # Warm-up: garante que cada braço seja testado ao menos uma vez.
        never_tried = [aid for aid in self.arm_ids if self.counts[aid] == 0]
        if never_tried:
            return never_tried[0]

        t = self._t + 1  # rodada atual (1-indexed para o log)
        scores = {}
        for aid in self.arm_ids:
            mean = self.sums[aid] / self.counts[aid]
            bonus = math.sqrt(2 * math.log(t) / self.counts[aid])
            scores[aid] = mean + bonus
        return max(scores, key=scores.get)

    def update(self, arm_id: str, reward: float) -> None:
        self._check_arm(arm_id)
        self.counts[arm_id] += 1
        self.sums[arm_id] += reward
        self._t += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "ucb1",
            "arm_ids": self.arm_ids,
            "counts": [self.counts[aid] for aid in self.arm_ids],
            "sums": [self.sums[aid] for aid in self.arm_ids],
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "UCB1Policy":
        return cls(arm_ids=state["arm_ids"], counts=state["counts"], sums=state["sums"])