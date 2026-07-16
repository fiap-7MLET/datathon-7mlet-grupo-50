"""
Epsilon-Greedy.

A cada rodada, com probabilidade `epsilon` escolhe um braço aleatório (exploração); com
probabilidade `1 - epsilon` escolhe o braço com maior média observada (explotação).
A taxa de exploração é constante e não decai com a evidência acumulada — essa é a principal
diferença em relação ao Thompson Sampling e ao UCB1.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import BanditPolicy


class EpsilonGreedyPolicy(BanditPolicy):
    def __init__(
        self,
        arm_ids: List[str],
        epsilon: float = 0.1,
        counts: Optional[List[int]] = None,
        sums: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(arm_ids)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon deve estar entre 0 e 1.")
        n = len(arm_ids)
        self.epsilon = epsilon
        self.counts: Dict[str, int] = dict(zip(arm_ids, counts or [0] * n))
        self.sums: Dict[str, float] = dict(zip(arm_ids, sums or [0.0] * n))
        self._rng = random.Random(seed)

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        never_tried = [aid for aid in self.arm_ids if self.counts[aid] == 0]
        if never_tried:
            return never_tried[0]

        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.arm_ids)

        means = {aid: self.sums[aid] / self.counts[aid] for aid in self.arm_ids}
        return max(means, key=means.get)

    def update(self, arm_id: str, reward: float) -> None:
        self._check_arm(arm_id)
        self.counts[arm_id] += 1
        self.sums[arm_id] += reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "epsilon_greedy",
            "arm_ids": self.arm_ids,
            "epsilon": self.epsilon,
            "counts": [self.counts[aid] for aid in self.arm_ids],
            "sums": [self.sums[aid] for aid in self.arm_ids],
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "EpsilonGreedyPolicy":
        return cls(
            arm_ids=state["arm_ids"],
            epsilon=state["epsilon"],
            counts=state["counts"],
            sums=state["sums"],
        )