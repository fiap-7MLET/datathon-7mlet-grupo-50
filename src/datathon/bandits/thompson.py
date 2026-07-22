"""
Thompson Sampling não-contextual (Beta-Bernoulli por braço).

Escolhida como política de produção em `notebooks/03_baseline_e_thompson_sampling.py`, pois
não exige hiperparâmetro de exploração manual, a exploração decai sozinha com a evidência acumulada, 
e é mais direta de estender para uma versão contextual no futuro.

Para cada braço mantemos uma distribuição Beta(alpha, beta) representando a crença sobre sua
taxa de conversão. `alpha` e `beta` começam em 1.0 (prior neutro, sem informação prévia).
A cada rodada:
    1. Sorteia-se um valor de Beta(alpha, beta) de cada braço.
    2. Escolhe-se o braço com o maior valor sorteado (equilibra exploração e explotação).
    3. Ao observar a recompensa: alpha += reward, beta += (1 - reward).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import BanditPolicy


class ThompsonSamplingPolicy(BanditPolicy):
    def __init__(
        self,
        arm_ids: List[str],
        alpha: Optional[List[float]] = None,
        beta: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(arm_ids)
        n = len(arm_ids)
        self.alpha: Dict[str, float] = dict(zip(arm_ids, alpha or [1.0] * n))
        self.beta: Dict[str, float] = dict(zip(arm_ids, beta or [1.0] * n))
        self._rng = random.Random(seed)

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        samples = {
            aid: self._rng.betavariate(self.alpha[aid], self.beta[aid])
            for aid in self.arm_ids
        }
        return max(samples, key=samples.get)

    def update(self, arm_id: str, reward: float) -> None:
        self._check_arm(arm_id)
        self.alpha[arm_id] += reward
        self.beta[arm_id] += (1 - reward)

    def posterior_mean(self, arm_id: str) -> float:
        """Média da crença atual sobre a taxa de conversão do braço (alpha / (alpha+beta))."""
        self._check_arm(arm_id)
        return self.alpha[arm_id] / (self.alpha[arm_id] + self.beta[arm_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "thompson_sampling",
            "arm_ids": self.arm_ids,
            "alpha": [self.alpha[aid] for aid in self.arm_ids],
            "beta": [self.beta[aid] for aid in self.arm_ids],
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "ThompsonSamplingPolicy":
        return cls(arm_ids=state["arm_ids"], alpha=state["alpha"], beta=state["beta"])