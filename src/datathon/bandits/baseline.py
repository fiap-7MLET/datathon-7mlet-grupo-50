"""
Baseline determinístico. Duas variantes implementadas, refletindo as duas formas mais 
comuns de "regra fixa":

- FixedArmPolicy: sempre oferece um único braço fixo, escolhido a priori.
- RandomPolicy: escolha uniforme entre os braços. 

Nenhuma das duas aprende com o feedback — `update()` é um no-op por design.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import BanditPolicy


class FixedArmPolicy(BanditPolicy):
    """Sempre recomenda o mesmo braço, definido na criação da política."""

    def __init__(self, arm_ids: List[str], fixed_arm: str):
        super().__init__(arm_ids)
        self._check_arm(fixed_arm)
        self.fixed_arm = fixed_arm

    @classmethod
    def from_catalog_best_base_rate(cls, arms: List[Dict[str, Any]]) -> "FixedArmPolicy":
        """Constrói a política escolhendo o braço com maior `base_conversion_rate` do catálogo."""
        best = max(arms, key=lambda a: a["base_conversion_rate"])
        arm_ids = [a["arm_id"] for a in arms]
        return cls(arm_ids=arm_ids, fixed_arm=best["arm_id"])

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        return self.fixed_arm

    def update(self, arm_id: str, reward: float) -> None:
        # Regra fixa não aprende com o feedback — comportamento esperado.
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "fixed_arm",
            "arm_ids": self.arm_ids,
            "fixed_arm": self.fixed_arm,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "FixedArmPolicy":
        return cls(arm_ids=state["arm_ids"], fixed_arm=state["fixed_arm"])


class RandomPolicy(BanditPolicy):
    """Escolhe um braço uniformemente ao acaso a cada rodada. Piso de comparação."""

    def __init__(self, arm_ids: List[str], seed: Optional[int] = None):
        super().__init__(arm_ids)
        self._rng = random.Random(seed)

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        return self._rng.choice(self.arm_ids)

    def update(self, arm_id: str, reward: float) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {"algorithm": "random", "arm_ids": self.arm_ids}

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "RandomPolicy":
        return cls(arm_ids=state["arm_ids"])