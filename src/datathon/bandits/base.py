"""
Contrato comum a todas as políticas de bandit do projeto.

Toda política implementa três métodos:
    - select_arm(context=None) -> str: escolhe um arm_id
    - update(arm_id, reward) -> None: atualiza o estado interno com o resultado observado
    - to_dict() / from_dict(): serialização do estado (usada pela API e pelo MLflow)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BanditPolicy(ABC):
    """Interface abstrata para uma política de multi-armed bandit."""

    def __init__(self, arm_ids: List[str]):
        if not arm_ids:
            raise ValueError("arm_ids não pode ser vazio.")
        self.arm_ids = list(arm_ids)

    @abstractmethod
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Retorna o arm_id escolhido para a rodada atual."""
        raise NotImplementedError

    @abstractmethod
    def update(self, arm_id: str, reward: float) -> None:
        """Atualiza o estado interno da política com a recompensa observada (0 ou 1)."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializa o estado da política (para salvar em JSON / registrar no MLflow)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, state: Dict[str, Any]) -> "BanditPolicy":
        """Reconstrói a política a partir do estado serializado."""
        raise NotImplementedError

    def reseed(self, seed: Optional[int]) -> None:
        """
        Refixa a fonte de aleatoriedade da política, tornando a próxima escolha reproduzível.

        A seed é uma escolha de quem chama (ex: a API, para o golden set), não faz parte da
        crença aprendida — por isso não é serializada em `to_dict()`.

        Este no-op é o comportamento correto para políticas determinísticas
        (`FixedArmPolicy`, `UCB1Policy`), que não têm fonte de aleatoriedade. As que têm
        sobrescrevem este método.
        """

    def _check_arm(self, arm_id: str) -> None:
        if arm_id not in self.arm_ids:
            raise ValueError(f"arm_id desconhecido: {arm_id!r}. Braços válidos: {self.arm_ids}")
        

