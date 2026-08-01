from .bandits.base import BanditPolicy
from .bandits.baseline import FixedArmPolicy, RandomPolicy
from .bandits.thompson import ThompsonSamplingPolicy
from .bandits.ucb import UCB1Policy
from .bandits.epsilon_greedy import EpsilonGreedyPolicy
from .bandits.contextual_thompson import ContextualThompsonSamplingPolicy

_REGISTRY = {
    "fixed_arm": FixedArmPolicy,
    "random": RandomPolicy,
    "thompson_sampling": ThompsonSamplingPolicy,
    "ucb1": UCB1Policy,
    "epsilon_greedy": EpsilonGreedyPolicy,
    "contextual_thompson_sampling": ContextualThompsonSamplingPolicy,
}


def load_policy(state: dict) -> BanditPolicy:
    """Reconstrói uma política a partir de um dicionário serializado (campo 'algorithm')."""
    algorithm = state.get("algorithm")
    if algorithm not in _REGISTRY:
        raise ValueError(f"Algoritmo desconhecido: {algorithm!r}. Opções: {list(_REGISTRY)}")
    return _REGISTRY[algorithm].from_dict(state)


__all__ = [
    "BanditPolicy",
    "FixedArmPolicy",
    "RandomPolicy",
    "ThompsonSamplingPolicy",
    "UCB1Policy",
    "EpsilonGreedyPolicy",
    "ContextualThompsonSamplingPolicy",
    "load_policy",
]
