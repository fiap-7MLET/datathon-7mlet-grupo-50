"""
Testes unitários das políticas de bandit (`datathon.bandits`).

Cada política é exercitada isoladamente, sem ambiente nem catálogo: `select_arm`,
`update`, `to_dict`/`from_dict` (o contrato de serialização usado pela API e pelo
MLflow) e as regras específicas de cada algoritmo (warm-up do UCB1, epsilon do
Epsilon-Greedy, prior neutro do Thompson Sampling).
"""

import math

import pytest

from datathon.bandits.base import BanditPolicy
from datathon.bandits.baseline import FixedArmPolicy, RandomPolicy
from datathon.bandits.epsilon_greedy import EpsilonGreedyPolicy
from datathon.bandits.thompson import ThompsonSamplingPolicy
from datathon.bandits.ucb import UCB1Policy

ARM_IDS = ["arm_a", "arm_b", "arm_c"]


# ---------------------------------------------------------------------------
# BanditPolicy (contrato base)
# ---------------------------------------------------------------------------


def test_base_policy_refuses_an_empty_arm_list():
    with pytest.raises(ValueError, match="arm_ids"):
        FixedArmPolicy(arm_ids=[], fixed_arm="arm_a")


def test_check_arm_accepts_known_arms_and_rejects_unknown_ones():
    policy = RandomPolicy(arm_ids=ARM_IDS, seed=1)

    policy._check_arm("arm_a")  # não levanta
    with pytest.raises(ValueError, match="desconhecido"):
        policy._check_arm("arm_desconhecido")


def test_reseed_is_a_noop_by_default_for_policies_without_randomness():
    """FixedArmPolicy não tem fonte de aleatoriedade; reseed() não deve fazer nada nem falhar."""
    policy = FixedArmPolicy(arm_ids=ARM_IDS, fixed_arm="arm_a")

    policy.reseed(123)

    assert policy.select_arm() == "arm_a"


def test_abstract_methods_raise_not_implemented_when_invoked_via_super():
    """
    Cobre o corpo dos métodos abstratos diretamente — o caminho normal (instanciar a ABC)
    é bloqueado pelo Python, então a única forma de exercitar essas linhas é chamando
    `super()` explicitamente a partir de uma subclasse concreta.
    """

    class Bare(BanditPolicy):
        def select_arm(self, context=None):
            return super().select_arm(context)

        def update(self, arm_id, reward):
            return super().update(arm_id, reward)

        def to_dict(self):
            return super().to_dict()

        @classmethod
        def from_dict(cls, state):
            return super().from_dict(state)

    bare = Bare(arm_ids=ARM_IDS)

    with pytest.raises(NotImplementedError):
        bare.select_arm()
    with pytest.raises(NotImplementedError):
        bare.update("arm_a", 1.0)
    with pytest.raises(NotImplementedError):
        bare.to_dict()
    with pytest.raises(NotImplementedError):
        Bare.from_dict({})


# ---------------------------------------------------------------------------
# FixedArmPolicy / RandomPolicy (baseline.py)
# ---------------------------------------------------------------------------


def test_fixed_arm_policy_always_recommends_the_same_arm_and_never_learns():
    policy = FixedArmPolicy(arm_ids=ARM_IDS, fixed_arm="arm_b")

    for _ in range(5):
        assert policy.select_arm() == "arm_b"
    policy.update("arm_b", 1.0)  # no-op, não deve alterar a escolha
    assert policy.select_arm() == "arm_b"


def test_fixed_arm_policy_rejects_a_fixed_arm_outside_the_catalog():
    with pytest.raises(ValueError, match="desconhecido"):
        FixedArmPolicy(arm_ids=ARM_IDS, fixed_arm="arm_fora_do_catalogo")


def test_fixed_arm_policy_from_catalog_picks_the_best_base_rate():
    arms = [
        {"arm_id": "arm_a", "base_conversion_rate": 0.10},
        {"arm_id": "arm_b", "base_conversion_rate": 0.30},
        {"arm_id": "arm_c", "base_conversion_rate": 0.05},
    ]

    policy = FixedArmPolicy.from_catalog_best_base_rate(arms)

    assert policy.fixed_arm == "arm_b"
    assert policy.select_arm() == "arm_b"


def test_fixed_arm_policy_round_trips_through_dict():
    original = FixedArmPolicy(arm_ids=ARM_IDS, fixed_arm="arm_c")

    restored = FixedArmPolicy.from_dict(original.to_dict())

    assert restored.arm_ids == original.arm_ids
    assert restored.fixed_arm == original.fixed_arm
    assert original.to_dict()["algorithm"] == "fixed_arm"


def test_random_policy_only_ever_returns_known_arms():
    policy = RandomPolicy(arm_ids=ARM_IDS, seed=0)

    draws = {policy.select_arm() for _ in range(50)}

    assert draws <= set(ARM_IDS)
    assert len(draws) > 1, "com 50 sorteios em 3 braços, esperar variedade"


def test_random_policy_reseed_makes_the_next_draw_reproducible():
    policy = RandomPolicy(arm_ids=ARM_IDS, seed=1)

    policy.reseed(99)
    first = policy.select_arm()
    policy.reseed(99)
    second = policy.select_arm()

    assert first == second


def test_random_policy_update_is_a_noop():
    policy = RandomPolicy(arm_ids=ARM_IDS, seed=1)

    policy.update("arm_a", 1.0)  # não deve levantar nem mudar o comportamento externo

    assert policy.to_dict() == {"algorithm": "random", "arm_ids": ARM_IDS}


def test_random_policy_round_trips_through_dict():
    original = RandomPolicy(arm_ids=ARM_IDS, seed=5)

    restored = RandomPolicy.from_dict(original.to_dict())

    assert restored.arm_ids == original.arm_ids


# ---------------------------------------------------------------------------
# EpsilonGreedyPolicy
# ---------------------------------------------------------------------------


def test_epsilon_greedy_rejects_an_epsilon_outside_zero_and_one():
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=1.5)


def test_epsilon_greedy_tries_every_arm_once_before_exploiting():
    """Warm-up: nenhum braço deve ser escolhido duas vezes antes de todos serem testados."""
    policy = EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=0.0, seed=1)

    seen = []
    for _ in range(len(ARM_IDS)):
        arm = policy.select_arm()
        seen.append(arm)
        policy.update(arm, reward=0.0)

    assert set(seen) == set(ARM_IDS)


def test_epsilon_greedy_exploits_the_best_known_arm_when_epsilon_is_zero():
    policy = EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=0.0, seed=1)
    for arm in ARM_IDS:
        policy.update(arm, reward=0.0)  # sai do warm-up

    policy.update("arm_b", reward=1.0)  # arm_b passa a ter a melhor média

    assert policy.select_arm() == "arm_b"


def test_epsilon_greedy_explores_when_epsilon_is_one():
    policy = EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=1.0, seed=1)
    for arm in ARM_IDS:
        policy.update(arm, reward=0.0)
    policy.update("arm_b", reward=1.0)

    draws = {policy.select_arm() for _ in range(50)}

    assert draws == set(ARM_IDS), "epsilon=1 deve visitar todos os braços, não só o melhor"


def test_epsilon_greedy_update_rejects_an_unknown_arm():
    policy = EpsilonGreedyPolicy(arm_ids=ARM_IDS, seed=1)

    with pytest.raises(ValueError, match="desconhecido"):
        policy.update("arm_fora", reward=1.0)


def test_epsilon_greedy_round_trips_through_dict():
    original = EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=0.2, seed=3)
    original.update("arm_a", 1.0)
    original.update("arm_b", 0.0)

    restored = EpsilonGreedyPolicy.from_dict(original.to_dict())

    assert restored.epsilon == original.epsilon
    assert restored.counts == original.counts
    assert restored.sums == original.sums


def test_epsilon_greedy_reseed_makes_the_exploration_draw_reproducible():
    policy = EpsilonGreedyPolicy(arm_ids=ARM_IDS, epsilon=1.0, seed=1)
    for arm in ARM_IDS:
        policy.update(arm, reward=0.0)  # sai do warm-up

    policy.reseed(7)
    first = policy.select_arm()
    policy.reseed(7)
    second = policy.select_arm()

    assert first == second


# ---------------------------------------------------------------------------
# UCB1Policy
# ---------------------------------------------------------------------------


def test_ucb1_tries_every_arm_once_before_scoring():
    policy = UCB1Policy(arm_ids=ARM_IDS)

    seen = []
    for _ in range(len(ARM_IDS)):
        arm = policy.select_arm()
        seen.append(arm)
        policy.update(arm, reward=0.0)

    assert set(seen) == set(ARM_IDS)


def test_ucb1_favours_the_better_arm_most_of_the_time_after_enough_evidence():
    """
    Uma única atualização não basta: o bônus de exploração cresce com log(t) e nunca some,
    então mesmo um braço claramente pior segue sendo revisitado de vez em quando — esse é
    o preço da garantia teórica do UCB1. O que vale medir é a maioria das escolhas, não a
    última.
    """
    policy = UCB1Policy(arm_ids=ARM_IDS)
    for arm in ARM_IDS:
        policy.update(arm, reward=0.0)

    choices = []
    for _ in range(200):
        arm = policy.select_arm()
        policy.update(arm, reward=1.0 if arm == "arm_c" else 0.0)
        choices.append(arm)

    assert choices.count("arm_c") > 150, "braço com melhor média deveria dominar as escolhas"


def test_ucb1_score_formula_matches_the_documented_bonus():
    """A fórmula é determinística: reproduzimos o cálculo de fora para travar a regressão."""
    policy = UCB1Policy(arm_ids=ARM_IDS)
    for arm in ARM_IDS:
        policy.update(arm, reward=0.0)
    policy.update("arm_a", reward=1.0)

    t = policy._t + 1
    expected = {
        "arm_a": (1.0 / 2) + math.sqrt(2 * math.log(t) / 2),
        "arm_b": 0.0 + math.sqrt(2 * math.log(t) / 1),
        "arm_c": 0.0 + math.sqrt(2 * math.log(t) / 1),
    }

    assert expected["arm_a"] != pytest.approx(expected["arm_b"])
    chosen = policy.select_arm()

    assert chosen == max(expected, key=lambda aid: expected[aid]) == "arm_b"


def test_ucb1_update_rejects_an_unknown_arm():
    policy = UCB1Policy(arm_ids=ARM_IDS)

    with pytest.raises(ValueError, match="desconhecido"):
        policy.update("arm_fora", reward=1.0)


def test_ucb1_round_trips_through_dict_including_the_round_counter():
    original = UCB1Policy(arm_ids=ARM_IDS)
    for arm in ARM_IDS:
        original.update(arm, reward=1.0)

    restored = UCB1Policy.from_dict(original.to_dict())

    assert restored.counts == original.counts
    assert restored.sums == original.sums
    assert restored._t == original._t == len(ARM_IDS)


# ---------------------------------------------------------------------------
# ThompsonSamplingPolicy
# ---------------------------------------------------------------------------


def test_thompson_sampling_starts_with_a_neutral_prior():
    policy = ThompsonSamplingPolicy(arm_ids=ARM_IDS)

    for arm in ARM_IDS:
        assert policy.posterior_mean(arm) == pytest.approx(0.5)


def test_thompson_sampling_update_shifts_the_posterior_mean_toward_the_reward():
    policy = ThompsonSamplingPolicy(arm_ids=ARM_IDS)

    policy.update("arm_a", reward=1.0)

    assert policy.posterior_mean("arm_a") > 0.5
    assert policy.posterior_mean("arm_b") == pytest.approx(0.5)


def test_thompson_sampling_posterior_mean_rejects_an_unknown_arm():
    policy = ThompsonSamplingPolicy(arm_ids=ARM_IDS)

    with pytest.raises(ValueError, match="desconhecido"):
        policy.posterior_mean("arm_fora")


def test_thompson_sampling_update_rejects_an_unknown_arm():
    policy = ThompsonSamplingPolicy(arm_ids=ARM_IDS)

    with pytest.raises(ValueError, match="desconhecido"):
        policy.update("arm_fora", reward=1.0)


def test_thompson_sampling_reseed_makes_the_next_draw_reproducible():
    policy = ThompsonSamplingPolicy(arm_ids=ARM_IDS, seed=1)

    policy.reseed(42)
    first = policy.select_arm()
    policy.reseed(42)
    second = policy.select_arm()

    assert first == second


def test_thompson_sampling_round_trips_through_dict():
    original = ThompsonSamplingPolicy(arm_ids=ARM_IDS, seed=2)
    original.update("arm_a", 1.0)
    original.update("arm_b", 0.0)

    restored = ThompsonSamplingPolicy.from_dict(original.to_dict())

    assert restored.alpha == original.alpha
    assert restored.beta == original.beta
    assert original.to_dict()["algorithm"] == "thompson_sampling"
