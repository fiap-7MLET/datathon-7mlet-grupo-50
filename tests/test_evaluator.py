"""
Testes das métricas da Etapa 4.

`summarize()` é o coração da avaliação e é uma função pura: recebe o resultado cru das
simulações e devolve as métricas comparáveis. Os testes exercitam essa aritmética com
números inventados — sem dataset, sem MLflow — para que uma quebra aponte para a métrica e
não para o ambiente.
"""

import math

import pytest

from datathon.evaluation.evaluator import PolicyMetrics, summarize, to_markdown


def result(rate, share):
    """Resultado de uma política no formato que `run_policy` devolve."""
    return {
        "conversion_rate": rate,
        "converged_arm": max(share, key=lambda arm: share[arm]),
        "arm_share_second_half": share,
    }


def results(**rates):
    """Cenário completo: baselines obrigatórios + as políticas passadas por nome."""
    base = {
        "baseline_fixo": result(0.20, {"arm_a": 1.0}),
        "baseline_aleatorio": result(0.10, {"arm_a": 0.5, "arm_b": 0.5}),
    }
    base.update({name: result(rate, {"arm_a": 0.9, "arm_b": 0.1}) for name, rate in rates.items()})
    return base


def by_policy(metrics):
    return {m.policy: m for m in metrics}


def test_uplift_is_measured_against_both_baselines():
    """0,25 contra um baseline fixo de 0,20 é +25%; contra o aleatório de 0,10, +150%."""
    metrics = by_policy(summarize(results(thompson_sampling=0.25)))

    thompson = metrics["thompson_sampling"]
    assert thompson.uplift_vs_fixed_pct == pytest.approx(25.0)
    assert thompson.uplift_vs_random_pct == pytest.approx(150.0)


def test_a_policy_worse_than_the_baseline_gets_a_negative_uplift():
    """Uplift negativo precisa aparecer como negativo — é o sinal de que o bandit não pagou."""
    metrics = by_policy(summarize(results(politica_ruim=0.15)))

    assert metrics["politica_ruim"].uplift_vs_fixed_pct == pytest.approx(-25.0)


def test_policies_are_ranked_by_conversion():
    """A ordem do relatório é a da conversão — a melhor política em primeiro."""
    metrics = summarize(results(thompson_sampling=0.25, ucb1=0.30))

    assert [m.policy for m in metrics][:2] == ["ucb1", "thompson_sampling"]


def test_convergence_reports_the_dominant_arm_and_how_concentrated_it_is():
    """
    A convergência é a segunda pergunta da Etapa 4: para onde a política migrou, e quanto
    ainda explora. Concentração perto de 1,0 significa exploração praticamente encerrada.
    """
    scenario = results()
    scenario["thompson_sampling"] = result(0.25, {"arm_a": 0.97, "arm_b": 0.03})

    thompson = by_policy(summarize(scenario))["thompson_sampling"]

    assert thompson.converged_arm == "arm_a"
    assert thompson.converged_arm_share == pytest.approx(0.97)
    assert thompson.arms_explored_second_half == 2


def test_a_zero_baseline_gives_undefined_uplift_instead_of_dividing_by_zero():
    """Baseline que nunca converte não define ganho percentual; NaN é honesto, crash não."""
    scenario = results(thompson_sampling=0.25)
    scenario["baseline_fixo"] = result(0.0, {"arm_a": 1.0})

    thompson = by_policy(summarize(scenario))["thompson_sampling"]

    assert math.isnan(thompson.uplift_vs_fixed_pct)


def test_evaluation_without_a_baseline_is_refused():
    """Sem baseline não há uplift — falhar aqui é melhor que publicar um relatório sem régua."""
    with pytest.raises(ValueError, match="baseline_fixo"):
        summarize({"thompson_sampling": result(0.25, {"arm_a": 1.0})})


def test_report_shows_each_policy_with_its_uplift():
    """O relatório é entregável da Etapa 4: cada política precisa aparecer com o uplift."""
    metrics = [
        PolicyMetrics(
            policy="thompson_sampling",
            conversion_rate=0.4128,
            uplift_vs_fixed_pct=24.5,
            uplift_vs_random_pct=123.1,
            converged_arm="arm_005",
            converged_arm_share=0.99,
            arms_explored_second_half=2,
        )
    ]
    meta = {
        "n_clients": 20000,
        "sampling_seed": 42,
        "outcome_seed": 2026,
        "catalog_version": "1.0.0",
    }

    report = to_markdown(metrics, meta)

    assert "thompson_sampling" in report
    assert "+24.5%" in report
    assert "arm_005" in report
