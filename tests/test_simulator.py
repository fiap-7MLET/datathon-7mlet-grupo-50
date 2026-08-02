"""
Testes de `datathon.simulator` — gerador de eventos sintéticos usado para popular o
ambiente de demonstração/enriquecimento (`offer_events.json` / `delayed_rewards.json`).

O módulo lê `data/synthetic_enrichment/offer_catalog.json` de `os.getcwd()` no
import — funciona aqui porque os testes rodam a partir da raiz do repositório
(`pyproject.toml` define `testpaths = ["tests"]` sem alterar o cwd), onde o arquivo já
existe. `main()` só *escreve* arquivos dentro do bloco `if __name__ == "__main__":`, então
chamá-la em teste é seguro — não tem efeito colateral em disco.
"""

from datathon import simulator


def test_arms_and_segment_multipliers_are_loaded_from_the_real_catalog():
    assert simulator.ARMS, "catálogo deveria ter pelo menos um braço"
    assert set(simulator.SEG_MULTIPLIERS) == set(simulator.ARMS)


def test_derive_segments_activates_previous_converter_and_high_season():
    ctx = {
        "poutcome": "success",
        "month": "mar",
        "job": "management",
        "contact_type": "telephone",
        "education": "basic.4y",
        "has_housing_loan": True,
        "has_personal_loan": True,
        "previous_campaigns": 1,
        "days_since_last_contact": 5,
    }

    segments = simulator.derive_segments(ctx)

    assert "previous_converter" in segments
    assert "high_season" in segments
    assert "debt_free" not in segments  # tem os dois empréstimos


def test_derive_segments_flags_low_engagement_when_never_contacted_before():
    ctx = {
        "poutcome": "nonexistent", "month": "jan", "job": "admin.",
        "contact_type": "telephone", "education": "basic.4y",
        "has_housing_loan": True, "has_personal_loan": True,
        "previous_campaigns": 0, "days_since_last_contact": 999,
    }

    assert "low_engagement" in simulator.derive_segments(ctx)


def test_derive_segments_flags_retired_clients():
    ctx = {
        "poutcome": "nonexistent", "month": "jan", "job": "retired",
        "contact_type": "telephone", "education": "basic.4y",
        "has_housing_loan": True, "has_personal_loan": True,
        "previous_campaigns": 1, "days_since_last_contact": 5,
    }

    assert "retired" in simulator.derive_segments(ctx)


def test_derive_segments_flags_student_digital_only_with_cellular_contact():
    student_cellular = {
        "poutcome": "nonexistent", "month": "jan", "job": "student",
        "contact_type": "cellular", "education": "basic.4y",
        "has_housing_loan": False, "has_personal_loan": False,
        "previous_campaigns": 1, "days_since_last_contact": 5,
    }
    student_phone = {**student_cellular, "contact_type": "telephone"}

    assert "student_digital" in simulator.derive_segments(student_cellular)
    assert "student_saver" in simulator.derive_segments(student_cellular)
    assert "student_digital" not in simulator.derive_segments(student_phone)
    assert "student_saver" in simulator.derive_segments(student_phone)


def test_compute_expected_reward_applies_segment_multipliers_and_caps_at_095():
    arm_id = next(iter(simulator.ARMS))
    base = simulator.ARMS[arm_id]["base_rate"]

    # Sem segmentos, a recompensa esperada é exatamente a base do catálogo.
    assert simulator.compute_expected_reward(arm_id, []) == round(base, 4)

    # Empilhar segmentos que multiplicam a taxa não pode nunca passar de 0.95.
    all_segments = list(simulator.SEG_MULTIPLIERS.get(arm_id, {}))
    boosted = simulator.compute_expected_reward(arm_id, all_segments)
    assert boosted <= 0.95


def test_thompson_sample_returns_a_value_per_arm_in_zero_one():
    alpha_beta = {aid: (1.0, 1.0) for aid in simulator.ARMS}

    samples = simulator.thompson_sample(next(iter(simulator.ARMS)), alpha_beta)

    assert set(samples) == set(simulator.ARMS)
    assert all(0.0 <= v <= 1.0 for v in samples.values())


def test_generate_context_returns_every_documented_field():
    import random

    ctx = simulator.generate_context(random.Random(1))

    expected_keys = {
        "job", "education", "age_bucket", "balance_bucket", "has_housing_loan",
        "has_personal_loan", "contact_type", "days_since_last_contact",
        "previous_campaigns", "poutcome", "month", "day_of_week",
    }
    assert set(ctx) == expected_keys
    assert ctx["job"] in simulator.JOBS
    assert ctx["contact_type"] in simulator.CONTACTS


def test_main_generates_the_configured_number_of_events_and_delayed_rewards():
    offer_events, delayed_rewards = simulator.main()

    assert len(offer_events) == simulator.N_EVENTS
    assert len(delayed_rewards) == simulator.N_EVENTS
    assert {e["arm_id"] for e in offer_events} <= set(simulator.ARMS)


def test_main_pairs_each_offer_event_with_its_delayed_reward_by_event_id():
    offer_events, delayed_rewards = simulator.main()

    offer_ids = [e["event_id"] for e in offer_events]
    reward_ids = [r["event_id"] for r in delayed_rewards]

    assert offer_ids == reward_ids


def test_main_is_reproducible_given_the_fixed_base_seed():
    """O gerador usa `BASE_SEED` fixo — duas chamadas de `main()` devem coincidir."""
    first, _ = simulator.main()
    second, _ = simulator.main()

    assert [e["arm_id"] for e in first] == [e["arm_id"] for e in second]
    assert [e["reward_observed"] for e in first] == [e["reward_observed"] for e in second]
