from datathon.bandits.contextual_thompson import (
    ContextualThompsonSamplingPolicy,
    context_segment,
)


def test_context_segment_uses_a_stable_priority():
    assert context_segment({"poutcome": "success", "job": "student"}) == "previous_converter"
    assert context_segment({"job": "student", "contact": "cellular"}) == "student_digital"
    assert context_segment({"job": "retired", "previous": 0}) == "retired"


def test_different_segments_can_favour_different_offers():
    state = {
        "algorithm": "contextual_thompson_sampling",
        "arm_ids": ["a", "b"],
        "posteriors": {
            segment: {"a": [1.0, 1.0], "b": [1.0, 1.0]}
            for segment in (
                "previous_converter", "student_digital", "retired",
                "low_engagement", "digital_channel", "general",
            )
        },
    }
    state["posteriors"]["student_digital"] = {"a": [500.0, 1.0], "b": [1.0, 500.0]}
    state["posteriors"]["retired"] = {"a": [1.0, 500.0], "b": [500.0, 1.0]}
    policy = ContextualThompsonSamplingPolicy.from_dict(state)

    assert policy.select_arm({"job": "student", "contact": "cellular"}) == "a"
    assert policy.select_arm({"job": "retired"}) == "b"


def test_serialization_preserves_the_contextual_posteriors():
    policy = ContextualThompsonSamplingPolicy(["a", "b"], seed=1)
    arm = policy.select_arm({"job": "retired"})
    policy.update(arm, 1)

    restored = ContextualThompsonSamplingPolicy.from_dict(policy.to_dict())

    assert restored.to_dict() == policy.to_dict()


def test_update_without_context_falls_back_to_the_last_select_arm_call():
    """Contrato usado pelo loop de simulação sequencial (`training/simulation.py`)."""
    policy = ContextualThompsonSamplingPolicy(["a", "b"], seed=1)
    policy.select_arm({"job": "retired"})

    policy.update("a", 1)

    assert policy.posteriors["retired"]["a"] == [2.0, 1.0]
    assert policy.posteriors["general"]["a"] == [1.0, 1.0]


def test_update_with_context_targets_that_segment_even_if_another_was_selected_since():
    """
    Reproduz o bug corrigido: um `select_arm` de outro segmento no meio-tempo (ex: outra
    requisição concorrente) não pode desviar onde o feedback é aplicado quando quem chama
    `update` passa o `context` de onde a recomendação original veio.
    """
    policy = ContextualThompsonSamplingPolicy(["a", "b"], seed=1)
    policy.select_arm({"job": "retired"})  # define _last_segment = "retired"
    policy.select_arm({"job": "student", "contact": "cellular"})  # sobrescreve para "student_digital"

    policy.update("a", 1, context={"job": "retired"})

    assert policy.posteriors["retired"]["a"] == [2.0, 1.0]
    assert policy.posteriors["student_digital"]["a"] == [1.0, 1.0]
