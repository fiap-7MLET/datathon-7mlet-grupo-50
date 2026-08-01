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
