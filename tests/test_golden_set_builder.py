"""
Testes de `policy_fingerprint` e `build_golden_set` (`datathon.evaluation.evaluator`).

`test_golden_set.py` verifica os arquivos golden já gravados contra a política publicada;
aqui testamos a função que os gera, com uma política e um catálogo sintéticos — sem
depender do dataset real nem dos cinco `CLIENT_PERSONAS` fixos.

`evaluate()`, `train_snapshot_policy()` e `main()` não são exercitados: dependem de
`data/processed/bank_marketing_processed.parquet` (não versionado — gerado por
`datathon.data_loader` a partir do CSV do Kaggle) e, no caso de `main()`, também de um
MLflow tracking server. Ver nota no relatório de cobertura.
"""

import json

from datathon.evaluation.evaluator import _write_golden, build_golden_set, policy_fingerprint

CATALOG = {
    "arms": [
        {"arm_id": "arm_a", "name": "Oferta A", "channel": "app"},
        {"arm_id": "arm_b", "name": "Oferta B", "channel": "email"},
    ]
}


def thompson_state(alpha, beta):
    return {
        "algorithm": "thompson_sampling",
        "arm_ids": ["arm_a", "arm_b"],
        "alpha": alpha,
        "beta": beta,
    }


def test_policy_fingerprint_is_stable_for_the_same_belief():
    state = thompson_state(alpha=[10.0, 1.0], beta=[1.0, 10.0])

    assert policy_fingerprint(state) == policy_fingerprint(state)


def test_policy_fingerprint_changes_when_the_belief_changes():
    a = thompson_state(alpha=[10.0, 1.0], beta=[1.0, 10.0])
    b = thompson_state(alpha=[11.0, 1.0], beta=[1.0, 10.0])

    assert policy_fingerprint(a) != policy_fingerprint(b)


def test_policy_fingerprint_ignores_training_metadata():
    """
    Retreinar com o mesmo resultado (mesmos alpha/beta) não deve invalidar o golden set —
    só `FINGERPRINT_FIELDS` (a crença) entra na impressão digital, não metadados de treino.
    """
    base = thompson_state(alpha=[10.0, 1.0], beta=[1.0, 10.0])
    with_metadata = {**base, "trained_on_n_clients": 999, "random_seed": 42}

    assert policy_fingerprint(base) == policy_fingerprint(with_metadata)


def test_build_golden_set_produces_one_case_per_client_persona():
    golden = build_golden_set(thompson_state(alpha=[500.0, 1.0], beta=[1.0, 500.0]), CATALOG)

    assert len(golden["cases"]) == 5
    assert golden["algorithm"] == "thompson_sampling"
    assert golden["contextual"] is False
    for case in golden["cases"]:
        assert case["expected"]["arm_id"] in {"arm_a", "arm_b"}


def test_build_golden_set_freezes_a_reproducible_recommendation_per_case():
    policy_state = thompson_state(alpha=[500.0, 1.0], beta=[1.0, 500.0])

    first = build_golden_set(policy_state, CATALOG)
    second = build_golden_set(policy_state, CATALOG)

    assert [c["expected"]["arm_id"] for c in first["cases"]] == [
        c["expected"]["arm_id"] for c in second["cases"]
    ]


def test_build_golden_set_only_embeds_the_policy_state_when_asked():
    policy_state = thompson_state(alpha=[500.0, 1.0], beta=[1.0, 500.0])

    without = build_golden_set(policy_state, CATALOG, embed_policy_state=False)
    with_embed = build_golden_set(policy_state, CATALOG, embed_policy_state=True)

    assert "policy_state" not in without
    assert with_embed["policy_state"] == policy_state


def test_build_golden_set_carries_a_custom_note():
    golden = build_golden_set(
        thompson_state(alpha=[1.0, 1.0], beta=[1.0, 1.0]), CATALOG, nota="nota customizada"
    )

    assert golden["nota"] == "nota customizada"


def test_write_golden_persists_the_json_and_creates_parent_directories(tmp_path):
    golden = build_golden_set(thompson_state(alpha=[500.0, 1.0], beta=[1.0, 500.0]), CATALOG)
    target = tmp_path / "nested" / "golden.json"

    _write_golden(target, golden)

    assert json.loads(target.read_text(encoding="utf-8")) == golden
