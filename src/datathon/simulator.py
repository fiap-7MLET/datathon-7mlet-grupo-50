import json
import random
from datetime import datetime, timedelta
import os

BASE_SEED = 42
RANDOM_STATE = random.Random(BASE_SEED)

N_EVENTS = 200
START_DATE = datetime(2026, 3, 1, 8, 0, 0)

CATALOG_PATH = os.path.join(os.getcwd(), "data", "synthetic_enrichment", "offer_catalog.json")

with open(CATALOG_PATH, encoding="utf-8") as f:
    catalog = json.load(f)

# Configuração dos braços
ARMS = {
    arm["arm_id"]: {
        "base_rate": arm["base_conversion_rate"],
        "delay":     arm["reward_delay_days"],
        "channel":   arm["channel"],
    }
    for arm in catalog["arms"]
}

# Multiplicadores por segmento por braço
SEG_MULTIPLIERS = {
    arm["arm_id"]: arm["segment_multipliers"]
    for arm in catalog["arms"]
}

JOBS = ["admin.","blue-collar","entrepreneur","housemaid","management",
        "retired","self-employed","services","student","technician","unemployed"]
EDUCATIONS = ["basic.4y","basic.6y","basic.9y","high.school",
              "professional.course","university.degree"]
MONTHS_HIGH = ["mar","sep","oct","dec"]
MONTHS_LOW  = ["jan","feb","apr","may","jun","jul","aug","nov"]
CONTACTS    = ["cellular","telephone"]
POUTCOMES   = ["success","failure","nonexistent","nonexistent","nonexistent"] # replicando a distribuição do dataset original
DAYS        = ["monday","tuesday","wednesday","thursday","friday"]

def derive_segments(ctx):
    segs = []
    if ctx["poutcome"] == "success":
        segs.append("previous_converter")
    if ctx["month"] in MONTHS_HIGH:
        segs.append("high_season")
    if ctx["job"] == "student" and ctx["contact_type"] == "cellular":
        segs.append("student_digital")
    if ctx["job"] == "student":
        segs.append("student_saver")
    if ctx["job"] == "retired":
        segs.append("retired")
    if ctx["education"] == "university.degree":
        segs.append("university_educated")
    if ctx["contact_type"] == "cellular":
        segs.append("digital_channel")
    if not ctx["has_housing_loan"] and not ctx["has_personal_loan"]:
        segs.append("debt_free")
    if ctx["previous_campaigns"] == 0 and ctx["days_since_last_contact"] == 999:
        segs.append("low_engagement")
    return segs

def compute_expected_reward(arm_id, segments):
    base = ARMS[arm_id]["base_rate"]
    mults = SEG_MULTIPLIERS.get(arm_id, {})
    multiplier = 1.0
    for seg in segments:
        if seg in mults:
            multiplier *= mults[seg]
    # cap at 0.95
    return min(round(base * multiplier, 4), 0.95)

def thompson_sample(arm_id, alpha_beta):
    """Retorna sample de Beta(alpha, beta) para cada braço."""
    samples = {}
    for aid, (a, b) in alpha_beta.items():
        samples[aid] = RANDOM_STATE.betavariate(a, b)
    return samples

def generate_context(rng):
    job  = rng.choice(JOBS)
    edu  = rng.choice(EDUCATIONS)
    age  = rng.choice(["18-24","25-34","35-44","45-54","55-64","65+"])
    bal  = rng.choice(["negative","low","medium","high","very_high"])
    housing = rng.random() < 0.5
    loan    = rng.random() < 0.2
    contact = rng.choice(CONTACTS)
    prev    = rng.choice([0, 0, 0, 1, 2, 3])
    pdays   = 0 if prev == 0 else rng.randint(1, 365)
    pout    = rng.choice(POUTCOMES)
    month   = rng.choice(MONTHS_HIGH + MONTHS_LOW)
    dow     = rng.choice(DAYS)
    return {
        "job": job,
        "education": edu,
        "age_bucket": age,
        "balance_bucket": bal,
        "has_housing_loan": housing,
        "has_personal_loan": loan,
        "contact_type": contact,
        "days_since_last_contact": pdays,
        "previous_campaigns": prev,
        "poutcome": pout,
        "month": month,
        "day_of_week": dow,
    }

def main():
    # Thompson Sampling state: Beta(alpha, beta) por braço
    alpha_beta = {aid: [1.0, 1.0] for aid in ARMS}

    offer_events   = []
    delayed_rewards = []

    current_time = START_DATE
    client_counter = 1000

    for i in range(N_EVENTS):
        event_seed = BASE_SEED + i
        event_rng  = random.Random(event_seed)

        # Avança tempo: entre 30 min e 8h do evento anterior
        current_time += timedelta(minutes=event_rng.randint(30, 480))

        client_id = f"client_{client_counter:05d}"
        client_counter += 1

        ctx      = generate_context(event_rng)
        segments = derive_segments(ctx)

        # Thompson Sampling: amostrar Beta para cada braço
        ts_rng = random.Random(event_seed + 10000)
        arm_samples = {}
        for aid in ARMS:
            a, b = alpha_beta[aid]
            arm_samples[aid] = ts_rng.betavariate(a, b)

        # Selecionar braço com maior sample
        selected_arm = max(arm_samples, key=lambda x: arm_samples[x])

        # Probabilidades normalizadas (para documentação)
        total = sum(arm_samples.values())
        arm_probs = {aid: round(v/total, 4) for aid, v in arm_samples.items()}

        expected_reward = compute_expected_reward(selected_arm, segments)

        # Exploração forçada? (epsilon=0.05)
        exploration_flag = event_rng.random() < 0.05
        if exploration_flag:
            selected_arm = event_rng.choice(list(ARMS.keys()))

        event_id = f"evt_{i+1:06d}"
        arm_cfg  = ARMS[selected_arm]
        delay_days = arm_cfg["delay"]

        # Gerar reward com probabilidade = expected_reward
        will_convert = event_rng.random() < expected_reward
        reward_ts    = current_time + timedelta(days=delay_days) if delay_days > 0 else current_time
        reward_within_window = True  # sintético: sempre dentro da janela

        if will_convert:
            reward_observed = 1.0
            status = "converted"
        else:
            reward_observed = 0.0
            status = "not_converted" if delay_days == 0 else "expired"

        # Atualizar Thompson Sampling
        alpha_beta[selected_arm][0] += reward_observed
        alpha_beta[selected_arm][1] += (1 - reward_observed)

        # Offer event
        offer_event = {
            "event_id": event_id,
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "client_id": client_id,
            "arm_id": selected_arm,
            "channel": arm_cfg["channel"],
            "context": ctx,
            "derived_segments": segments,
            "policy_version": "v2.0.0",
            "selection_algorithm": "thompson_sampling",
            "arm_probabilities": arm_probs,
            "exploration_flag": exploration_flag,
            "random_seed": event_seed,
            "reward_expected": expected_reward,
            "reward_observed": reward_observed,
            "reward_timestamp": reward_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
        }
        offer_events.append(offer_event)

        # Delayed reward (sempre gerado, inclusive para não-conversão)
        delayed_reward = {
            "event_id": event_id,
            "client_id": client_id,
            "arm_id": selected_arm,
            "reward_observed": int(reward_observed),
            "reward_value": reward_observed,
            "reward_timestamp": reward_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "delay_days_expected": delay_days,
            "delay_days_actual": delay_days,
            "reward_within_window": reward_within_window,
            "conversion_type": "subscription" if reward_observed == 1.0 else "no_conversion",
            "segments_active": segments,
            "expected_reward_at_decision": expected_reward,
            "reward_delta": round(reward_observed - expected_reward, 4),
            "notes": (
                f"conversão em {delay_days}d via {arm_cfg['channel']}"
                if reward_observed == 1.0
                else f"sem conversão após {delay_days}d de espera"
            ),
        }
        delayed_rewards.append(delayed_reward)

    # Estatísticas
    n_converted = sum(1 for d in delayed_rewards if d["reward_observed"] == 1)
    print(f"Eventos gerados: {N_EVENTS}")
    print(f"Conversões: {n_converted} ({100*n_converted/N_EVENTS:.1f}%)")
    arm_counts = {}
    for e in offer_events:
        arm_counts[e["arm_id"]] = arm_counts.get(e["arm_id"], 0) + 1
    print("Distribuição de braços:")
    for arm, count in sorted(arm_counts.items()):
        print(f"  {arm}: {count} ({100*count/N_EVENTS:.1f}%)")

    return offer_events, delayed_rewards

if __name__ == "__main__":
    offer_events, delayed_rewards = main()

    with open("data/synthetic_enrichment/offer_events.json", "w") as f:
        for e in offer_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open("data/synthetic_enrichment/delayed_rewards.json", "w") as f:
        for r in delayed_rewards:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nArquivos gerados: offer_events.json, delayed_rewards.json")