"""
Ambiente de simulação usado para treinar e comparar as políticas (Etapa 3).

Reproduz o ambiente do notebook `03_baseline_e_thompson_sampling.ipynb`, mas rodando sobre
as classes de política de `datathon.bandits` — de modo que o que é medido aqui é exatamente
o código que a API serve, e não uma segunda implementação paralela.

O modelo de recompensa vem do catálogo: `base_conversion_rate` do braço multiplicado pelos
`segment_multipliers` de cada segmento que o cliente ativa. Note que os segmentos entram no
*ambiente* (a recompensa), não na *política* — ver README, “Escolhas de design”.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..bandits.base import BanditPolicy

HIGH_SEASON_MONTHS = ["mar", "sep", "oct", "dec"]
MAX_CONVERSION_RATE = 0.95


def derive_segments(row: pd.Series) -> List[str]:
    """
    Segmentos que o cliente ativa. Sobrepõem-se: um cliente pode casar com vários.

    Reproduz deliberadamente as regras do notebook 03 para que os números aqui sejam
    comparáveis aos da Etapa 3. Duas divergências conhecidas em relação ao
    `offer_catalog.json`, que pertencem à Etapa 3 e devem ser resolvidas com quem a
    entregou — não corrigidas aqui em silêncio, sob pena de os números deixarem de bater:

    - `admin_worker` (`job == 'admin.'`) está no catálogo com multiplicadores em todos os
      braços, mas o notebook nunca o ativa. Na prática esses multiplicadores são inertes.
    - `low_engagement` é derivado aqui das flags `had_previous_contact`/`contacted_before`;
      o catálogo enuncia a regra como `previous == 0 AND pdays == 999`. Mesmo conceito,
      vocabulários diferentes.
    """
    segs = []
    if row["poutcome"] == "success":
        segs.append("previous_converter")
    if row["month"] in HIGH_SEASON_MONTHS:
        segs.append("high_season")
    if row["job"] == "student" and row["contact"] == "cellular":
        segs.append("student_digital")
    if row["job"] == "student":
        segs.append("student_saver")
    if row["job"] == "retired":
        segs.append("retired")
    if row["education"] == "university.degree":
        segs.append("university_educated")
    if row["contact"] == "cellular":
        segs.append("digital_channel")
    if row["housing"] == "no" and row["loan"] == "no" and row.get("default", "no") == "no":
        segs.append("debt_free")
    if row["had_previous_contact"] == 0 and row["contacted_before"] == 0:
        segs.append("low_engagement")
    return segs


class OfferEnvironment:
    """
    Matriz contrafactual de desfechos: para cada cliente, o resultado que *teria* acontecido
    em cada braço. Usa common random numbers — o mesmo ruído para todos os braços — para que
    a comparação entre políticas não seja contaminada por sorte diferente.
    """

    def __init__(self, clients: pd.DataFrame, catalog: Dict[str, Any], seed: int):
        self.arm_ids = [a["arm_id"] for a in catalog["arms"]]
        base_rate = {a["arm_id"]: a["base_conversion_rate"] for a in catalog["arms"]}
        seg_mult = {a["arm_id"]: a["segment_multipliers"] for a in catalog["arms"]}

        segments = [derive_segments(row) for _, row in clients.iterrows()]
        self.contexts = clients.to_dict(orient="records")
        n_clients, n_arms = len(clients), len(self.arm_ids)

        probabilities = np.zeros((n_clients, n_arms))
        for j, arm_id in enumerate(self.arm_ids):
            for i, segs in enumerate(segments):
                multiplier = 1.0
                for seg in segs:
                    if seg in seg_mult[arm_id]:
                        multiplier *= seg_mult[arm_id][seg]
                probabilities[i, j] = min(base_rate[arm_id] * multiplier, MAX_CONVERSION_RATE)

        noise = np.random.RandomState(seed).random_sample((n_clients, n_arms))
        self.outcomes = (noise < probabilities).astype(int)
        self._arm_index = {arm_id: j for j, arm_id in enumerate(self.arm_ids)}

    def __len__(self) -> int:
        return len(self.outcomes)

    def reward(self, client_index: int, arm_id: str) -> int:
        """Recompensa observada (0/1) se este cliente receber este braço."""
        return int(self.outcomes[client_index, self._arm_index[arm_id]])


def run_policy(policy: BanditPolicy, environment: OfferEnvironment) -> Dict[str, Any]:
    """
    Roda a política sobre todo o horizonte, atualizando-a a cada recompensa observada.

    Devolve a taxa de conversão final e a distribuição de braços na segunda metade do
    horizonte (para qual braço a política convergiu depois de aprender).
    """
    rewards = np.zeros(len(environment))
    choices = []

    for i in range(len(environment)):
        arm_id = policy.select_arm(environment.contexts[i])
        reward = environment.reward(i, arm_id)
        policy.update(arm_id, reward)
        rewards[i] = reward
        choices.append(arm_id)

    second_half = choices[len(choices) // 2 :]
    share = {
        arm_id: second_half.count(arm_id) / len(second_half)
        for arm_id in sorted(set(second_half))
    }
    return {
        "conversion_rate": float(rewards.mean()),
        "converged_arm": max(share, key=lambda arm_id: share[arm_id]),
        "arm_share_second_half": share,
    }
