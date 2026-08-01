# Etapa 4 — Avaliação das políticas

Horizonte simulado: **20.000 clientes**
(amostragem seed=42, desfechos seed=2026, catálogo v1.0.0).

Todas as políticas enfrentam a **mesma** matriz de desfechos (*common random numbers*), então a diferença entre elas é decisão, não sorte.

| política | conversão | uplift vs fixo | uplift vs aleatório | convergiu para | concentração | braços na 2ª metade |
|---|---:|---:|---:|---|---:|---:|
| contextual_thompson_sampling | 0.4139 | +24.8% | +123.7% | arm_005 | 85.4% | 9 |
| thompson_sampling | 0.4128 | +24.5% | +123.0% | arm_005 | 99.5% | 9 |
| ucb1 | 0.3922 | +18.3% | +111.9% | arm_005 | 92.5% | 9 |
| epsilon_greedy | 0.3913 | +18.0% | +111.5% | arm_005 | 91.5% | 9 |
| baseline_fixo | 0.3317 | +0.0% | +79.2% | arm_008 | 100.0% | 1 |
| baseline_aleatorio | 0.1850 | -44.2% | +0.0% | arm_006 | 11.8% | 9 |

Baseline de referência do uplift: `baseline_fixo` — sempre a melhor oferta do catálogo, o que se faria sem bandit.

Leitura da concentração: quanto mais perto de 100%, menos a política ainda explora. Concentração alta no fim do horizonte é o comportamento esperado do Thompson Sampling — a exploração decai à medida que a evidência se acumula.

> Gerado por `uv run datathon-evaluate`. Não editar à mão.
