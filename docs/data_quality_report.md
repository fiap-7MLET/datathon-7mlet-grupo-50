# Bank Marketing — Relatório de Qualidade & Decisões de Pré-Processamento

> **Dataset:** `bank-additional-full.csv`  
> **Shape raw:** 41.188 linhas × 21 colunas  
> **Variável target:** `y` — adesão a depósito a prazo (yes/no)  
> **Separador CSV:** `;`  
> **Pipeline:** `BankMarketingPreprocessor` (`src/01_data_loader.py`)
> **EDA:** `notebooks/01_eda.ipynb`

---

## 1. Relatório de Qualidade dos Dados

### 1.A Missing Values

O dataset codifica valores ausentes com a string `"unknown"` em vez de `NaN`. Substituindo `"unknown"` por `NaN`, o percentual real de nulos por coluna é:

| Coluna | % Missing (unknown) | Nível | Tratamento aplicado |
|--------|---------------------|-------|---------------------|
| `default` | 20,9% | Alto | Mantido como categoria `"unknown"` |
| `education` | 4,2% | Baixo | Mantido como categoria `"unknown"` |
| `housing` | 2,4% | Baixo | Mantido como categoria `"unknown"` |
| `loan` | 2,4% | Baixo | Mantido como categoria `"unknown"` |
| `job` | 0,8% | Muito baixo | Mantido como categoria `"unknown"` |
| `marital` | 0,2% | Muito baixo | Mantido como categoria `"unknown"` |

**Decisão:** Para variáveis categóricas, `"unknown"` é mantido como categoria explícita (`fillna("unknown")`) em vez de descarte — a ausência pode ser informativa para o modelo. Para numéricas, imputação pela mediana. A coluna `default` merece atenção especial por ter 20,9% de ausência — foi mantida, mas é candidata a remoção caso gere ruído.

---

### 1.B Registros Duplicados

| Métrica | Valor |
|---------|-------|
| Total de linhas (raw) | 41.188 |
| Linhas duplicadas encontradas | **12 (0,03%)** |
| Linhas após remoção | 41.176 |

**Decisão:** `drop_duplicates()` sem impacto relevante na base (<0,05%).

---

### 1.C Outliers e Distribuições

Contagem de outliers via IQR (limite: Q1 − 1,5×IQR | Q3 + 1,5×IQR):

| Coluna | Outliers (IQR) | Severidade | Decisão |
|--------|---------------|------------|---------|
| `previous` | 5.625 | Alta | Sem tratamento — valores reais (max=7 contatos) |
| `duration` | 2.963 | Alta | Coluna removida (data leakage — ver 1.D) |
| `campaign` | 2.406 | Alta | Cap no percentil 99 (`clip`) |
| `pdays` | 1.515 | Aparente | Valor 999 = código "nunca contactado"; transformado em flag binária |
| `age` | 469 | Baixa | Sem tratamento — distribuição aceitável (cauda direita natural) |
| `cons.conf.idx` | 447 | Macro | Sem tratamento — outliers refletem ciclos econômicos reais |

> **Nota sobre variáveis macroeconômicas:** `emp.var.rate`, `cons.price.idx`, `euribor3m` e `nr.employed` não apresentaram outliers IQR. Suas distribuições multimodais são estrutura real dos dados (snapshots de períodos econômicos distintos), não ruído.

---

### 1.D Detecção de Data Leakage

| Coluna | Correlação com y | Diagnóstico |
|--------|-----------------|-------------|
| `duration` | 0,405 |  **LEAKAGE CONFIRMADO** — só é conhecida após a ligação terminar. Separação clara no boxplot yes vs no. **Removida.** |
| `pdays` | -0,325 |  Possível viés de processo — convertida em flag binária (`contacted_before`) |
| `previous` | 0,230 |  Atenção — viés de sobrevivência: quem foi mais contatado foi selecionado por resposta anterior |
| `nr.employed` | -0,355 |  Multicolinearidade severa (r=0,95 com `euribor3m`). **Removida.** |
| `emp.var.rate` | -0,298 |  Multicolinearidade severa (r=0,97 com `euribor3m`). **Removida.** |

---

## 2. Decisões de Pré-Processamento

### Decisão 1 — Remoção de Colunas

```python
cols_to_drop = ["duration", "pdays", "emp.var.rate", "nr.employed"]
df_clean = df_clean.drop(columns=cols_to_drop)
```

- **`duration`:** data leakage — duração da ligação só é conhecida após o contato terminar. Correlação 0,40 com target, confirmada por boxplot (mediana de `yes` é 2–3× maior que `no`).
- **`pdays`:** convertida em flag binária. O valor 999 não é outlier real; é código para "nunca contactado" (~96% dos registros). Usar o número bruto distorceria modelos lineares.
- **`emp.var.rate`:** correlação 0,97 com `euribor3m` — redundância completa. `euribor3m` é mantido por ser mais granular e interpretável.
- **`nr.employed`:** correlação 0,95 com `euribor3m` — mesma lógica. Variável agregada de mercado de trabalho já capturada pelo juro.

---

### Decisão 2 — Feature Engineering: Flags de Contato Anterior

```python
df_clean['contacted_before']    = (df_clean['pdays'] != 999).astype(int)
df_clean['had_previous_contact'] = (df_clean['previous'] > 0).astype(int)
```

| Nova coluna | Lógica |
|-------------|--------|
| `contacted_before` | 1 se o cliente já foi contactado em campanha anterior, 0 se nunca |
| `had_previous_contact` | 1 se há registro de contato anterior (`previous > 0`) |

As duas colunas de origem foram transformadas em flags binárias para eliminar a distorção do valor especial 999 e evitar que o modelo trate a escala numérica de `pdays` como contínua. A coluna `previous` original é mantida junto da flag para preservar a informação de intensidade.

---

### Decisão 3 — Tratamento de Outliers em `campaign`

```python
df_clean['campaign'] = df_clean['campaign'].clip(upper=df_clean['campaign'].quantile(0.99))
```

| Parâmetro | Valor |
|-----------|-------|
| Estratégia | Winsorização (`clip`) |
| Limite superior | Percentil 99 da distribuição de `campaign` |

A maioria dos clientes recebe 1–3 ligações. Valores acima de ~10 são anômalos e provavelmente ineficazes (retornos decrescentes). Valores extremos como 40–56 contatos distorcem média e gradientes em modelos lineares. A winsorização pelo P99 preserva os registros (sem exclusão) e trunca apenas os valores mais extremos, mantendo a informação relativa de "muitos contatos".

---

### Decisão 4 — Discretização de `age` em `age_group`

```python
df_clean['age_group'] = pd.cut(df_clean['age'],
                    bins=[0, 30, 40, 50, 60, 100],
                    labels=['<30','30-40','40-50','50-60','60+'])
```

| Faixa | Racional |
|-------|----------|
| `<30` | Jovens adultos — perfil estudante/recém-formado |
| `30–40` | Adultos jovens — carreiras em formação |
| `40–50` | Meia-idade — pico de renda e acumulação |
| `50–60` | Pré-aposentadoria — maior aversão a risco |
| `60+` | Aposentados/seniores — retirados do mercado de trabalho |

A discretização captura relações não-lineares entre idade e propensão à conversão que modelos lineares não detectam facilmente. A coluna numérica `age` original é **mantida** — `age_group` é feature adicional, não substituta.

---

### Decisão 5 — Imputação de Nulos

```python
df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
df_clean[cat_cols] = df_clean[cat_cols].fillna("unknown")
```

| Tipo de coluna | Estratégia | Justificativa |
|----------------|------------|---------------|
| Numéricas | Mediana | Robusta a outliers; não distorce distribuição assimétrica |
| Categóricas | Categoria `"unknown"` | Preserva informação de ausência como sinal preditivo |

---

## 3. Principais Achados da EDA

### 3.A Perfil da Base

- 41.188 registros, sem nulos reais (apenas `"unknown"` codificado). 12 duplicatas removidas (<0,05%).
- Perfil modal do cliente: `admin.`, casado, nível superior, sem inadimplência, com financiamento habitacional, sem empréstimo pessoal, contato via celular.
- Variável target fortemente **desbalanceada**: ~89% `no` vs ~11% `yes`. Exige técnicas de balanceamento ou pesos no modelo (SMOTE, `class_weight`).

---

### 3.B Segmentos com Maior e Menor Conversão

| Variável | Categoria | Taxa yes | Observação |
|----------|-----------|----------|------------|
| `poutcome` | `success` | **65,1%** | Melhor indicador individual |
| `month` | `mar` | **50,5%** | Juro baixo (~1,3%) |
| `month` | `dec` | **48,9%** | Juro baixo (~1,0%) |
| `month` | `sep` / `oct` | **44–45%** | Euribor despencando |
| `job` | `student` / `retired` | **25–31%** | Nichos de alto valor, volume limitado |
| `contact` | `cellular` | **14,7%** | vs 5,2% para `telephone` — gap de ~10 pp |
| `default` | `yes` | **0%** | Nunca convertem — excluir da campanha |
| `month` | `may` | **6,4%** | Maior volume (14k), pior conversão |

---

### 3.C Multicolinearidade — Bloco Macro

| Par | Correlação |
|-----|-----------|
| `emp.var.rate` × `euribor3m` | **0,97** |
| `euribor3m` × `nr.employed` | **0,95** |
| `emp.var.rate` × `nr.employed` | **0,91** |
| `cons.price.idx` × `emp.var.rate` | 0,78 |
| `pdays` × `previous` | -0,59 |

As três primeiras variáveis medem essencialmente o mesmo fenômeno econômico. `euribor3m` é mantido como representante do bloco por ser mais interpretável e granular. Variáveis verdadeiramente independentes entre si: `age` e `campaign`.

---

### 3.D Variáveis Numéricas vs Target

- **`euribor3m`:** discriminador forte. `yes` concentrado em 1–2% (juros baixos); `no` em 4–5% (juros altos). Ambiente de juro baixo = maior receptividade a investimento.
- **`emp.var.rate`:** `yes` em valores negativos (recessão), `no` em positivos (expansão). Contra-intuitivo — desemprego crescente favorece adesão (clientes buscam segurança).
- **`cons.conf.idx`:** `yes` distribuído em valores menos negativos — clientes menos pessimistas aderem mais.
- **`age`:** distribuições muito similares entre `yes` e `no` — isoladamente não é discriminador forte; mais útil combinado com `job` ou `education`.
- **`previous`:** grupos com contato anterior convertem mais. Flag `had_previous_contact` captura esse sinal.

---

### 3.E Estabilidade Temporal — Sazonalidade vs Juros

A variação mensal na taxa de conversão é **quase inteiramente explicada pelo nível do `euribor3m`**, não pelo mês em si:

| Mês | Euribor3m médio | Taxa de conversão | Interpretação |
|-----|----------------|-------------------|---------------|
| Março | ~1,3% | ~50% |  Juros baixo — alta receptividade |
| Maio | ~3,3% | ~6% |  Juros subindo — conversão despenca |
| Julho | ~4,5% | ~9% |  Juros no pico — conversão mínima |
| Set/Out | ~1,0% | ~44% |  Juros cai — conversão dispara |
| Dezembro | ~1,0% | ~49% |  Juros baixo novamente |

**Implicação para modelagem:** `month` como feature categórica é redundante com `euribor3m`. O juro captura o efeito de forma numérica e mais rica.

**Paradoxo de volume:** a campanha foi mais intensa em maio (14k contatos, 6% de conversão) — o pior período macroeconômico — sugerindo desalinhamento histórico na estratégia de timing.

---

## 4. Schema Final do Dataset Processado

Shape final: **41.176 × 20** (excluindo target)

| Coluna | Tipo | Status | Observação |
|--------|------|--------|------------|
| `age` | int | original | Mantida |
| `job` | object | original | `"unknown"` como categoria |
| `marital` | object | original | |
| `education` | object | original | `"unknown"` como categoria |
| `default` | object | original |  20,9% de ausência — monitorar |
| `housing` | object | original | |
| `loan` | object | original | |
| `contact` | object | original | Telephone vs cellular |
| `month` | object | original | Proxy de `euribor3m` — avaliar remoção na feature selection |
| `day_of_week` | object | original | Distribuição homogênea — baixo poder preditivo esperado |
| `campaign` | int | modificada | Winsorizado no P99 |
| `previous` | int | original | Mantida |
| `poutcome` | object | original | Melhor preditor categórico |
| `cons.price.idx` | float | original | Macro |
| `cons.conf.idx` | float | original | Macro |
| `euribor3m` | float | original | Representante do bloco macro colinear |
| `contacted_before` | int | **nova** | Flag derivada de `pdays` |
| `had_previous_contact` | int | **nova** | Flag derivada de `previous` |
| `age_group` | category | **nova** | Discretização de `age` |
| `y` | int | target |  Desbalanceada: ~89% no / ~11% yes |

**Colunas removidas:** `duration` (leakage), `pdays` (substituída por flag), `emp.var.rate` e `nr.employed` (multicolinearidade).
