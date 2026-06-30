# Data Dictionary - Bank Marketing Dataset

## 1. Variáveis Originais

| Variável       | Tipo                | Descrição                                | Observações / Valores Possíveis                    |
| -------------- | ------------------- | ---------------------------------------- | -------------------------------------------------- |
| age            | numérica            | Idade do cliente                         | 18-95                                              |
| job            | categórica          | Tipo de emprego                          | admin, blue-collar, entrepreneur, unknown...       |
| marital        | categórica          | Estado civil                             | married, single, divorced...                       |
| education      | categórica          | Nível educacional                        | basic, high school, university, unknown...         |
| default        | categórica          | Possui crédito em default?               | yes / no / unknown                                 |
| housing        | categórica          | Possui financiamento imobiliário?        | yes / no / unknown                                 |
| loan           | categórica          | Possui empréstimo pessoal?               | yes / no / unknown                                 |
| contact        | categórica          | Tipo de contato                          | celular / telefone                                 |
| month          | categórica          | Mês do último contato                    | jan-dec                                            |
| day_of_week    | categórica          | Dia da semana do último contato          | mon-fri                                            |
| duration       | numérica            | Duração da última ligação (segundos)     | Forte leakage                                      |
| campaign       | numérica            | Nº de contatos na campanha atual         | Inclui último contato / 1-56                       |
| pdays          | numérica            | Dias desde último contato anterior       | 999 = nunca contatado                              |
| previous       | numérica            | Nº de contatos anteriores                | 0-7                                                |
| poutcome       | categórica          | Resultado da campanha anterior           | failure, nonexistent, success                      |
| emp.var.rate   | numérica            | Taxa de variação de emprego (trimestral) | Indicador macroeconômico                           |
| cons.price.idx | numérica            | Índice de preços ao consumidor           | Indicador mensal                                   |
| cons.conf.idx  | numérica            | Índice de confiança do consumidor        | Indicador mensal                                   |
| euribor3m      | numérica            | Taxa Euribor 3 meses                     | Indicador diário                                   |
| nr.employed    | numérica            | Número de empregados                     | Indicador trimestral                               |
| y              | categórica (target) | Cliente subscreveu depósito a prazo?     | yes / no                                           |

---

## 2. Variáveis Derivadas

| Variável              | Tipo | Como foi criada                          | Objetivo                                                              | Risco de Leakage |
| --------------------- | ---- | ---------------------------------------- | --------------------------------------------------------------------- | ---------------- |
| contacted_before      | int  | (df_clean['pdays'] != 999).astype(int)   | Indica se o cliente já foi contactado antes (baseado em pdays ≠999)   | Não              |
| had_previous_contact  | int  | (df_clean['previous'] > 0).astype(int)   | Indica se há registro explícito de contatos anteriores (previous > 0) | Não              |

---

## 3. Observações Gerais

* Valores "unknown" representam dados ausentes em variáveis categóricas
* A variável `duration` não deve ser utilizada em modelos preditivos realistas devido a vazamento de informação (data leakage)
* Dataset apresenta desbalanceamento na variável target (`y`)
* Algumas variáveis são indicadores macroeconômicos e podem introduzir dependência temporal

---

## 4. Fonte

Moro, S., Cortez, P., & Rita, P. (2014).
A Data-Driven Approach to Predict the Success of Bank Telemarketing.
Decision Support Systems.
http://dx.doi.org/10.1016/j.dss.2014.03.001

---
