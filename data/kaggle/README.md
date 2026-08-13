# Dataset - Bank Marketing

## Fonte

* **Nome:** Bank Marketing Dataset
* **Fonte:** Kaggle
* **Link:** https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
* **Identificador Kaggle:** `henriqueyamahata/bank-marketing`
* **Autor original:** UCI Machine Learning Repository (via Kaggle notebook)

---

## Descrição

Este dataset contém dados de campanhas de marketing direto de uma instituição bancária.

O objetivo é construir uma **Plataforma de Experimentação Adaptativa** através do algoritmo *multi-armed bandit*.

As campanhas foram realizadas principalmente por telefone, e muitas vezes exigiram múltiplos contatos com o mesmo cliente.

---

## Versão

* Versão utilizada: 1
* Base derivada do dataset clássico "Bank Marketing" (UCI)
* Arquivo esperado: `bank-additional-full.csv`
* MD5 verificado: `f6cb2c1256ffe2836b36df321f46e92c`

---

## Licença

* Dataset original: UCI Machine Learning Repository
* Uso para fins acadêmicos e educacionais
* Verificar termos no Kaggle: https://www.kaggle.com/

---

## Limitações

* Dataset histórico (não representa comportamento atual necessariamente)
* Forte desbalanceamento da variável target (`y`)
* Presença de variáveis com alto poder preditivo (risco de leakage)
* Dados limitados ao contexto de campanhas telefônicas
* Não contém informações temporais completas para modelagem causal robusta

---

## Instruções de Download

1. Configure a autenticação da CLI do Kaggle.

2. Na raiz do repositório, instale as dependências:

```bash
uv sync
```

3. Baixe e extraia o dataset diretamente no diretório esperado pelo projeto:

```bash
uv run kaggle datasets download \
  -d henriqueyamahata/bank-marketing \
  -p data/kaggle \
  --unzip
```

O comando cria:

```
data/kaggle/bank-additional-full.csv
data/kaggle/bank-additional-names.txt
```

4. Gere a base processada usada pelo treino:

```bash
uv run python -m datathon.data_loader
```

O resultado será `data/processed/bank_marketing_processed.parquet`.

---

##  Arquivos relacionados

* Dicionário de dados: `docs/data_dictionary.md`
* Análise exploratória: `notebooks/01_eda.ipynb`
* Relatório de qualidade: `docs/data_quality_report.md`

---
