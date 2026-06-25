# Dataset - Bank Marketing

## Fonte

* **Nome:** Bank Marketing Dataset
* **Fonte:** Kaggle
* **Link:** https://www.kaggle.com/code/henriqueyamahata/bank-marketing-classification-roc-f1-recall
* **Autor original:** UCI Machine Learning Repository (via Kaggle notebook)

---

## Descrição

Este dataset contém dados de campanhas de marketing direto de uma instituição bancária.

O objetivo é construir uma **Plataforma de Experimentação Adaptativa** através do algoritmo *multi-armed bandit*.

As campanhas foram realizadas principalmente por telefone, e muitas vezes exigiram múltiplos contatos com o mesmo cliente.

---

## Versão

* Versão utilizada: versão disponível no Kaggle em Junho de 2026
* Base derivada do dataset clássico "Bank Marketing" (UCI)

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

1. Acesse o link do Kaggle:
   https://www.kaggle.com/code/henriqueyamahata/bank-marketing-classification-roc-f1-recall

2. Baixe os dados associados ao notebook (ou dataset original "Bank Marketing")

3. Alternativamente, via Kaggle API:

```bash
kaggle datasets download -d <dataset-name>
```

4. Salve os arquivos na pasta:

```
data/kaggle/
```

---

##  Arquivos relacionados

* Dicionário de dados: `docs/data_dictionary.md`
* Análise exploratória: `notebooks/eda.ipynb`
* Relatório de qualidade: `docs/data_quality_report.md`

---
