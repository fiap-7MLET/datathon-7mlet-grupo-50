# datathon-7mlet-grupo-50 — Plataforma de Ofertas Adaptativas

Datathon 7MLET. **Visão do problema:** experimentação adaptativa para ofertas bancárias
usando multi-armed bandit.

Uma instituição financeira digital precisa decidir, em cada canal, qual oferta apresentar a
cada cliente elegível. Regras fixas e testes A/B longos desperdiçam tráfego e demoram a
reagir. A abordagem adaptativa (multi-armed bandit) equilibra exploração e explotação e
aprende com as respostas observadas.

**Base Kaggle:** [Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
(versão 1, origem UCI). A coluna `duration` é descartada por vazamento temporal. Detalhes em
[`data/kaggle/README.md`](data/kaggle/README.md) e [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Instruções de execução local

### Ambiente virtual

Pré-requisitos: Python 3.11, 3.12 ou 3.13 e o gerenciador de pacotes
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

Na raiz do repositório, execute:

```bash
uv sync --extra dev
```

O `uv` cria automaticamente o ambiente virtual em `.venv/` e instala as dependências
registradas em `pyproject.toml` e `uv.lock`. Não é necessário ativar o ambiente: os comandos
com `uv run` usam o `.venv` correto automaticamente.

Se preferir ativá-lo manualmente:

```bash
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell

python --version
deactivate                     # encerra o ambiente quando terminar
```

### Pipeline completo

```bash
# baixa e extrai bank-additional-full.csv em data/kaggle/
uv run kaggle datasets download \
  -d henriqueyamahata/bank-marketing \
  -p data/kaggle \
  --unzip

uv run python -m datathon.data_loader  # prepara o Parquet usado no treino

uv run datathon-train            # Etapa 3 + 7: treina as políticas e registra no MLflow
uv run datathon-evaluate         # Etapa 4: métricas comparativas + relatório
uv run datathon-serve            # Etapa 5: sobe a API em http://127.0.0.1:8000
uv run mlflow ui                 # inspeciona parâmetros e métricas dos experimentos
uv run pytest                    # testes
```

O download pela CLI do Kaggle exige autenticação configurada. Depois do download, o comando
`datathon.data_loader` transforma `data/kaggle/bank-additional-full.csv` em
`data/processed/bank_marketing_processed.parquet`, arquivo usado pelo treino.

### Exemplo de chamada

```bash
curl -X POST "http://127.0.0.1:8000/recommend?seed=42" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "job": "student", "contact": "cellular", "month": "mar",
       "education": "university.degree", "housing": "no", "loan": "no",
       "default": "no", "poutcome": "success", "previous": 2}'
```

```json
{
  "arm_id": "arm_004",
  "arm_name": "LCI / LCA 90 dias",
  "channel": "app",
  "score": 0.7432,
  "segment": "previous_converter",
  "algorithm": "contextual_thompson_sampling",
  "contextual": true
}
```

Documentação interativa em `http://127.0.0.1:8000/docs`. `GET /health` mostra qual política
está sendo servida e se ela veio do MLflow ou do arquivo local. `GET /` redireciona para
`/demo` — abrir a raiz do serviço sem saber a rota exata não bate mais em 404.

### Página de demonstração — `/demo`

`http://127.0.0.1:8000/demo` é uma página de apresentação servida pela própria API, sempre
pela política publicada (20.000 clientes). Ao clicar em *Recomendar*, a resposta aparece em
dois passos, nessa ordem — Passo 1 de 2 (segmento) acima do Passo 2 de 2 (oferta), ligados
por uma seta na tela — porque é a ordem real da decisão: atributos decidem o segmento, o
segmento decide a oferta.

1. **Cliente.** Um formulário com 8 campos de **perfil** — idade, ocupação, estado civil,
   educação, default, financiamento, empréstimo e canal. De propósito, fica de fora:
   - `month`, `day_of_week`, `campaign` — descrevem o contato em si (quando ele acontece, em
     que ponto da campanha atual), não um fato sobre o cliente. A API continua aceitando os
     três para quem chamar `/recommend` diretamente, com o valor padrão do schema.
   - `previous` e `poutcome` — apesar de serem 2 dos 4 campos que de fato definem o segmento
     (`context_segment()`), editá-los exigiria explicar a ordem de prioridade das regras na
     tela (`poutcome` decide antes de tudo; `previous` antes do canal). Em vez disso, ficam
     **fixos**: herdados do caso golden carregado, ou tratados como "cliente novo"
     (`previous=0`, `poutcome=unknown`) no modo Personalizado partindo do zero. Só `job` e
     `contact` — os outros 2 campos decisivos, marcados com **●** — continuam ajustáveis.

   Os cinco casos golden da Etapa 4 aparecem como *presets* no seletor "Cliente" — escolher
   um preenche o formulário inteiro. Editar qualquer campo troca automaticamente para
   **Personalizado** e o selo muda de `🔒 caso golden set testado` para `🧪 exploração livre
   — fora do golden set`: o formulário continua funcionando, só deixa de ser um caso
   protegido por teste.

   Os demais campos (idade, estado civil, educação, default, financiamento, empréstimo)
   ainda não influenciam a recomendação com a política atual — mostrado na tela, não
   escondido: é uma limitação documentada do segmento contextual de hoje, não um bug do
   formulário.
2. **Oferta recomendada**, com a crença da política sobre **cada** braço do segmento em
   barras — quanta evidência sustenta cada uma. O botão *Explorar 20×* chama a API 20 vezes
   sem fixar a seed — a política sorteia de verdade a cada chamada — e conta quantas vezes
   cada oferta saiu, provando ao vivo que a exploração é real.

A demo não mostra mais um seletor de "crença publicada vs. em aprendizado" — simplificação
deliberada: as duas políticas eram o mesmo algoritmo em dois pontos de treino diferentes
(20.000 vs. 500 clientes), pensadas para ilustrar a decadência da exploração, mas essa
evidência já está coberta por `reports/etapa4_avaliacao.md` (coluna "concentração", para
todas as políticas), então manter as duas na tela só somava uma explicação sem necessidade.
O snapshot de 500 clientes continua sendo gerado e testado (`uv run datathon-evaluate
--write-golden`, `tests/golden/golden_set_em_aprendizado.json`) — só não aparece mais aqui.

A seed usada em cada chamada também não aparece mais na tela — é um parâmetro técnico sem
explicação de negócio (ver "Reprodutibilidade" abaixo); continua sendo enviada por baixo dos
panos, e dá para conferir no painel "Resposta crua da API", que mostra a URL completa da
chamada com `?seed=`.

`GET /policy` expõe a crença em JSON, para quem quiser auditar a decisão sem a página.

### Reprodutibilidade — o parâmetro `seed`

Thompson Sampling decide sorteando da distribuição posterior de cada braço, então duas
chamadas idênticas podem devolver ofertas diferentes. O parâmetro opcional `seed` fixa esse
sorteio: com ele a resposta é sempre a mesma, o que torna possível ter casos de teste
congelados (o golden set da Etapa 4) e ensaiar a demo sabendo o que a API vai responder.

A seed é escolha de quem chama, não parte do que o modelo aprendeu — por isso ela não é
serializada no estado da política. Sem `seed`, cada chamada sorteia de novo, que é o
comportamento de produção.

As cinco personas do golden set compartilham a mesma seed — `GOLDEN_SEED = 42`, definida em
`client_personas.py` — e o modo Personalizado da demo usa a mesma constante
(`DEFAULT_CUSTOM_SEED = 42` em `demo.html`). De propósito: a seed não é parâmetro do modelo,
é só o sorteio dentro do segmento. Se cada persona tivesse uma seed diferente (como era
antes), uma eventual diferença de oferta entre duas personas poderia ser por causa do
sorteio, não do perfil. Com a seed igual para todas, sobra só uma explicação possível: o
segmento — o argumento de personalização fica limpo.

A página `/demo` não mostra a seed — de propósito. É um parâmetro técnico de
reprodutibilidade, não uma alavanca de negócio; mostrar um número que ninguém consegue
explicar só distrai da história (segmento → oferta). Ela continua sendo enviada em toda
chamada (visível na URL do painel "Resposta crua da API"). Para testar outra seed: mude
`GOLDEN_SEED` em `client_personas.py` (regrava o golden set com `uv run datathon-evaluate
--write-golden`) ou `DEFAULT_CUSTOM_SEED` em `demo.html`, ou chame a API diretamente com
`?seed=`.

Cada segmento acumula sua própria evidência. Dentro de um segmento, a exploração decai à
medida que os posteriors se concentram; entre segmentos, clientes com contextos diferentes
podem receber ofertas diferentes.

## Escolhas de design

| Escolha | Decisão | Porquê |
| --- | --- | --- |
| Algoritmo de produção | Thompson Sampling contextual | Mantém um posterior Beta por braço e segmento; usa o perfil na decisão sem parâmetro manual de exploração. |
| Baseline | Braço fixo de maior `base_conversion_rate` + escolha aleatória | É a decisão que um time de marketing tomaria olhando só a ficha técnica da oferta; o aleatório é o piso. |
| Contexto | **Contextual por segmento** | O perfil é mapeado para `previous_converter`, `student_digital`, `retired`, `low_engagement`, `digital_channel` ou `general`; cada grupo aprende sua distribuição por braço. |
| API | FastAPI | Validação por schema (Pydantic) e documentação OpenAPI automática. |
| Rastreamento | MLflow local | Ferramenta do curso; registra parâmetros, métricas e o artefato da política. |

### Resultado (20.000 clientes simulados)

| Estratégia | Conversão | Uplift vs baseline fixo |
| --- | --- | --- |
| **Thompson Sampling contextual** | **0.4139** | **+24.8%** |
| Thompson Sampling global | 0.4128 | +24.5% |
| UCB1 | 0.3922 | +18.2% |
| Epsilon-Greedy (0.10) | 0.3913 | +18.0% |
| Baseline fixo | 0.3317 | — |
| Baseline aleatório | 0.1850 | −44.2% |

Todos os algoritmos adaptativos superam o baseline. A versão contextual obteve a maior
conversão e mantém decisões diferentes por segmento. Reproduza com `uv run datathon-train`.

Estes números vêm da re-execução do ambiente da Etapa 3 pelas classes de política que a API
serve (`uv run datathon-train`), e é essa execução que fica registrada no MLflow. O notebook 03
implementa a variante **não-contextual** (Thompson global, um Beta por braço, sem segmento na
decisão) — é com ela que os números abaixo são comparáveis. A versão **contextual**, eleita para
produção, só existe nas classes de `src/datathon/bandits/` e é avaliada por `uv run
datathon-train` / `datathon-evaluate`, não dentro do notebook.

Baseline fixo, Thompson (global), UCB1 e Epsilon-Greedy do notebook batem com os valores abaixo
na 3ª/4ª casa decimal — a pequena diferença vem de o notebook sortear com `numpy.random` e as
classes usarem `random.Random`, dois geradores diferentes mesmo com a mesma seed. O **baseline
aleatório** diverge mais (2ª casa, ~3,6 p.p.): é a única política que nunca converge — permanece
uniforme entre braços com taxas bem diferentes o horizonte inteiro — então a média final é
sensível à sequência exata de sorteios, e não só à seed. As demais convergem para o mesmo braço
dominante e por isso são estáveis entre notebook e código.

### Etapa 4 — avaliação e golden set

`uv run datathon-evaluate` refaz a comparação acima medindo, além da conversão e do uplift,
a **convergência**: para qual braço cada política migrou na segunda metade do horizonte e
com que concentração. Saída versionada em [`reports/etapa4_avaliacao.md`](reports/etapa4_avaliacao.md)
e num run MLflow (`avaliacao-etapa4`). Como todas as políticas enfrentam a mesma matriz de
desfechos (*common random numbers*), a diferença entre elas é decisão, não sorte.

Os **cinco casos golden** ficam em `tests/golden/`, congelados em dois momentos da vida da
política — o contraste entre os dois arquivos é o argumento da etapa:

| arquivo | treino | os 5 casos recebem |
| --- | --- | --- |
| `golden_set.json` | 20.000 clientes | ofertas estabilizadas por segmento |
| `golden_set_em_aprendizado.json` | 500 clientes | maior variação por exploração |

Cada arquivo guarda a impressão digital dos posteriors que o gerou. Se a política for
retreinada e a crença mudar, o teste falha pedindo `uv run datathon-evaluate --write-golden`
— mudou o modelo, mudou a recomendação, e o time vê antes de gravar o vídeo.

#### Cinco casos de teste

| Cliente | Segmento | Recomendação | Score | Por que faz sentido |
| --- | --- | --- | ---: | --- |
| Conversor anterior | `previous_converter` | LCI / LCA 90 dias | 0,7432 | O histórico positivo permite uma oferta de investimento com maior compromisso. |
| Estudante digital | `student_digital` | Poupança estudantil | 0,9394 | Produto e canal combinam com estágio de vida e comportamento digital. |
| Aposentado | `retired` | CDB liquidez diária | 0,8547 | Combina perfil conservador com liquidez e simplicidade. |
| Cliente digital de massa | `digital_channel` | Poupança estudantil | 0,3075 | Maior multiplicador do catálogo para este segmento (1,8×) — o nome sugere público estudantil, mas o catálogo não restringe por idade. |
| Baixo engajamento | `low_engagement` | CDB liquidez diária | 0,3993 | Oferta simples como próximo passo, em vez de um produto complexo. |

Os valores estão congelados em `tests/golden/golden_set.json`.

### Etapa 7 — evidência de MLOps

O experimento MLflow `datathon-ofertas-mab` registra parâmetros, conversão, uplift e o JSON
da política. A execução final produziu o run de treino
`1543f1d1fb814b5882bb4de401fec8cc` e o run de avaliação
`158ffe08bbcb4a3385f23db3c748598c`. O estado local do MLflow não é versionado; a evidência
reproduzível é o comando de treino, o relatório versionado e
`data/processed/policy_state.json`.

## Mapa de pastas

```
src/datathon/
├── bandits/            # políticas: base (interface), thompson, ucb, epsilon_greedy, baseline
├── training/           # ambiente de simulação + treino com rastreamento MLflow (Etapas 3 e 7)
├── evaluation/         # métricas comparativas + golden set (Etapa 4)
├── api/                # serviço FastAPI: recommender (domínio), policy_store (carga), main (HTTP)
│   └── static/demo.html    # página de apresentação servida em /demo (Etapa 8)
├── client_personas.py  # os 5 clientes fixos, compartilhados pelo golden set e pela demo
├── data_loader.py      # limpeza do dataset Kaggle (Etapa 2)
└── simulator.py        # geração dos eventos sintéticos de oferta/recompensa
notebooks/              # 01 EDA · 02 enriquecimento sintético · 03 baseline vs adaptativos
data/                   # kaggle (bruto) · processed (tratado + política) · synthetic_enrichment (catálogo)
tests/golden/           # os 5 casos congelados da Etapa 4
reports/                # relatório versionado da avaliação
```

## Governança, LGPD e limites de uso

A finalidade é experimentar ofertas para clientes previamente elegíveis, nunca decidir
crédito, preço, limite ou acesso a serviço essencial. A base legal e a comunicação ao titular
devem ser validadas pelo encarregado de dados. Identificadores diretos, renda, patrimônio e
atributos sensíveis não entram na política. Eventos detalhados devem ser retidos somente pelo
período aprovado e depois agregados ou eliminados.

Decisões sensíveis permanecem com humano no loop. Devem ser monitoradas conversão por segmento,
taxa de exploração, distribuição de ofertas, latência, erros e mudanças de composição. Queda de
conversão, concentração anormal, ausência de feedback ou disparidade relevante exige pausar a
política e voltar ao baseline fixo. As recompensas são simuladas e não provam causalidade nem
desempenho real; produção exige experimento controlado, revisão jurídica, teste de vieses e
limites de frequência de contato.

## Etapa 8 — roteiro do vídeo (até 5 minutos)

1. **0:00–0:40:** problema de negócio e limitações de regras fixas/testes A/B longos.
2. **0:40–1:15:** base Kaggle, remoção de `duration` e preparação dos 41.176 registros.
3. **1:15–2:10:** baseline 0,3317 versus Thompson contextual 0,4139 (+24,8%).
4. **2:10–3:30:** abrir `/demo`, executar três personas e explicar os segmentos.
5. **3:30–4:15:** abrir o MLflow e mostrar parâmetros, métricas e artefato.
6. **4:15–5:00:** arquitetura AWS, limites, humano no loop e próximos passos.

**Link do vídeo:** adicionar aqui antes da submissão.

## Etapa 6 — Arquitetura-alvo em nuvem (AWS)

Em produção, os dados de campanhas e as respostas dos clientes poderão ser armazenados no **Amazon S3**, mantendo separadas as camadas de dados brutos, tratados e os artefatos dos modelos. O treinamento e a avaliação das políticas de recomendação seriam executados no **Amazon SageMaker**, com o **MLflow** registrando parâmetros, métricas e versões dos experimentos. Após a validação, a aplicação e a política aprovada seriam empacotadas em uma imagem de contêiner e publicadas no **Amazon Elastic Container Registry (ECR)**.

A API FastAPI seria executada no **Amazon ECS com AWS Fargate**, permitindo escalar o serviço de recomendação sem administrar servidores, e seria exposta de forma segura pelo **Amazon API Gateway**. Segredos e credenciais ficariam no **AWS Secrets Manager**, sem serem incluídos no código. Logs, latência, erros e métricas de negócio, como conversão, recompensa média e distribuição das ofertas, seriam acompanhados pelo **Amazon CloudWatch**. O fluxo de integração e entrega contínua poderia ser automatizado com **GitHub Actions**, executando os testes antes de publicar uma nova imagem no ECR e atualizar o serviço no ECS.

```mermaid
flowchart LR
    client[Cliente / Canal digital]
    repo[GitHub<br/>Código-fonte]
    pipeline[GitHub Actions<br/>Testes e CI/CD]

    subgraph aws[AWS Cloud]
        gateway[Amazon API Gateway]
        gateway --> alb[Application Load Balancer]
        alb --> api[Amazon ECS + AWS Fargate<br/>API FastAPI]

        api -->|Consulta política aprovada| artifacts[(Amazon S3<br/>Artefatos do modelo)]
        api -->|Lê credenciais| secrets[AWS Secrets Manager]
        api -->|Logs e métricas| monitoring[Amazon CloudWatch]

        raw[(Amazon S3<br/>Dados brutos e tratados)] --> training[Amazon SageMaker<br/>Treinamento e avaliação]
        training --> tracking[MLflow<br/>Experimentos e métricas]
        training -->|Publica política aprovada| artifacts

        registry[Amazon ECR<br/>Imagens de contêiner] -->|Nova versão| api
    end

    client --> gateway
    repo --> pipeline
    pipeline -->|Publica imagem| registry
    pipeline -->|Atualiza serviço| api
```
