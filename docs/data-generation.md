# Documentação - Enriquecimento Sintético

**Versão:** 1.0.0  
**Data:** 2026-06-29  
**Dataset base:** UCI Bank-Marketing (n = 41.188 clientes)  
**Arquivos de referência:** `data/synthetic_enrichment`

---

## 1. Introdução

No vocabulário de **Multi-Armed Bandit (MAB)**, um "braço" é uma ação possível que o sistema pode tomar — neste caso, uma oferta ou mensagem apresentada ao cliente. O algoritmo MAB aprende ao longo do tempo quais braços convertem mais para cada tipo de cliente, equilibrando exploração (testar braços menos conhecidos) e explotação (preferir braços que já provaram resultado).

Nesse contexto, os dados sintéticos são a simulação do experimento em produção: representam o que aconteceria se o sistema MAB estivesse rodando com clientes reais, tomando decisões em tempo real, aguardando resultados com delay e aprendendo com cada conversão.

Assim, esta documentação define descreve os *json* `offer_catalog.json`, `offer_events.json` e `delayed_rewards.json`

---

## 1.1. Papel de cada arquivo no projeto (o que é usado onde)

Os três arquivos são gerados através do código `src/datathon/simulator.py`

| Arquivo | Papel no projeto |
|---|---|---|
| `offer_catalog.json` | **Modelo de recompensa do ambiente.** Define os braços, `base_conversion_rate` e `segment_multipliers` — a fonte de verdade sobre "quanto cada oferta converte para cada perfil". |
| `offer_events.json` (200 linhas) | **Evidência de que a camada de experimentação sabe registrar uma decisão**: contexto do cliente, braço escolhido, segmentos derivados, probabilidades e sinalização de exploração |
| `delayed_rewards.json` (200 linhas) | **Evidência de que o delayed reward está modelado**: liga um `offer_event` ao resultado observado dias depois, respeitando o `reward_delay_days` de cada braço. |

**Por que `offer_events.json`/`delayed_rewards.json` não entram na avaliação dos algoritmos:** esses 200 eventos foram gerados usando o próprio Thompson Sampling como política de coleta (`generate_synthetic.py` já roda um TS simulado para escolher os braços). Usar esse log para depois "avaliar" o Thompson Sampling seria circular: estaríamos validando o algoritmo com dados que ele mesmo produziu. Além disso, 200 eventos são poucos para qualquer algoritmo de bandit convergir de forma visível

---

## Arquivo 1: `offer_catalog.json`

Este catálogo define todos os braços disponíveis, os segmentos de clientes, as taxas de conversão esperadas e os multiplicadores por segmento.

### Taxa de conversão global do dataset

```python
df['y_bin'] = (df['y'] == 'yes').astype(int)
base = df['y_bin'].mean()
# Resultado: 0.113 — 11,3% dos clientes aceitaram o depósito a prazo
```

Esse número é a **âncora de tudo**. Ele vem diretamente dos dados e responde à pergunta: "se você abordasse um cliente aleatório do banco com uma oferta de depósito, quantos por cento aceitariam?" A resposta medida é 11,3%.

### Multiplicadores globais dos segmentos

Para cada segmento, calculamos a taxa de conversão específica e dividimos pela taxa geral:

```python
taxa_segmento = df[df[segmento]]['y_bin'].mean()
multiplicador_global = taxa_segmento / base
```

O multiplicador global responde: "clientes desse segmento convertem quantas vezes mais do que a média?" Esses valores são o ponto de partida para calibrar os `segment_multipliers` de cada braço — mas não são usados diretamente, porque o multiplicador por braço precisa considerar a adequação do produto ao perfil, não só a propensão geral do segmento.

---

### Segmentos

Os segmentos foram criados a partir de três critérios combinados:

1. **Poder discriminante alto** — variável com grande diferença de taxa entre categorias, medida nos gráficos de distribuição por variável categórica do dataset
2. **Volume suficiente** — segmentos muito pequenos não geram dados suficientes para o MAB aprender (mínimo estimado: 500 clientes)
3. **Sentido de negócio** — a diferença de taxa deve ter uma explicação lógica e documentável, não ser ruído estatístico

#### Segmentos com base empírica

Calculados diretamente do dataset como `taxa_segmento / base_global`:

| Segmento | Regra de criação | Taxa medida | Multiplicador global |
|---|---|---|---|
| `previous_converter` | `poutcome == 'success'` | 65,1% | 5,78× |
| `high_season` | `month in ['mar','sep','oct','dec']` | 46,5% | 4,12× |
| `student_digital` | `job == 'student' AND contact == 'cellular'` | 36,4% | 3,23× |
| `student_saver` | `job == 'student'` | 31,4% | 2,78× |
| `retired` | `job == 'retired'` | 25,3% | 2,24× |
| `university_educated` | `education == 'university.degree'` | 13,7% | 1,21× |
| `digital_channel` | `contact == 'cellular'` | 14,7% | 1,30× |
| `admin_worker` | `job == 'admin.'` | 13,0% | 1,15× |

Essa definição foi baseada nas análises que constam no notebook `notebooks/eda.ipynb`, nas quais as análises dos itens `Segmentação das variáveis categóricas por resultado (y)`, `Interações entre Segmentos` e `Análise Matriz Volume x Taxa de Conversão`, mostraram que as variáveis com o maior poder discrimatório eram: 

- poutcome;
- month (mar/sep/oct/dez);
- job (student/retired/admin);
- contact (cellular);
- education (university.degree.);

**Pontos de Atenção:**

- O heatmap de `job × contact` mostrou que estudantes abordados por celular convertem 36,4%, mas estudantes abordados por telefone fixo convertem apenas 15,2% — uma diferença de 2,4×. Usar só `job == 'student'` misturaria dois comportamentos muito distintos num único segmento, diluindo o sinal.
  
- `high_season` é uma variável contextual, pois não é uma característica do cliente, mas do momento. Os meses mar/set/out/dez têm taxas de conversão 4× maiores que a média, provavelmente por sazonalidade fiscal e de planejamento financeiro. Isso tem consequências para o algoritmo: o braço certo para um cliente em março pode não ser o mesmo braço certo para o mesmo cliente em maio.

- `admin_worker` tem uma taxa individual baixa, mas o segmento existe por volume: administradores representam ~10 mil clientes no dataset. Em segmentos de nicho como `student_digital` ou `previous_converter`, o MAB pode demorar para acumular observações suficientes para aprender. O `admin_worker` é capaz de sustentar o experimento com volume constante enquanto os nichos ainda estão sendo explorados.

#### Segmentos derivados por lógica de negócio

Esses segmentos não têm taxa medida diretamente porque dependem de combinações de colunas com baixo poder discriminante individual:

**`debt_free`** (`housing == 'no' AND loan == 'no' AND default == 'no'`): as colunas `housing` e `loan` isoladamente variam menos de 1 ponto percentual na taxa de conversão. Mas a combinação das três colunas identifica um perfil financeiro coerente: cliente sem comprometimento de renda. Esse perfil tem sentido de negócio claro para produtos que exigem capital disponível (LCI/LCA, CDB longo).

**`low_engagement`** (`previous == 0 AND contacted_before == 0`): identifica clientes que nunca foram abordados em campanhas anteriores. Definem o público-alvo do braço de reengajamento: clientes que o banco ainda não tentou converter.

**Pontos de Atenção:**

- **Estado civil** tem variância pequena no dataset (10,2% a 15,0%) e seu uso como critério de segmentação pode gerar questões regulatórias e de viés — decisões de oferta baseadas em estado civil são sensíveis do ponto de vista de fairness em serviços financeiros.

- `housing` e `loan` não viraram segmentos individuais, pois os gráficos de distribuição mostram variância mínima: `housing` varia de 10,8% a 11,6%, `loan` de 10,9% a 11,3%. Com variação de menos de 1 ponto percentual, o custo de criar e manter um segmento não se justifica.

---

### Braços — decisões de design

#### Como a `base_conversion_rate` foi definida para cada braço

A `base_conversion_rate` representa a taxa de conversão esperada do produto para um cliente aleatório da população, sem nenhuma segmentação aplicada. 

O dataset Bank-Marketing tem apenas um produto (depósito a prazo), que foi usado como âncora empírica para o CDB, por exemplo. Logo, todos os outros produtos foram estimados por relação com essa âncora, usando o seguinte raciocínio:

- Produtos com **menor fricção** que o CDB (sem prazo fixo, sem valor mínimo, adesão imediata) tendem a converter mais porque removem objeções. Estimados acima de 0,11.
- Produtos com **maior complexidade** que o CDB (exigem mais comprometimento, entendimento de benefício fiscal, ciclo de decisão mais longo) tendem a converter menos. Estimados abaixo de 0,11.
- O **braço de controle** não é uma oferta de produto — é ausência de ação. A taxa de 0,04 representa conversão orgânica, estimada como ~35% da taxa geral, com base na premissa de que a maioria das conversões do dataset foi induzida pela campanha e não teria ocorrido espontaneamente.

| Braço | Base rate | Posição relativa ao CDB | Justificativa da estimativa |
|---|---|---|---|
| Poupança estudantil | 0,16 | +45% | Maior propensão do dataset (estudante = 31,4%) + zero fricção de produto |
| CDB liquidez diária | 0,15 | +36% | Remove a única objeção do CDB (prazo fixo), amplia público elegível |
| LCI/LCA 90 dias | 0,14 | +27% | Isenção de IR + prazo curto = dois diferenciais competitivos |
| Poupança digital | 0,13 | +18% | Produto mais simples, sem prazo, menor comprometimento |
| **CDB 12 meses** | **0,11** | **âncora** | **Medido diretamente: (df['y']=='yes').mean() = 0,113** |
| CDB 24 meses | 0,09 | −18% | Prazo longo reduz público elegível; exige maior confiança |
| Previdência PGBL | 0,08 | −27% | Maior complexidade: benefício fiscal, contrato, carência |
| Reengajamento | 0,06 | −45% | Funil de dois passos — converte engajamento, não produto direto |
| Controle | 0,04 | −64% | Conversão orgânica; estimado como ~35% da taxa geral |

#### Como o `reward_delay_days` foi definido para cada braço

O delay é o número de dias que o sistema deve aguardar antes de registrar se houve conversão. Algoritmos como Thompson Sampling atualizam suas crenças quando recebem o sinal de recompensa — se atualizarem cedo demais, usam informação incompleta e subestimam a taxa real do braço.

O delay reflete dois fenômenos diferentes dependendo do produto:

**Delay técnico**: tempo de processamento bancário independente da decisão do cliente. Transferências, confirmações de depósito, processamento de contratos. Afeta principalmente CDB e LCI/LCA.

**Delay de decisão**: tempo que o cliente precisa para deliberar antes de assinar. Previdência e CDB longo têm ciclos de decisão mais longos porque o cliente pesquisa, compara e posterga. O sistema precisa esperar esse ciclo antes de concluir que não houve conversão.

| Braço | Delay | Tipo dominante | Raciocínio |
|---|---|---|---|
| Controle | 0 dias | nenhum | Sem ação ativa, sem delay a medir |
| CDB liquidez diária | 2 dias | técnico | Adesão imediata via app, equivalente a abrir conta de investimento |
| Poupança estudantil | 2 dias | técnico | Produto 100% digital, sem análise de crédito |
| Poupança digital | 3 dias | técnico | Ativação automática, sinal disponível após confirmação de primeiro depósito |
| LCI/LCA 90 dias | 5 dias | técnico | Processamento similar ao CDB, prazo mais curto acelera confirmação |
| CDB 12 meses | 7 dias | técnico | TED/transferência + processamento interno do banco |
| Reengajamento | 10 dias | decisão | Ciclo completo: receber mensagem → processar → reagir → iniciar jornada |
| CDB 24 meses | 10 dias | decisão | Cliente delibera mais antes de comprometer capital por 2 anos |
| Previdência PGBL | 14 dias | decisão + técnico | Pesquisa de produto + assinatura de contrato + carência inicial |

#### Como os `segment_multipliers` foram definidos por braço

O multiplicador de um segmento em um braço específico **não é o mesmo** que o multiplicador global do segmento. O multiplicador global mede propensão geral a qualquer depósito bancário. O multiplicador por braço mede afinidade com aquele produto específico, considerando se o produto é adequado para o momento de vida e perfil financeiro daquele segmento.

O processo de calibração foi:

1. Tomar o multiplicador global do segmento como referência de partida
2. Ajustar para cima se o produto tem características que combinam especificamente com o perfil
3. Ajustar para baixo se o produto tem características inadequadas para o perfil
4. Documentar o raciocínio de cada ajuste

O caso mais ilustrativo é o segmento `retired`, que tem multiplicador global de 2,24× (aposentados convertem muito acima da média), mas multiplicadores muito diferentes por produto:

| Produto | Multiplier | Por que diverge do global |
|---|---|---|
| CDB liquidez diária | 2,5× | Remove a principal objeção do segmento (prazo fixo) — produto ideal |
| Poupança digital | 1,6× | Liquidez e segurança combinam com o perfil |
| LCI/LCA 90 dias | 1,4× | Isenção de IR ainda é atrativa; prazo curto ajuda |
| CDB 12 meses | 1,2× | Capital disponível, mas prazo incomoda — multiplier moderado |
| CDB 24 meses | 0,7× | Prazo pode ultrapassar o horizonte de planejamento |
| Previdência PGBL | 0,3× | Produto de acumulação para quem está na fase de retirada — inadequação de produto explícita |
| Poupança estudantil | 0,2× | Produto completamente fora do perfil |

---

### Braços — decisões campo a campo

#### arm_001 — CDB 12 meses

**`base_conversion_rate`: 0,11**

Único braço com taxa medida diretamente do dataset. O produto do Bank-Marketing é um depósito a prazo — equivalente funcional ao CDB de 12 meses no contexto brasileiro. A taxa medida foi 0,113, arredondada para 0,11 para evitar falsa precisão em um prior que será sobrescrito pelo experimento.

**`reward_delay_days`: 7**

Delay do tipo técnico. Um CDB requer: (1) decisão do cliente, (2) transferência TED ou PIX para o banco, (3) processamento interno e confirmação da aplicação no sistema. Sete dias úteis é uma estimativa conservadora que cobre o ciclo completo sem risco de registrar falso negativo por delay de processamento.

**`segment_multipliers` — decisões de ajuste:**

- `previous_converter: 2,5` — o multiplicador global desse segmento é 5,78×, mas ajustado para baixo porque clientes que já converteram antes tendem a conhecer melhor os produtos do banco e frequentemente buscam algo mais elaborado na segunda contratação. CDB padrão é produto de entrada, não de retenção premium.
- `retired: 1,2` — aposentados têm capital disponível e familiaridade com CDB, mas o prazo de 12 meses é uma fricção real para quem prioriza liquidez. Multiplier moderado — nem alto nem baixo.
- `low_engagement: 0,6` — clientes sem histórico de interação respondem mal a produtos que exigem comprometimento de prazo na primeira abordagem. CDB direto é prematuro para esse perfil.

---

#### arm_002 — Poupança digital

**`base_conversion_rate`: 0,13**

Estimado 18% acima do CDB (0,11 × 1,18 ≈ 0,13). A poupança digital tem menor barreira de entrada em todas as dimensões: sem prazo mínimo, sem valor mínimo, sem comprometimento de capital. Cada objeção removida amplia o público potencial e eleva a taxa esperada.

**`reward_delay_days`: 3**

Poupança digital é ativada automaticamente após adesão no app. O sinal de conversão não depende de transferência bancária — o primeiro depósito automático (ou a confirmação de saldo) é suficiente. Três dias cobrem o processamento da confirmação com margem.

**`segment_multipliers` — decisões de ajuste:**

- `student_digital: 2,0` — poupança é o produto de entrada natural para jovens que estão começando a vida financeira. Remove todas as objeções do público estudantil: sem mínimo, sem prazo, sem comprometimento. O alto multiplicador global desse segmento (3,23×) é parcialmente mantido porque o produto é adequado, mas moderado porque estudantes frequentemente têm saldo baixo para depositar.
- `retired: 1,6` — aposentado valoriza liquidez e segurança acima de rendimento. Poupança combina diretamente com esse perfil, em contraste com CDB ou previdência. É um dos poucos produtos onde retired tem multiplier alto.
- `university_educated: 1,0` — educação superior não diferencia aqui porque poupança é produto universal com rendimento baixo. Universitários com literacia financeira tendem a buscar produtos de maior rendimento quando têm capital disponível.
- `low_engagement: 1,1` — produto de baixa fricção pode funcionar como porta de entrada para clientes nunca abordados. Levemente positivo porque é o produto de menor comprometimento do catálogo.

---

#### arm_003 — Previdência privada (PGBL)

**`base_conversion_rate`: 0,08**

Estimado 27% abaixo do CDB (0,11 × 0,73 ≈ 0,08). A previdência PGBL tem o maior conjunto de exigências do catálogo: o cliente precisa ter renda tributável suficiente para usar a dedução no IR, entender o mecanismo do benefício fiscal, aceitar o comprometimento de longo prazo e passar pelo processo de contratação (análise de perfil + assinatura). Cada exigência filtra parte do público potencial. Canal email foi definido porque o produto demanda comunicação detalhada — não cabe em uma notificação push.

**`reward_delay_days`: 14**

Combina delay de decisão e delay técnico. Do lado da decisão: o cliente precisa pesquisar, comparar fundos, consultar o impacto no IR e eventualmente discutir com um assessor. Do lado técnico: a contratação envolve assinatura de contrato e registro do plano com período de carência inicial. Quatorze dias é a estimativa mais conservadora do catálogo, justificada pela maior complexidade do ciclo de venda.

**`segment_multipliers` — decisões de ajuste:**

- `retired: 0,3` — a inadequação de produto mais clara do catálogo. PGBL é um plano de previdência para quem está na fase de acumulação — construindo patrimônio para se aposentar. Um cliente já aposentado está na fase oposta, de retirada. Oferecer previdência a aposentados não é apenas ineficiente: é potencialmente inadequado do ponto de vista regulatório (suitability). O multiplier baixíssimo é intencional.
- `student_digital: 0,5` — estudante tem horizonte temporal longo (argumento a favor), mas geralmente não tem renda tributável suficiente para aproveitar a dedução do PGBL no IR (argumento determinante contra). Produto errado para o momento de vida financeira.
- `university_educated: 1,8` — literacia financeira alta aumenta diretamente a capacidade de entender e valorizar o benefício fiscal do PGBL. Profissionais com diploma tendem a declarar IR na forma completa e a buscar otimização tributária. Multiplier alto porque o diferencial do produto (benefício fiscal) é compreendido por esse segmento.
- `debt_free: 1,6` — sem comprometimentos financeiros ativos, o cliente tem capacidade de aportar mensalmente sem comprometer liquidez. Perfil financeiro ideal para previdência de longo prazo.
- `low_engagement: 0,4` — produto de máxima complexidade e comprometimento para um cliente que nunca demonstrou interesse em nenhum produto bancário. A probabilidade de conversão é muito baixa, e uma abordagem prematura pode criar associação negativa.

---

#### arm_004 — LCI / LCA 90 dias

**`base_conversion_rate`: 0,14**

Estimado 27% acima do CDB (0,11 × 1,27 ≈ 0,14). Dois fatores combinados justificam a taxa mais alta: a isenção de IR aumenta o rendimento líquido efetivo (tornando o produto competitivo com CDBs de taxa maior) e o prazo curto de 90 dias reduz significativamente a objeção de liquidez comparado ao CDB de 12 meses. A combinação dos dois diferenciais amplia o público receptivo.

**`reward_delay_days`: 5**

Processamento similar ao CDB, mas ligeiramente mais rápido porque o prazo mais curto da LCI/LCA implica uma jornada de contratação menos burocrática internamente. Cinco dias cobrem o ciclo de transferência e confirmação com margem.

**`segment_multipliers` — decisões de ajuste:**

- `university_educated: 2,0` — o diferencial principal da LCI/LCA é a isenção de IR. Para que esse diferencial seja atrativo, o cliente precisa: (a) saber que existe, (b) entender quanto representa no rendimento líquido, (c) comparar com alternativas. Literacia financeira alta é precondição para valorizar o produto. Maior multiplier desse segmento no catálogo.
- `debt_free: 2,2` — cliente sem comprometimentos financeiros + produto com liquidez razoável (90 dias) = combinação direta de capacidade e adequação. Maior multiplier do segmento neste braço.
- `student_digital: 0,8` — estudante geralmente não tem renda tributável, portanto a isenção de IR não é um diferencial real para ele. O produto não tem o apelo correto para esse segmento.
- `retired: 1,4` — aposentado com renda de complemento (aluguel, dividendos, pensão) ainda paga IR sobre esses rendimentos. A isenção é atrativa e o prazo de 90 dias combina com a preferência por liquidez do segmento.

---

#### arm_005 — CDB liquidez diária

**`base_conversion_rate`: 0,15**

Estimado como o mais alto entre os CDBs (0,15), acima do CDB padrão (0,11). A justificativa é que a liquidez diária remove a única barreira que o CDB tradicional tem: o prazo fixo. Ao eliminar essa objeção, o produto se torna elegível para segmentos que resistem a qualquer forma de comprometimento — aposentados, clientes com baixo engajamento, clientes que nunca investiram antes. Esse público adicional, somado ao público original do CDB, eleva a taxa esperada.

**`reward_delay_days`: 2**

Produto funcionalmente equivalente a uma conta de investimento de resgate imediato. A adesão é digital e o saldo fica disponível no dia seguinte ao depósito. Dois dias cobrem a confirmação técnica com margem mínima.

**`segment_multipliers` — decisões de ajuste:**

- `retired: 2,5` — o maior multiplier do segmento `retired` em qualquer braço do catálogo. A liquidez diária remove exatamente a objeção que faz aposentados evitarem CDB tradicional. É o produto mais adequado para o perfil de liquidez desse segmento.
- `low_engagement: 1,8` — cliente sem histórico de engajamento tem aversão percebida a comprometimento. Um produto que devolve o dinheiro a qualquer momento elimina o principal argumento contra. Multiplier alto porque a remoção de fricção é mais valiosa para quem nunca investiu do que para quem já tem experiência.
- `university_educated: 1,1` — universitário com literacia financeira tende a comparar rendimentos e perceber que liquidez diária implica taxa menor que CDB a prazo. Para quem entende a diferença, o sacrifício de rendimento é uma desvantagem real. Multiplier ligeiramente positivo porque o produto ainda é melhor que poupança.

---

#### arm_006 — Oferta de reengajamento

**`base_conversion_rate`: 0,06**

Este braço não converte diretamente em produto bancário — converte em **engajamento**. A taxa de 0,06 mede a probabilidade de o cliente abrir a mensagem, clicar e iniciar alguma forma de interação, não de contratar um produto. O funil tem dois passos: reengajamento primeiro, produto na campanha seguinte.

Estimado bem abaixo do CDB porque: (1) é um passo anterior no funil, não a conversão final; (2) o público-alvo (`low_engagement`) é por definição o de menor propensão a qualquer interação; (3) o canal email/SMS tem taxa de abertura menor que app/push para esse perfil.

**`reward_delay_days`: 10**

O ciclo de reativação tem etapas sequenciais: o cliente recebe a mensagem, leva alguns dias para abri-la, processa o conteúdo, eventualmente clica, e inicia uma jornada de produto que pode levar mais alguns dias. Dez dias captura esse ciclo completo de reativação, não só o delay técnico.

**`segment_multipliers` — decisões de ajuste:**

- `low_engagement: 2,5` — este braço existe fundamentalmente para este segmento. Cliente nunca abordado é o único que pode ser genuinamente reativado, porque não tem fadiga de campanha nem associação negativa com comunicações anteriores do banco. O multiplier alto reflete que é o público certo para a mensagem certa.
- `previous_converter: 0,5` — cliente que já converteu em campanha anterior já está ativo no banco. Enviar uma mensagem de reengajamento do tipo "veja o rendimento que você perdeu" para alguém que já investe é: (a) inadequado porque o pressuposto da mensagem está errado, (b) potencialmente irritante porque contradiz a experiência atual do cliente.
- `digital_channel: 1,4` — mensagens de reengajamento entregues por celular têm taxa de abertura substancialmente maior do que por email ou carta. O canal amplifica a efetividade da mensagem independentemente do conteúdo.

---

#### arm_007 — CDB 24 meses (premium)

**`base_conversion_rate`: 0,09**

Estimado 18% abaixo do CDB padrão (0,11 × 0,82 ≈ 0,09). O prazo de 24 meses cria duas barreiras adicionais: (1) o público elegível é mais restrito — apenas clientes com capital disponível que não precisarão do dinheiro por 2 anos; (2) o ciclo de decisão é mais longo — comprometer capital por 24 meses é uma decisão que o cliente posterga e pesquisa mais.

A taxa mais baixa não significa produto pior: a rentabilidade para o banco e para o cliente é maior. Significa que o público disposto a esse comprometimento é um subconjunto menor.

**`reward_delay_days`: 10**

Diferente do CDB de 12 meses (delay técnico de 7 dias), aqui o delay é predominantemente de decisão. O processamento técnico seria similar, mas o cliente leva mais tempo para assinar porque o comprometimento é maior. Dez dias é uma estimativa que reconhece esse ciclo de deliberação mais longo.

**`segment_multipliers` — decisões de ajuste:**

- `previous_converter: 3,5` — o maior multiplier desse segmento em qualquer braço do catálogo. Cliente que já converteu anteriormente e está sendo abordado novamente está demonstrando disposição clara a comprometimento com o banco. CDB premium de 24 meses é a oferta natural para esse perfil — é um upgrade lógico do relacionamento.
- `retired: 0,7` — prazo de 24 meses pode ultrapassar o horizonte de planejamento financeiro de aposentados mais velhos. Diferente do CDB de 12 meses (multiplier 1,2), aqui a objeção de prazo é mais severa e o multiplier cai abaixo de 1.
- `low_engagement: 0,3` — produto de máximo comprometimento para quem nunca demonstrou qualquer interesse em produto bancário. Próximo de zero — seria o produto mais inadequado para a primeira abordagem.
- `student_digital: 0,6` — estudante raramente tem capital para comprometer por 24 meses. O produto exige uma maturidade financeira que não combina com o momento de vida do segmento.

---

#### arm_008 — Poupança estudantil

**`base_conversion_rate`: 0,16**

Taxa mais alta do catálogo. A estimativa é justificada por três fatores que se multiplicam:

**Fator 1 — propensão do segmento:** estudantes têm a maior taxa de conversão do dataset (31,4%), aproximadamente 3× a média geral. Isso está documentado nos gráficos de distribuição por variável categórica.

**Fator 2 — adequação do produto:** poupança estudantil remove todas as barreiras de entrada possíveis — sem valor mínimo, sem prazo, sem análise de crédito, sem documentação adicional além do cadastro básico. Não existe produto mais acessível no catálogo.

**Fator 3 — canal:** estudantes são o segmento com maior uso de celular. A combinação `student + cellular` converte 36,4% no dataset. O produto é entregue via app/push, que é o canal preferido do segmento.

A combinação de alta propensão do segmento com produto de zero fricção e canal ideal justifica a estimativa de 0,16.

**`reward_delay_days`: 2**

Adesão 100% digital via app, sem análise de crédito ou aprovação. O produto é ativado automaticamente após o cadastro. Dois dias é o menor delay prático para qualquer produto que envolva processamento bancário.

**`segment_multipliers` — decisões de ajuste:**

- `student_digital: 3,5` — produto construído para este segmento exato. Estudante + celular + poupança simplificada é a combinação de máxima afinidade do catálogo. O multiplier é o mais alto de qualquer segmento em qualquer braço.
- `student_saver: 3,0` — versão mais ampla do student_digital (todos os canais, não só celular). Mesmo público, multiplier ligeiramente menor porque inclui estudantes abordados por telefone fixo, que convertem menos.
- `digital_channel: 1,8` — produto 100% digital aumenta a relevância do canal celular como preditor de conversão. Cliente que usa celular como canal principal está mais preparado para completar a adesão via app.
- `retired: 0,2` — produto de onboarding estudantil completamente fora do perfil de um aposentado. Multiplier mínimo — abaixo apenas de braços com inadequação explícita de produto.
- `university_educated: 1,6` — universitário tem renda ou bolsa e começa a pensar em estruturar as finanças. Poupança estudantil pode ser a porta de entrada antes de produtos mais elaborados.

---

#### arm_control — Nenhuma ação

**`base_conversion_rate`: 0,04**

A taxa de conversão orgânica — clientes que buscam e contratam produto por conta própria, sem nenhuma abordagem ativa do banco — não é medida diretamente no dataset, porque todos os registros do dataset correspondem a clientes que foram abordados por campanha.

A estimativa de 0,04 usa o seguinte raciocínio: se a taxa geral com campanha é 0,113 e a taxa sem campanha (controle) fosse igual, o experimento não teria valor. A literatura de marketing direto sugere que campanhas tipicamente geram 2× a 4× a taxa orgânica em produtos financeiros de varejo. Usando o fator mais conservador (3×): taxa orgânica ≈ 0,113 / 3 ≈ 0,038, arredondado para 0,04.

O braço de controle é o benchmark do experimento: se um braço ativo não superar consistentemente 0,04 ao longo do tempo, ele não está gerando valor incremental sobre o que aconteceria sem nenhuma intervenção.

**`reward_delay_days`: 0**

Sem ação ativa, não há delay a medir. A conversão orgânica é detectada em tempo real pelo sistema de CRM quando o cliente contrata um produto por conta própria.

---

### Revisão e manutenção

#### Quando revisar os multiplicadores

- Após acumular 500+ observações por braço no experimento — nesse ponto, os dados reais do MAB substituem os priors definidos aqui
- Ao adicionar um novo produto ao catálogo (novos braços exigem novos multiplicadores para todos os segmentos)
- Se o MAB convergir para um único braço sem explorar os demais — pode indicar prior muito forte ou multiplicador mal calibrado

#### O que não alterar retroativamente

`arm_id`, `created_at` e os valores originais de `base_conversion_rate` não devem ser modificados após o início do experimento. Alterações retroativas invalidam a comparabilidade dos resultados acumulados. Para corrigir uma calibração, crie uma nova versão do catálogo com `version` incrementado e documente a mudança.

---

## Arquivo 2: `offer_events.jsonl`

Cada linha é um **evento de impressão**, isto é, o momento exato em que o sistema MAB tomou a decisão de qual braço apresentar a um cliente específico.

### Exemplo de um evento

```json
{
  "event_id": "evt_000001",
  "timestamp": "2024-03-01T09:14:22Z",
  "client_id": "client_04821",
  "arm_id": "arm_008",
  "channel": "app",
  "context": {
    "job": "student",
    "education": "university.degree",
    "age_bucket": "18-24",
    "balance_bucket": "low",
    "has_housing_loan": false,
    "has_personal_loan": false,
    "contact_type": "cellular",
    "days_since_last_contact": 999,
    "previous_campaigns": 0,
    "poutcome": "nonexistent",
    "month": "mar",
    "day_of_week": "monday"
  },
  "derived_segments": ["student_digital", "student_saver", "digital_channel",
                       "debt_free", "low_engagement", "high_season"],
  "policy_version": "v2.0.0",
  "selection_algorithm": "thompson_sampling",
  "arm_probabilities": {
    "arm_001": 0.0812, "arm_002": 0.1103, "arm_003": 0.0445,
    "arm_004": 0.0921, "arm_005": 0.1678, "arm_006": 0.0334,
    "arm_007": 0.0556, "arm_008": 0.3701, "arm_control": 0.0450
  },
  "exploration_flag": false,
  "random_seed": 42,
  "reward_expected": 0.8960,
  "reward_observed": 1.0,
  "reward_timestamp": "2024-03-03T09:14:22Z",
  "status": "converted"
}
```

### Dicionário de campos

| Campo | Tipo | O que significa |
|---|---|---|
| `event_id` | string | ID único do evento. Formato `evt_XXXXXX`. Imutável. |
| `timestamp` | ISO 8601 | Momento da decisão e apresentação da oferta. |
| `client_id` | string | ID sintético. **Não corresponde a nenhum ID do Kaggle.** Gerado independentemente. |
| `arm_id` | string | Braço selecionado. |
| `channel` | string | Canal de entrega. Herdado do braço selecionado. |
| `context` | objeto | **Variáveis brutas do cliente** — o que o algoritmo enxerga para tomar a decisão. Nunca contém segmentos processados. |
| `derived_segments` | array | Segmentos derivados das variáveis brutas pelo sistema. É o output da classificação, não o input. |
| `policy_version` | string | Versão do `arms_catalog.json` ativo no momento. |
| `selection_algorithm` | string | Algoritmo MAB usado. |
| `arm_probabilities` | objeto | **Probabilidade normalizada** que o Thompson Sampling atribuiu a cada braço. Soma = 1.0. Campo crítico para auditoria e reprodução. |
| `exploration_flag` | boolean | `true` = evento de exploração forçada (ε-greedy, 5% dos eventos). O braço pode não ser o que o algoritmo escolheria otimamente. |
| `random_seed` | integer | Semente usada para este evento específico. Permite reprodução exata. |
| `reward_expected` | float | Taxa esperada = `base_rate × ∏(multipliers dos segmentos ativos)`. |
| `reward_observed` | float/null | `1.0` = converteu, `0.0` = não converteu, `null` = ainda pendente. |
| `reward_timestamp` | ISO 8601 | Momento em que o reward foi registrado. |
| `status` | string | `pending` / `converted` / `not_converted` / `expired` |

**Pontos de Atenção:**

- O `context` contém as variáveis brutas (job, education, age_bucket...) — são os dados que o sistema recebe. O `derived_segments` é o que o algoritmo calculou a partir dessas variáveis — é o output da classificação.

---

## Arquivo 3: `delayed_rewards.json`

Cada linha é um **evento de recompensa**, isto é, o momento em que o sistema soube se a oferta converteu ou não. Sempre tem exatamente um registro por `event_id` do `offer_events`.

O arquivo existe separado do `offer_events`, porque em produção real, o reward chega dias depois da oferta. O sistema que processa a decisão (MAB) e o sistema que registra a conversão (CRM/core bancário) são sistemas diferentes, com tempos diferentes. O `offer_events` é escrito no momento T. O `delayed_rewards` é escrito no momento T + delay_days.

A decisão de incluir não-conversões foi importante: o arquivo original submetido continha apenas eventos positivos (reward = 1). Isso criaria um dataset enviesado e o Thompson Sampling precisaria tanto de sucessos quanto de falhas para atualizar corretamente os parâmetros Beta. Um arquivo que só registra conversões é como um médico que só documenta os pacientes que melhoram.

### Exemplo de um reward de conversão

```json
{
  "event_id": "evt_000001",
  "client_id": "client_04821",
  "arm_id": "arm_008",
  "reward_observed": 1,
  "reward_value": 1.0,
  "reward_timestamp": "2024-03-03T09:14:22Z",
  "delay_days_expected": 2,
  "delay_days_actual": 2,
  "reward_within_window": true,
  "conversion_type": "subscription",
  "segments_active": ["student_digital", "student_saver", "digital_channel",
                      "debt_free", "low_engagement", "high_season"],
  "expected_reward_at_decision": 0.8960,
  "reward_delta": 0.1040,
  "notes": "conversão em 2d via app"
}
```

### Exemplo de um reward de não-conversão

```json
{
  "event_id": "evt_000007",
  "client_id": "client_04827",
  "arm_id": "arm_003",
  "reward_observed": 0,
  "reward_value": 0.0,
  "reward_timestamp": "2024-03-17T11:30:00Z",
  "delay_days_expected": 14,
  "delay_days_actual": 14,
  "reward_within_window": true,
  "conversion_type": "no_conversion",
  "segments_active": ["digital_channel"],
  "expected_reward_at_decision": 0.088,
  "reward_delta": -0.088,
  "notes": "sem conversão após 14d de espera"
}
```

### Dicionário de campos

| Campo | Tipo | O que significa |
|---|---|---|
| `event_id` | string | Liga este reward ao evento de oferta correspondente. |
| `client_id` | string | Mesmo ID do offer_event. |
| `arm_id` | string | Braço que gerou este reward. |
| `reward_observed` | integer | `1` = converteu, `0` = não converteu. Inteiro para facilitar soma direta no update do Thompson Sampling. |
| `reward_value` | float | Mesmo valor em float. Reservado para rewards fracionários futuros (ex: valor da aplicação normalizado). |
| `reward_timestamp` | ISO 8601 | Momento do registro do reward. |
| `delay_days_expected` | integer | Delay documentado no `arms_catalog.json` para este braço. |
| `delay_days_actual` | integer | Delay real observado. Em dados sintéticos = esperado. Em produção, pode divergir. |
| `reward_within_window` | boolean | `true` = reward chegou dentro do prazo esperado. `false` = chegou fora (tardio ou antecipado). |
| `conversion_type` | string | `subscription` = contratou o produto. `no_conversion` = não converteu. Extensível para tipos parciais. |
| `segments_active` | array | Segmentos que estavam ativos no momento da decisão. Replica o `derived_segments` do offer_event para facilitar análise sem join. |
| `expected_reward_at_decision` | float | Taxa esperada no momento da decisão (do offer_event). |
| `reward_delta` | float | `reward_observed - expected_reward`. Positivo = converteu melhor que esperado. Negativo = pior. |
| `notes` | string | Texto livre descritivo. |

---

## Modelagem dos delayed rewards e horizonte temporal

### O problema do delayed reward em MAB

Em um MAB clássico, o algoritmo recebe o reward imediatamente após apresentar o braço. Em contexto bancário, isso não acontece: você mostra a oferta hoje e só sabe o resultado em dias.

Se o algoritmo atualizar suas crenças antes do reward chegar, vai usar informação incompleta — vai subestimar a taxa real do braço porque ainda não processou as conversões pendentes.

### Como modelamos o delay

Cada braço tem um `reward_delay_days` documentado no `offer_catalog.json`. O algoritmo deve:

1. Registrar o `offer_event` imediatamente com `status = "pending"`
2. **Não** atualizar os parâmetros Beta do Thompson Sampling até receber o `delayed_reward`
3. Após `delay_days_expected` dias, processar o `delayed_reward` e atualizar

```
Tempo T:       oferta apresentada → offer_event gravado (status: pending)
Tempo T+2d:    reward do arm_005 (CDB liquidez) → delayed_reward gravado → Beta atualizado
Tempo T+7d:    reward do arm_001 (CDB 12m) → delayed_reward gravado → Beta atualizado
Tempo T+14d:   reward do arm_003 (Previdência) → delayed_reward gravado → Beta atualizado
```

### Horizonte temporal do experimento

| Braço | Delay | Justificativa |
|---|---|---|
| `arm_control` | 0 dias | Sem ação ativa — sem delay |
| `arm_005` (CDB liquidez) | 2 dias | Adesão imediata via app |
| `arm_008` (Poupança estudantil) | 2 dias | Produto 100% digital |
| `arm_002` (Poupança digital) | 3 dias | Ativação automática + confirmação |
| `arm_004` (LCI/LCA 90d) | 5 dias | Processamento similar ao CDB, prazo menor |
| `arm_001` (CDB 12m) | 7 dias | Transferência + processamento interno |
| `arm_006` (Reengajamento) | 10 dias | Ciclo completo de reativação |
| `arm_007` (CDB 24m premium) | 10 dias | Ciclo de decisão longo do cliente |
| `arm_003` (Previdência PGBL) | 14 dias | Análise, contrato, carência inicial |

**Implicação para o horizonte do experimento:** o experimento precisa rodar por no mínimo **14 dias** antes de ter o primeiro conjunto completo de rewards. Para acumular dados suficientes para convergência do Thompson Sampling (estimativa: 500+ observações por braço), o horizonte recomendado é **90 dias**.

### Status de expiração

Um evento passa para `status = "expired"` quando `delay_days_actual > delay_days_expected × 1.5`. Isso sinaliza que o reward chegou fora da janela esperada — pode indicar problema no pipeline de dados, não necessariamente não-conversão. Em dados sintéticos, todos os rewards chegam dentro da janela.

---

## Como o Thompson Sampling usa esses dados

O Thompson Sampling mantém uma distribuição Beta(α, β) por braço. Para cada evento:

```python
# Antes da decisão: amostrar de Beta(α, β)
sample = random.betavariate(alpha, beta)

# Após receber o delayed_reward:
if reward_observed == 1:
    alpha += 1   # conversão: atualiza o parâmetro de sucesso
else:
    beta  += 1   # não-conversão: atualiza o parâmetro de falha
```

O estado inicial é Beta(1, 1) — distribuição uniforme, o algoritmo não assume nada antes de ver dados. Os `base_conversion_rate` do catálogo podem ser usados como priors mais informativos: Beta(α₀, β₀) onde α₀/(α₀+β₀) = base_rate.

---

## Como reproduzir os dados sintéticos

```bash
# Clonar o repositório
git clone <repo>

# Instalar dependências (apenas stdlib Python, sem dependências externas)
python --version

# Gerar os arquivos sintéticos
python src/datathon/simulator.py

# Verificar output
wc -l data/synthetic_enrichment/offer_events.json      # deve ser 200
wc -l data/synthetic_enrichment/delayed_rewards.json   # deve ser 200

# Validar primeiro evento
head -1 data/synthetic_enrichment/offer_events.json | python -m json.tool
```

Com `BASE_SEED = 42`, o output é determinístico — os mesmos 200 eventos serão gerados toda vez.

Para alterar o volume:
```python
# em generate_synthetic.py, linha 16:
N_EVENTS = 200  # altere para o número desejado
```

---

## Sementes aleatórias controladas

Todo elemento aleatório do experimento usa sementes derivadas de `BASE_SEED = 42`:

| Componente | Semente | Código |
|---|---|---|
| Sequência global | `BASE_SEED = 42` | `random.Random(42)` |
| Evento `i` | `BASE_SEED + i` | `random.Random(42 + i)` |
| Thompson Sampling do evento `i` | `BASE_SEED + i + 10000` | `random.Random(10042 + i)` |

Para reproduzir qualquer evento específico:
```python
import random
event_index = 0          # evt_000001
event_rng = random.Random(42 + event_index)
ts_rng    = random.Random(10042 + event_index)
# event_rng: gera contexto do cliente e resultado de conversão
# ts_rng: gera amostras Beta do Thompson Sampling
```

---

## Relação entre os arquivos

```
offer_catalog.json
    ↓ define braços, taxas, delays, segmentos
generate_synthetic.py
    ↓ lê o catálogo + BASE_SEED = 42
    ↓ gera eventos com Thompson Sampling simulado
offer_events.jsonl          delayed_rewards.jsonl
    ↓ momento da decisão         ↓ momento do resultado
    └──────── ligados por event_id ────────┘
```

O `offer_events` é o log de ações do algoritmo. O `delayed_rewards` é o log de resultados do mundo. A separação entre os dois é o coração da modelagem de delayed reward.

### Referências

- Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier.
- Chapelle, O., & Li, L. (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS.
- Rossi, P., McCulloch, R., & Allenby, G. (1996). *The Value of Purchase History Data in Target Marketing.* Journal of Marketing Research.
- Russo, D., Van Roy, B., Kazerouni, A., & Osband, I. (2018). *A Tutorial on Thompson Sampling.* Foundations and Trends in Machine Learning.