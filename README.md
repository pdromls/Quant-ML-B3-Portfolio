# Cross-Sectional Machine Learning e Backtesting Orientado a Eventos na B3 (IBrX 50)

Um pipeline quantitativo em Python para seleção cross-sectional de ações no mercado financeiro brasileiro (B3 — Universo IBrX 50, 2012–2026), acompanhado de um terminal analítico web interativo desenvolvido em Streamlit e um relatório executivo em Jupyter Notebook.

O foco metodológico do projeto concentra-se na integridade científica:
- Engenharia de atributos estritamente causal (sem viés de antecipação ou look-ahead bias).
- Padronização diária cross-sectional via Z-Score com winsorização [-4, +4], neutralizando o viés macroeconômico direcional e isolando o alpha relativo.
- Divisão temporal cronológica 70/20/10 com purga e embargo temporal de 21 dias úteis nas fronteiras de Treino, Validação e Teste, impedindo vazamento de dados futuros (data leakage).
- Simulação Walk-Forward ao longo de 12 anos (2015–2026, totalizando 142 rebalanceamentos mensais) para testar empiricamente o fenômeno de concept drift, confrontando modelos com retreino periódico anual contra modelos estáticos (congelados).
- Política de alocação suave (Soft Weighting Top 15), eliminando a sensibilidade a empates e o efeito falésia (cliff edge) inerente ao corte rígido tradicional (Hard Cut Top 10).
- Comparação sistemática com os dois benchmarks oficiais de mercado: o Buy & Hold do Universo (média equiponderada 1/N de todas as ações elegíveis) e o Ibovespa (ETF BOVA11).

---

## Guia Completo de Uso do Terminal Web (Streamlit)

O projeto disponibiliza uma interface visual analítica desenvolvida em Streamlit (`app.py`), permitindo auditoria forense, simulação e acompanhamento de carteiras recomendadas.

### 1. Inicialização da Aplicação
No terminal do seu ambiente (com as dependências instaladas), execute:
```bash
streamlit run app.py
```
A aplicação abrirá automaticamente no navegador em `http://localhost:8501`.

### 2. Controles Globais na Barra Lateral (Sidebar)
A barra lateral permite configurar os parâmetros mestres de toda a sessão:
- Fonte de Dados:
  - Base Local Higienizada (2012-2026): Opção padrão e recomendada. Carrega instantaneamente a base de dados tratada de 41 ações do IBrX 50, contendo 3.667 pregões auditados até a data presente.
  - API Yahoo Finance (Ao Vivo): Permite sincronizar cotações em tempo real da B3. Conta com o botão "Atualizar Cotações via API" para baixar novas cotações e "Resetar API" para restaurar a base local.
- Modelo de Machine Learning:
  - Regressão Ridge: Modelo linear regularizado com penalidade L2.
  - XGBoost: Algoritmo de gradient boosted trees com divisão por profundidade (depth-wise).
  - LightGBM: Algoritmo de gradient boosted trees com divisão ótima por folha (leaf-wise).
- Política de Ponderação de Carteira:
  - Soft Weighting (Top 15 com decaimento linear): Aloca pesos decrescentes proporcionais (12,50% para o 1º colocado até 0,83% para o 15º), mitigando sensibilidade a empates e empurrando o retorno para +1.568,85% com Sharpe 1,02.
  - Hard Cut (Top 10 com Margem de Tolerância): Aloca 10% fixos para as 10 melhores ações, aplicando uma margem de tolerância no ranking (filtro anti-giro) para evitar compras e vendas desnecessárias por variações mínimas de score.

---

### 3. Navegação entre as Abas do Terminal

#### Aba 1: Visão Geral e Curvas Históricas Acumuladas (2015–2026)
Objetivo: Avaliar a performance de longo prazo das estratégias contra o mercado.
- Gráfico Patrimonial Composto: Exibe a curva de capital acumulada comparando os três modelos de Machine Learning (Ridge, XGBoost e LightGBM) contra os dois índices de mercado (Buy & Hold 1/N e Ibovespa BOVA11).
- Tabela Executiva de Métricas Institucionais: Apresenta lado a lado métricas auditadas de Retorno Acumulado (%), CAGR (% a.a.), Volatilidade Anualizada (%), Índice de Sharpe, Drawdown Máximo (%), Taxa de Sucesso Mensal (%) e Alpha Excedente vs. Mercado.
- Controles de Período: Permite ajustar a janela temporal de visualização entre 2015 e 2026.

#### Aba 2: Previsão vs. Realidade (Auditoria de Decisões e Calendário de Pregões)
Objetivo: Inspecionar exatamente o que cada modelo recomendou comprar e verificar o que aconteceu no mercado real.

Esta aba possui 3 modos de inspeção selecionáveis via radio button:

1. Visão Compacta por Ano (Histórico Completo 2015–2026):
   - Seletor de Ano: Permite escolher qualquer ano da série histórica (2015 a 2026, com o ano corrente selecionado por padrão).
   - Exibe 5 cartões KPI consolidados do ano escolhido (Ridge, XGBoost, LightGBM, Buy & Hold Universo e Ibovespa).
   - Apresenta o gráfico de evolução de capital ao longo daquele ano.
   - Apresenta gráfico de barras com os retornos mês a mês para auditar a consistência das previsões.
   - Tabela detalhada mês a mês com retorno do modelo, retorno do benchmark, Alpha e taxa de acerto.

2. Inspeção Pontual por Data Específica (Calendário de Pregões Diários):
   - Seletor de Calendário (DD/MM/AAAA): Permite clicar e abrir o calendário interativo para selecionar qualquer dia específico da série histórica (2015 a 2026).
   - Trava e Snap Inteligente de Dias Úteis: Se o usuário selecionar um sábado, domingo ou feriado da B3, o sistema exibe uma mensagem informativa e ajusta automaticamente o cálculo para o pregão imediatamente anterior com cotação oficial.
   - Barra de Navegação Rápida: Botões [Pregão Anterior], [Próximo Pregão] e [Hoje / Mais Recente] para avançar ou recuar dia a dia apenas entre pregões reais da bolsa.
   - Apuração de Cotações Reais: Compara o preço de compra no 1º pregão do mês (data base da recomendação T0) com o preço de fechamento no dia selecionado (T).
   - Alternador de Ordenação: Opção para visualizar o gráfico na ordem de recomendação do modelo (maior peso para o menor) ou ordenado do melhor desempenho para o pior.
   - Gráfico de Barras com Cores Semânticas: Ações com retorno superior ao benchmark são coloridas em verde; ações com retorno inferior são coloridas em vermelho. Inclui linhas de referência horizontais para Buy & Hold e Ibovespa.
   - Modo Comparativo Lado a Lado: Se a opção "Comparar os 3 Modelos Lado a Lado" for selecionada, o dashboard renderiza o confronto simultâneo de Ridge, XGBoost e LightGBM na data escolhida.

3. Auditoria Mês a Mês (Histórico Completo 2015–2026):
   - Navegação Rápida entre Meses: Botões [◀ Mês Anterior], [Próximo Mês ▶] e [Mês Mais Recente] para percorrer a história mês a mês sem necessidade de reabrir o seletor.
   - Seletor dropdown cobrindo todos os 142 meses da série temporal para análise individual detalhada.
   - Alternador de Ordenação: Permite alternar entre ordenação por recomendação do modelo (maior peso / rank) ou por melhor retorno percentual.

#### Aba 3: Explicabilidade e Importância Econômica dos Fatores
Objetivo: Compreender os fundamentos econômicos por trás das decisões dos modelos.
- Gráficos de Barras Horizontais com Nomes Intuitivos em Português:
  - Regressão Ridge: Coeficientes Beta Padronizados expressos em basis points (bps).
  - XGBoost: Importância por ganho de informação (Gain %) em nível de nó.
  - LightGBM: Importância por ganho percentual por folha (leaf-wise split).
- Mapeamento Claro dos 12 Indicadores Técnicos:
  - Volatilidade Realizada (21 dias)
  - Posição nas Bandas de Bollinger
  - Momentum Trimestral (Retorno 63d)
  - Retorno Mensal (21 dias)
  - Retorno Quinzenal (10 dias)
  - Retorno Semanal (5 dias)
  - Retorno Diário (1 dia)
  - RSI - Força Relativa (14 dias)
  - Inclinação entre Médias (SMA 20/50)
  - Distância da Média Móvel de 50 dias
  - Distância da Média Móvel de 20 dias
  - Aceleração de Volume (5d vs. 21d)
- Interpretação Financeira dos Fatores:
  - ret_63d positivo: Captura o efeito Momentum trimestral da literatura quantitativa clássica.
  - ret_1d e bb_pos negativos: Refletem o filtro de sobrecompra e reversão de curto prazo, penalizando ativos sobrecomprados.
  - vol_21d: Principal variável de calibração do prêmio de risco nos modelos de árvore.

#### Aba 4: Data Quality Assurance e Validação de Integridade
Objetivo: Auditar a base de dados subjacente.
- Botão "Executar Auditoria de Dados Agora": Dispara um teste automatizado formal em todas as matrizes de preço e volume.
- Validação em tempo real de ausência de valores nulos, infinitos, preços não positivos e sincronismo temporal (status: 100% aprovado).

---

## Metodologia Quantitativa

### 1. Higienização de Dados Reais da B3 (data_loader.py)
- Filtro de Feriados Espúrios: Remoção de datas com menos de 20 ações ativas (feriados municipais ou dias sem pregão oficial da B3).
- Máscara de Pré-Listagem: Ativos com volume zerado antes do início de sua liquidez real no IBrX 50 (ex.: SUZB3 e ENGI11) são mascarados como NaN, impedindo falsos retornos de 0%.
- Ajuste Contínuo: Preços de abertura e fechamento ajustados pelo fator acumulado de dividendos, juros sobre capital próprio (JCP) e desdobramentos.

### 2. Engenharia de Atributos Causal e Padronização Cross-Sectional (features.py)
- 12 Indicadores Técnicos: Retornos defasados (1d, 5d, 10d, 21d, 63d), volatilidade realizada (21d), distâncias e razão de médias móveis (SMA 20/50), posição nas Bandas de Bollinger, RSI normalizado em [-1, +1] e volume relativo.
- Z-Score Cross-Sectional: Padronização relativa das ações a cada pregão (Z = (X - mu) / sigma), com corte rígido (winsorize) em [-4, +4] para neutralizar o viés direcional diário da bolsa e isolar o alpha relativo.

### 3. Divisão Temporal com Purga e Embargo de 21 Dias (splits.py)
- Como o objetivo preditivo é o excesso de retorno nos próximos 21 pregões úteis (t -> t+21), uma divisão cronológica simples sofreria vazamento temporal nas fronteiras.
- Foi aplicada purga matemática de 21 dias no final do Treino e no final da Validação, garantindo isolamento total entre os conjuntos:
  - Treino (70%): Abril de 2015 a Agosto de 2022
  - Validação (20%): Setembro de 2022 a Outubro de 2024 (seleção de hiperparâmetros)
  - Teste Puro (10%): Novembro de 2024 a Novembro de 2025 (avaliação estritamente out-of-sample)

### 4. Algoritmos de Machine Learning Avaliados (models.py)
- Regressão Linear Ridge: Solução analítica fechada (X^T X + lambda I)^(-1) X^T y com regularização L2 e penalidade ótima lambda.
- XGBoost: Gradient Boosted Trees com divisão simétrica por nível (depth-wise).
- LightGBM: Gradient Boosted Trees com divisão ótima por folha (leaf-wise).

### 5. Simulação Walk-Forward com Retreino Periódico (walk_forward.py)
- Avaliação empírica do fenômeno de Concept Drift ao longo de 12 anos (2015–2026, 142 rebalanceamentos mensais):
  - Modelos Dinâmicos: Retreinados anualmente com janela móvel de 756 pregões úteis (3 anos).
  - Modelos Congelados: Calibrados apenas no início da série (2012–2014) e operados sem retreino por mais de uma década.
  - Benchmarks: Buy & Hold do Universo (1/N) e ETF Ibovespa (BOVA11).

---

## Resultados Experimentais

### 1. Métricas de Machine Learning (Treino 70% / Validação 20% / Teste Puro 10%)

| Modelo | Partição | Amostras | RMSE | MAE | R^2 OOS (%) | IC Pearson | Rank IC (Spearman) | Acurácia Direcional (%) | AUC-ROC | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RIDGE | 1. Treino (70%) | 13.265 | 0,0989 | 0,0693 | +1,09% | +0,1045 | +0,0582 | 52,83% | 0,5316 | 0,4657 |
| RIDGE | 2. Validação (20%) | 4.063 | 0,0799 | 0,0582 | -2,98% | -0,0574 | -0,0470 | 48,14% | 0,4820 | 0,4410 |
| RIDGE | 3. Teste Puro (10%) | 2.171 | 0,0883 | 0,0634 | -2,10% | -0,0348 | -0,0175 | 47,95% | 0,4861 | 0,4350 |
| XGBOOST | 1. Treino (70%) | 13.265 | 0,0980 | 0,0688 | +2,95% | +0,1908 | +0,1047 | 54,55% | 0,5547 | 0,4545 |
| XGBOOST | 2. Validação (20%) | 4.063 | 0,0798 | 0,0580 | -2,60% | -0,0138 | -0,0182 | 50,14% | 0,4973 | 0,4185 |
| XGBOOST | 3. Teste Puro (10%) | 2.171 | 0,0886 | 0,0638 | -2,77% | -0,0389 | -0,0220 | 46,71% | 0,4866 | 0,3783 |
| LIGHTGBM | 1. Treino (70%) | 13.265 | 0,0955 | 0,0674 | +7,71% | +0,3412 | +0,2166 | 57,69% | 0,6028 | 0,4901 |
| LIGHTGBM | 2. Validação (20%) | 4.063 | 0,0800 | 0,0582 | -3,28% | +0,0004 | -0,0099 | 49,94% | 0,4985 | 0,4162 |
| LIGHTGBM | 3. Teste Puro (10%) | 2.171 | 0,0890 | 0,0639 | -3,87% | -0,0267 | -0,0052 | 48,13% | 0,5003 | 0,3953 |

Diagnóstico do Trade-off Viés-Variância:
- O LightGBM apresentou alta capacidade de memorização no treino (R^2 = +7,71% e Rank IC = 0,2166), mas sofreu redução de desempenho no teste estático devido à baixa razão sinal-ruído das séries financeiras.
- A Regressão Ridge, com regularização L2 estrutural (lambda = 10), evitou o sobreajuste e obteve a menor degradação de erro fora da amostra (RMSE = 0,0883).

---

### 2. Simulação Walk-Forward Multi-Ano: Retreino Periódico vs. Modelos Congelados (2015–2026)

Resultados apurados em 142 rebalanceamentos mensais com retreino móvel anual:

| Estratégia | Modo de Execução | Retorno Acumulado (%) | CAGR (% a.a.) | Volatilidade (% a.a.) | Sharpe Ratio | Max Drawdown (%) | Alpha vs. Universo (% a.a.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Buy & Hold (Universo 1/N) | Benchmark Passivo | +615,00% | 19,74% | 24,26% | 0,81 | -37,16% | 0,00% |
| Ibovespa (BOVA11) | Benchmark Ponderado | +245,30% | 11,28% | 26,14% | 0,73 | -44,21% | -8,46% |
| ML: RIDGE (Walk-Forward) | Retreino Anual (12m) | +1.415,50% | 28,28% | 27,93% | 1,01 | -39,34% | +8,54% |
| ML: XGBOOST (Walk-Forward) | Retreino Anual (12m) | +827,44% | 22,63% | 29,17% | 0,78 | -45,89% | +2,89% |
| ML: LIGHTGBM (Walk-Forward) | Retreino Anual (12m) | +777,54% | 22,01% | 29,27% | 0,75 | -41,76% | +2,27% |
| ML: RIDGE (Congelado) | Estático (Sem Retreino) | +590,71% | 19,37% | 21,87% | 0,89 | -36,71% | -0,37% |
| ML: XGBOOST (Congelado) | Estático (Sem Retreino) | +563,51% | 18,93% | 23,59% | 0,80 | -41,24% | -0,81% |
| ML: LIGHTGBM (Congelado) | Estático (Sem Retreino) | +828,28% | 22,64% | 22,75% | 1,00 | -37,54% | +2,90% |

---

### 3. Políticas de Portfólio: Hard Cut (Top 10) vs. Soft Weighting (Top 15)

| Modelo | Política de Ponderação | Retorno Acumulado (%) | CAGR (% a.a.) | Volatilidade (% a.a.) | Sharpe Ratio | Max Drawdown (%) | Alpha vs. Universo (% a.a.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Buy & Hold (Universo) | Equiponderado (1/N) | +615,00% | 19,74% | 24,26% | 0,81 | -37,16% | 0,00% |
| ML: RIDGE | Hard Cut (Top 10 com Margem de Tolerância) | +1.415,50% | 28,28% | 27,93% | 1,01 | -39,34% | +8,54% |
| ML: RIDGE | Soft Weighting (Top 15 Rank Linear) | +1.568,85% | 29,41% | 28,81% | 1,02 | -35,23% | +9,67% |
| ML: XGBOOST | Hard Cut (Top 10 com Margem de Tolerância) | +827,44% | 22,63% | 29,17% | 0,78 | -45,89% | +2,89% |
| ML: XGBOOST | Soft Weighting (Top 15 Rank Linear) | +862,40% | 23,05% | 28,59% | 0,81 | -40,82% | +3,31% |
| ML: LIGHTGBM | Hard Cut (Top 10 com Margem de Tolerância) | +777,54% | 22,01% | 29,27% | 0,75 | -41,76% | +2,27% |
| ML: LIGHTGBM | Soft Weighting (Top 15 Rank Linear) | +670,39% | 20,57% | 28,74% | 0,72 | -42,05% | +0,83% |

Vantagens da Ponderação Suave (Soft Weighting):
- Redução do Risco Extremo: O Max Drawdown do XGBoost recua de -45,89% para -40,82%, e o da Regressão Ridge recua de -39,34% para -35,23%.
- Eliminação do Efeito Falésia (Cliff Edge): Em vez de alocar 10% para o 10º colocado e 0% para o 11º, o peso decai linearmente (12,5% -> 11,7% -> ... -> 0,8%), neutralizando a sensibilidade a empates e à aleatoriedade estocástica nas divisões de nós de árvores.

---

### 4. Relatório Visual em Jupyter Notebook (notebooks/01_ml_quant_strategies_b3.ipynb)

Para manter a documentação no README limpa, leve e focada na arquitetura, todas as visualizações gráficas geradas pelo pipeline estão reunidas no Jupyter Notebook demonstrativo (`notebooks/01_ml_quant_strategies_b3.ipynb`) e disponíveis de forma dinâmica no terminal Streamlit (`app.py`).

Ao abrir e executar o notebook, você tem acesso a:
- Importância dos atributos técnicos (coeficientes Beta em basis points para Ridge e ganho de informação para XGBoost e LightGBM).
- Resumo gráfico das métricas de Machine Learning na partição de Teste Puro Out-of-Sample (RMSE, MAE, R², IC e Rank IC).
- Curva de excesso de retorno acumulado (Alpha) da seleção Top 10 frente ao Buy & Hold do Universo.
- Comparativo multi-ano de Walk-Forward com retreino periódico anual vs. modelos congelados ao longo de 12 anos.
- Gráficos comparativos entre as políticas de construção de portfólio (Hard Cut com Margem de Tolerância vs. Soft Weighting Top 15).

---

## Estrutura do Repositório

```text
quant-ml-b3-portfolio/
├── app.py                                 # Terminal Web Interativo em Streamlit
├── Dockerfile                             # Imagem containerizada otimizada (Python 3.11 + libgomp1)
├── docker-compose.yml                     # Orquestrador para execução com 1 comando
├── .dockerignore                          # Filtro de arquivos para o build do container
├── README.md                              # Documentação técnica completa
├── pyproject.toml                         # Configurações de empacotamento moderno e pytest
├── requirements.txt                       # Dependências do projeto
├── .gitignore                             # Filtro de arquivos locais e temporários
├── data/
│   └── processed/                         # Cotações históricas tratadas (41 ações IBrX 50 + BOVA11/BRAX11)
├── src/
│   └── quant_ml_b3/
│       ├── __init__.py
│       ├── data_loader.py                 # Leitura local de matrizes e download via Yahoo Finance
│       ├── features.py                    # 12 indicadores técnicos causais e Z-Score cross-sectional
│       ├── splits.py                      # Divisão cronológica 70/20/10 com purga temporal (21 dias)
│       ├── models.py                      # Algoritmos Ridge, XGBoost e LightGBM
│       ├── metrics.py                     # Métricas quantitativas (IC, Rank IC, Sharpe, Drawdown, Alpha)
│       ├── walk_forward.py                # Motor de simulação Walk-Forward multi-ano
│       └── plotting.py                    # Gráficos para relatórios
├── scripts/
│   ├── run_experiment.py                  # CLI para rodar o pipeline clássico 70/20/10 e Walk-Forward
│   ├── export_dashboard_data.py           # Pipeline que gera as tabelas completas para o Streamlit
│   └── update_market_data_to_today.py     # Script para sincronizar dados de mercado até a data corrente
├── notebooks/
│   └── 01_ml_quant_strategies_b3.ipynb    # Jupyter Notebook executado com análise passo a passo
├── outputs/
│   ├── figures/                           # Visualizações estáticas em PNG
│   └── tables/                            # Séries temporais, carteiras e retornos em CSV
└── tests/                                 # Suíte automatizada de testes com pytest
    ├── test_features_and_splits.py        # Validação de causalidade, purga e Z-score
    ├── test_models_and_metrics.py         # Validação de convergência de modelos e métricas
    └── test_walk_forward.py               # Validação de motor e margem de tolerância do Walk-Forward
```

---

## Como Executar o Projeto

### Opção A: Execução via Docker (Recomendado / 1 Comando)
Caso possua o Docker instalado, basta executar no diretório raiz:
```bash
docker compose up
```
O container baixará as dependências, configurará o ambiente Linux com suporte multithread para LightGBM/XGBoost e disponibilizará o terminal web em `http://localhost:8501`.

### Opção B: Execução Local com Python

#### 1. Instalação das Dependências
```bash
git clone https://github.com/pdromls/quant-ml-b3-portfolio.git
cd quant-ml-b3-portfolio
pip install -r requirements.txt
```

#### 2. Executar o Terminal Web Streamlit
```bash
streamlit run app.py
```
O dashboard será aberto no navegador em `http://localhost:8501`.

#### 3. Abrir o Relatório em Jupyter Notebook
```bash
jupyter notebook notebooks/01_ml_quant_strategies_b3.ipynb
```

#### 4. Executar os Testes Automatizados (pytest)
```bash
pytest -v
```

#### 5. Executar os Pipelines de Modelagem e Exportação
```bash
# Executar os experimentos de Machine Learning e simulação Walk-Forward:
python scripts/run_experiment.py

# Exportar dados atualizados para o dashboard:
python scripts/export_dashboard_data.py
```

---

## Autor

Pedro Magalhães  
Ciência da Computação  
GitHub: @pdromls
