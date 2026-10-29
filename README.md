# Cross-Sectional Machine Learning e Backtesting Orientado a Eventos na B3 (IBrX 50)

Um pipeline quantitativo em Python para seleção cross-sectional de ações no mercado financeiro brasileiro (B3 — Universo IBrX 50, 2012–2026), acompanhado de um terminal analítico web interativo desenvolvido em Streamlit.

O foco metodológico do projeto concentra-se na integridade científica:
- Engenharia de atributos estritamente causal (sem viés de antecipação ou look-ahead bias).
- Padronização diária cross-sectional via Z-Score com winsorização [-4, +4], neutralizando o viés macroeconômico direcional e isolando o alpha relativo.
- Divisão temporal cronológica 70/20/10 com purga e embargo temporal de 21 dias úteis nas fronteiras de Treino, Validação e Teste, impedindo vazamento de dados futuros (data leakage).
- Simulação Walk-Forward ao longo de 12 anos (2015–2026, totalizando 142 rebalanceamentos mensais) para testar empiricamente o fenômeno de concept drift, confrontando modelos com retreino periódico anual contra modelos estáticos (congelados).
- Política de alocação suave (Soft Weighting Top 15), eliminando a sensibilidade a empates e o efeito falésia (cliff edge) inerente ao corte rígido tradicional (Hard Cut Top 10).
- Comparação sistemática com os dois benchmarks oficiais de mercado: o Buy & Hold do Universo (média equiponderada 1/N de todas as ações elegíveis) e o Ibovespa (ETF BOVA11).

---

## Terminal Web Interativo (Streamlit)

O projeto inclui uma aplicação web analítica desenvolvida em Streamlit (`app.py`), permitindo auditoria visual, monitoramento de carteiras recomendadas e inspeção interativa de resultados.

### Execução da Aplicação
```bash
streamlit run app.py
```
O servidor local iniciará e disponibilizará o dashboard no endereço `http://localhost:8501`.

### Estrutura das Abas do Terminal

#### Aba 1: Visão Geral e Curvas Históricas Acumuladas (2015–2026)
- Gráfico interativo com a evolução do patrimônio líquido acumulado (curva composta) dos três modelos de Machine Learning (Regressão Ridge, XGBoost, LightGBM) frente aos benchmarks oficiais (Buy & Hold 1/N e Ibovespa BOVA11).
- Tabela executiva de métricas financeiras institucionais: Retorno Acumulado (%), Taxa Composta Anual de Crescimento (CAGR % a.a.), Volatilidade Anualizada (%), Índice de Sharpe, Drawdown Máximo (%), Taxa de Sucesso Mensal (%) e Alpha Anualizado vs. Universo e vs. Ibovespa.
- Controles laterais para alternar entre as políticas de ponderação (Soft Weighting Top 15 vs. Hard Cut Top 10) e recortar períodos temporais.

#### Aba 2: Previsão vs. Realidade (Auditoria de Decisões e Calendário de Pregões)
Esta aba disponibiliza ferramentas para auditar o que cada modelo previu em relação ao que efetivamente ocorreu no mercado:

1. Visão Compacta do Ano Corrente (YTD 2026):
   - Painel com 5 indicadores de retorno acumulado no ano de 2026 para os três modelos e os benchmarks de mercado.
   - Gráfico de rentabilidade mensal acumulada de janeiro a outubro de 2026.
   - Gráfico de barras agrupadas comparando mês a mês o retorno do modelo selecionado contra o Buy & Hold e o Ibovespa.
   - Tabela de auditoria mensal com o retorno realizado, retornos dos índices, Alpha mensal e taxa de acerto.

2. Inspeção Pontual por Data Específica (Calendário de Pregões Diários):
   - Seletor de data interativo em formato de calendário (DD/MM/AAAA) abrangendo todo o período de 2015 a 2026.
   - Validação automática de dias úteis da B3: caso uma data sem pregão (final de semana ou feriado) seja selecionada, o sistema detecta a indisponibilidade e ajusta automaticamente para o pregão imediatamente anterior com cotação oficial.
   - Barra de navegação rápida com os botões [Pregão Anterior], [Próximo Pregão] e [Hoje / Mais Recente], permitindo navegar exclusivamente entre dias de negociação real.
   - Apuração do preço de entrada no primeiro pregão do mês (T0) e do preço de fechamento no pregão selecionado (T).
   - Cálculo do retorno acumulado de cada ação até a data escolhida, da contribuição ponderada para o portfólio e do retorno dos benchmarks (Buy & Hold e Ibovespa) no mesmo intervalo.
   - Gráfico de barras com código de cores: barras verdes para ações que superaram o Buy & Hold do Universo e barras vermelhas para ações com retorno inferior ao índice de referência.
   - Tabela detalhada com rank, ticker, score previsto, peso percentual, preço de entrada, preço atual, retorno percentual e contribuição.
   - Suporte ao modo individual ou comparativo lado a lado dos três modelos no mesmo pregão.

3. Previsões Mês a Mês em 2026 e Histórico Multi-Ano Completo (2015–2026):
   - Seleção de qualquer mês da série histórica de 12 anos para inspecionar as recomendações geradas no primeiro dia útil e o resultado apurado no fechamento mensal.

#### Aba 3: Explicabilidade e Importância Econômica dos Fatores
- Análise de contribuição dos 12 atributos preditivos:
  - Regressão Ridge: Coeficientes Beta Padronizados expressos em basis points (bps).
  - XGBoost: Ganho de informação percentual (Gain %) nas divisões das árvores.
  - LightGBM: Ganho percentual por folha (leaf-wise split).
- Interpretação econômica e financeira dos coeficientes:
  - ret_63d (positivo): Captura o momentum trimestral clássico.
  - ret_21d (negativo): Reflete a reversão à média mensal (short-term reversal), penalizando ativos que subiram excessivamente no curto prazo.
  - vol_21d (negativo): Penaliza ativos com volatilidade extrema, reduzindo o risco de cauda e drawdowns severos.

#### Aba 4: Data Quality Assurance e Validação de Integridade
- Ferramenta de auditoria formal das matrizes de cotação da B3.
- Verificação automatizada de ausência de nulos, valores infinitos, preços não positivos e integridade de datas (base 100% aprovada).

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
| ML: RIDGE | Hard Cut (Top 10 1/N com Histerese) | +1.415,50% | 28,28% | 27,93% | 1,01 | -39,34% | +8,54% |
| ML: RIDGE | Soft Weighting (Top 15 Rank Linear) | +1.568,85% | 29,41% | 28,81% | 1,02 | -35,23% | +9,67% |
| ML: XGBOOST | Hard Cut (Top 10 1/N com Histerese) | +827,44% | 22,63% | 29,17% | 0,78 | -45,89% | +2,89% |
| ML: XGBOOST | Soft Weighting (Top 15 Rank Linear) | +862,40% | 23,05% | 28,59% | 0,81 | -40,82% | +3,31% |
| ML: LIGHTGBM | Hard Cut (Top 10 1/N com Histerese) | +777,54% | 22,01% | 29,27% | 0,75 | -41,76% | +2,27% |
| ML: LIGHTGBM | Soft Weighting (Top 15 Rank Linear) | +670,39% | 20,57% | 28,74% | 0,72 | -42,05% | +0,83% |

Vantagens da Ponderação Suave (Soft Weighting):
- Redução do Risco Extremo: O Max Drawdown do XGBoost recua de -45,89% para -40,82%, e o da Regressão Ridge recua de -39,34% para -35,23%.
- Eliminação do Efeito Falésia (Cliff Edge): Em vez de alocar 10% para o 10º colocado e 0% para o 11º, o peso decai linearmente (12,5% -> 11,7% -> ... -> 0,8%), neutralizando a sensibilidade a empates e à aleatoriedade estocástica nas divisões de nós de árvores.

---

## Estrutura do Repositório

```text
quant-ml-b3-portfolio/
├── app.py                                 # Terminal Web Interativo em Streamlit
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
│   └── 01_ml_quant_strategies_b3.ipynb    # Jupyter Notebook demonstrativo pré-executado
├── outputs/
│   ├── figures/                           # Visualizações estáticas em PNG
│   └── tables/                            # Séries temporais, carteiras e retornos em CSV
└── tests/                                 # Suíte automatizada de testes com pytest
    ├── test_features_and_splits.py        # Validação de causalidade, purga e Z-score
    ├── test_models_and_metrics.py         # Validação de convergência de modelos e métricas
    └── test_walk_forward.py               # Validação de motor e histerese do Walk-Forward
```

---

## Como Executar o Projeto

### 1. Instalação das Dependências
```bash
git clone https://github.com/pdromls/quant-ml-b3-portfolio.git
cd quant-ml-b3-portfolio
pip install -r requirements.txt
```

### 2. Executar o Terminal Web Streamlit
```bash
streamlit run app.py
```
O dashboard será aberto no navegador em `http://localhost:8501`.

### 3. Executar os Testes Automatizados (pytest)
```bash
pytest -v
```

### 4. Executar os Pipelines de Modelagem e Exportação
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
