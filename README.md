# Quant Alpha Terminal & Cross-Sectional Machine Learning na B3 (IBrX 50)

![Python](https://img.shields.io/badge/Python-3.10%2B-2563EB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F51B5?logo=plotly&logoColor=white)
![Modelos](https://img.shields.io/badge/Modelos-Ridge%20%7C%20XGBoost%20%7C%20LightGBM-059669)
![Testes](https://img.shields.io/badge/Testes-Pytest%20Passing-10B981)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-D97706)

Um pipeline quantitativo profissional de **Ciência de Dados, Machine Learning e Backtesting Orientado a Eventos** em Python para seleção *cross-sectional* de ações no mercado financeiro brasileiro (**B3 — Universo IBrX 50**, 2012–2026), acompanhado de um **Terminal Web Interativo em Streamlit**.

O foco central deste projeto é a **integridade metodológica e científica**: engenharia de atributos estritamente causal (livre de *look-ahead bias*), divisão temporal cronológica com purga e embargo temporal para eliminar vazamento de rótulos futuros (*data leakage*), mitigação de *concept drift* via simulação **Walk-Forward com Retreino Periódico ao longo de 12 anos (142 rebalanceamentos mensais)** e confronto com os dois benchmarks oficiais de mercado: o **Buy & Hold do Universo (1/N)** e o **Ibovespa (BOVA11)**.

---

## 🖥️ Terminal Web Interativo (`app.py`)

O repositório inclui um dashboard analítico profissional construído em **Streamlit** e **Plotly**, projetado para monitoramento, auditoria e exploração interativa de todas as previsões e resultados das carteiras quantitativas.

### Como Executar o Terminal Localmente:
```bash
streamlit run app.py
```

### Funcionalidades do Dashboard:

#### 1. 📈 Aba 1: Visão Geral & Curvas Históricas Acumuladas (2015–2026)
* **Comparação Patrimonial:** Traçado interativo da curva de evolução de capital de cada modelo (**Regressão Ridge**, **XGBoost**, **LightGBM**) contra os dois benchmarks oficiais (**Buy & Hold 1/N** e **Ibovespa BOVA11**).
* **Tabela de Métricas Financeiras Institucionais:** Exibição lado a lado de **Retorno Acumulado (%)**, **CAGR (% a.a.)**, **Volatilidade Anualizada (%)**, **Índice de Sharpe**, **Drawdown Máximo (%)**, **Taxa de Sucesso Mensal (%)** e **Alpha Anualizado vs. Mercado**.
* **Filtros Dinâmicos:** Seletor de período temporal (2015 a 2026) e alternador entre políticas de alocação (**Soft Weighting Top 15** vs. **Hard Cut Top 10**).

#### 2. 🎯 Aba 2: Previsão vs. Realidade (Auditoria de Decisões & Calendário de Pregões)
* **⚡ Visão Compacta do Ano Corrente (YTD 2026):**
  * 5 KPIs consolidados de retorno acumulado no ano de 2026 (Ridge, XGBoost, LightGBM, Buy & Hold Universo e Ibovespa).
  * Gráfico de evolução de rentabilidade composta ao longo de 2026.
  * Gráfico de barras agrupadas comparando os retornos mensais do modelo escolhido vs. mercado.
  * Tabela de auditoria do ano com apuração de Alpha mensal e taxas de acerto.
* **📅 Inspecionar por Data Específica (Calendário de Pregões Diários):**
  * **Menu de Calendário (`DD/MM/AAAA`):** Permite selecionar qualquer dia específico para inspecionar como a carteira recomendada no início do mês estava performando exatamente naquele pregão.
  * **Validação Inteligente de Pregões da B3:** Identifica automaticamente se a data selecionada foi um dia útil de bolsa. Caso o usuário clique em um final de semana ou feriado, o sistema exibe um alerta e ajusta automaticamente (*snap*) para o pregão imediatamente anterior com cotação oficial.
  * **Barra de Navegação Rápida:** Botões `[◀ Pregão Anterior]`, `[Próximo Pregão ▶]` e `[Hoje / Mais Recente]` para avançar ou recuar dia a dia apenas entre pregões reais.
  * **Preços e Rentabilidade Real:** Compara o preço de compra no 1º dia útil do mês ($T_0$) com a cotação no dia selecionado ($T$), calculando o retorno individual de cada ativo, a contribuição ponderada e se superou o Buy & Hold do Universo e o Ibovespa.
  * **Gráfico de Barras com Cores Semânticas:** Barras verdes para ações que superaram o mercado e vermelhas para as que ficaram abaixo, com linhas horizontais de referência para Buy & Hold e Ibovespa.
  * **Modo Comparativo Lado a Lado:** Permite confrontar a carteira dos 3 modelos simultaneamente na data escolhida.
* **🔴 Previsões Mês a Mês em 2026 & 📜 Histórico Multi-Ano Completo (2015–2026):**
  * Inspeção forense de qualquer mês dos 12 anos de histórico, visualizando a alocação emitida no 1º pregão e o resultado no fechamento do período.

#### 3. 🧠 Aba 3: Explicabilidade & Importância Econômica dos Fatores
* Decomposição da contribuição de cada um dos 12 indicadores técnicos:
  * **Regressão Ridge:** Coeficientes Beta Padronizados em *basis points* (bps).
  * **XGBoost:** Ganho de informação relativo (% de *Gain* por nível de divisão).
  * **LightGBM:** Ganho de informação por folha (*leaf-wise split*).
* Interpretação quantitativa dos fatores: comprovação empírica do efeito **Momentum Trimestral (`ret_63d`)** e da **Reversão à Média Mensal (`ret_21d`)**.

#### 4. 🛡️ Aba 4: Data Quality Assurance & Validação de Integridade
* Painel de diagnóstico em tempo real que audita formalmente a matriz de cotações da B3.
* Verifica ausência de valores nulos, infinitos, preços não positivos e integridade de datas (100% aprovado).

---

## 🎯 Destaques Metodológicos

1. **Higienização de Dados Reais da B3 (`data_loader.py`):**
   * **Filtro de Feriados Espúrios:** Remoção de datas com menos de 20 ações ativas (feriados municipais ou dias sem pregão oficial da B3).
   * **Máscara de Pré-Listagem:** Ativos com volume zerado antes do início de sua liquidez real no IBrX 50 (ex.: `SUZB3` e `ENGI11`) são mascarados como `NaN`, impedindo falsos retornos de 0%.
   * **Ajuste Contínuo:** Preços de abertura e fechamento ajustados pelo fator acumulado de dividendos, juros sobre capital próprio (JCP) e desdobramentos.

2. **Engenharia de Atributos Causal ($\le t$) & Padronização Cross-Sectional (`features.py`):**
   * **12 Indicadores Técnicos:** Retornos defasados (1d, 5d, 10d, 21d, 63d), volatilidade realizada (21d), distâncias e razão de médias móveis (SMA 20/50), posição nas Bandas de Bollinger, RSI normalizado em $[-1, +1]$ e volume relativo.
   * **Z-Score Cross-Sectional:** Padronização relativa das ações a cada pregão ($Z = \frac{X - \mu_{dia}}{\sigma_{dia}}$), com corte rígido (*winsorize*) em $[-4, +4]$ para neutralizar o viés direcional diário da bolsa e isolar o *alpha* relativo.

3. **Divisão Temporal com Purga / Embargo de 21 Dias (`splits.py`):**
   * Como o objetivo preditivo é o excesso de retorno nos **próximos 21 pregões úteis** ($t \to t+21$), uma divisão cronológica simples sofreria vazamento temporal nas fronteiras.
   * Aplicamos uma **purga matemática de 21 dias** no final do Treino e no final da Validação, garantindo isolamento total entre os conjuntos:
     * **Treino (70%):** Abr/2015 a Ago/2022
     * **Validação (20%):** Set/2022 a Out/2024 (seleção de hiperparâmetros)
     * **Teste Puro (10%):** Nov/2024 a Nov/2025 (avaliação estritamente *out-of-sample*)

4. **Comparação de Arquiteturas de ML (`models.py`):**
   * **Regressão Linear Ridge ($L_2$ Baseline):** Solução matricial fechada $(X^T X + \lambda I)^{-1} X^T y$ com seleção ótima do coeficiente de penalidade ($\lambda$).
   * **XGBoost:** *Gradient Boosted Trees* com divisão simétrica por nível (*depth-wise*).
   * **LightGBM:** *Gradient Boosted Trees* com divisão ótima por folha (*leaf-wise*).

5. **Simulação Walk-Forward Multi-Ano com Retreino Periódico (`walk_forward.py`):**
   * Avaliação do fenômeno de **Concept Drift** ao longo de 12 anos (2015–2026) comparando:
     * **Modelos Dinâmicos:** Retreinados anualmente com janela móvel de 756 pregões (3 anos).
     * **Modelos Congelados (Static):** Calibrados apenas no início (2012–2014) e operados sem retreino.
     * **Benchmarks de Mercado:** Buy & Hold do Universo (1/N) e ETF Ibovespa (BOVA11).

---

## 📊 Resultados e Avaliação dos Modelos

### 1. Métricas de Machine Learning (Treino 70% / Validação 20% / Teste Puro 10%)

| Modelo | Partição | Amostras | RMSE | MAE | $R^2$ OOS (%) | IC Pearson | Rank IC (Spearman) | Acurácia Direcional (%) | AUC-ROC | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RIDGE** | 1. Treino (70%) | 13.265 | 0,0989 | 0,0693 | +1,09% | +0,1045 | +0,0582 | 52,83% | 0,5316 | 0,4657 |
| **RIDGE** | 2. Validação (20%) | 4.063 | 0,0799 | 0,0582 | -2,98% | -0,0574 | -0,0470 | 48,14% | 0,4820 | 0,4410 |
| **RIDGE** | **3. Teste Puro (10%)** | 2.171 | **0,0883** | **0,0634** | **-2,10%** | **-0,0348** | **-0,0175** | **47,95%** | **0,4861** | **0,4350** |
| **XGBOOST** | 1. Treino (70%) | 13.265 | 0,0980 | 0,0688 | +2,95% | +0,1908 | +0,1047 | 54,55% | 0,5547 | 0,4545 |
| **XGBOOST** | 2. Validação (20%) | 4.063 | 0,0798 | 0,0580 | -2,60% | -0,0138 | -0,0182 | 50,14% | 0,4973 | 0,4185 |
| **XGBOOST** | **3. Teste Puro (10%)** | 2.171 | **0,0886** | **0,0638** | **-2,77%** | **-0,0389** | **-0,0220** | **46,71%** | **0,4866** | **0,3783** |
| **LIGHTGBM**| 1. Treino (70%) | 13.265 | 0,0955 | 0,0674 | +7,71% | +0,3412 | +0,2166 | 57,69% | 0,6028 | 0,4901 |
| **LIGHTGBM**| 2. Validação (20%) | 4.063 | 0,0800 | 0,0582 | -3,28% | +0,0004 | -0,0099 | 49,94% | 0,4985 | 0,4162 |
| **LIGHTGBM**| **3. Teste Puro (10%)** | 2.171 | **0,0890** | **0,0639** | **-3,87%** | **-0,0267** | **-0,0052** | **48,13%** | **0,5003** | **0,3953** |

> **Diagnóstico Científico do Trade-off Viés-Variância:**
> - O **LightGBM** exibiu alta capacidade de memorização no treino ($R^2 = +7,71\%$ e Rank IC $= 0,2166$), mas sofreu perda de generalização no teste estático devido à baixa razão sinal-ruído das séries financeiras.
> - A **Regressão Ridge**, com forte restrição estrutural de complexidade ($L_2$, $\lambda = 10$), evitou sobreajuste a ruídos locais e apresentou a menor degradação de erro ($RMSE = 0,0883$).

---

### 2. Simulação Walk-Forward Multi-Ano: Retreino Periódico vs. Modelos Congelados (2015–2026)

Impacto empírico do retreino periódico ao longo de **142 rebalanceamentos mensais**:

| Estratégia | Modo de Execução | Retorno Acumulado (%) | CAGR (% a.a.) | Volatilidade (% a.a.) | Sharpe Ratio | Max Drawdown (%) | Alpha vs. Universo (% a.a.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (Universo 1/N)** | Benchmark Passivo | +615,00% | 19,74% | 24,26% | 0,81 | -37,16% | 0,00% |
| **Ibovespa (BOVA11)** | Benchmark Ponderado | +245,30% | 11,28% | 26,14% | 0,73 | -44,21% | -8,46% |
| **ML: RIDGE (Walk-Forward)** | Retreino Anual (12m) | **+1.415,50%** | **28,28%** | 27,93% | **1,01** | **-39,34%** | **+8,54%** |
| **ML: XGBOOST (Walk-Forward)** | Retreino Anual (12m) | **+827,44%** | **22,63%** | 29,17% | **0,78** | **-45,89%** | **+2,89%** |
| **ML: LIGHTGBM (Walk-Forward)** | Retreino Anual (12m) | **+777,54%** | **22,01%** | 29,27% | **0,75** | **-41,76%** | **+2,27%** |
| **ML: RIDGE (Congelado)** | Estático (Sem Retreino) | +590,71% | 19,37% | 21,87% | 0,89 | -36,71% | -0,37% |
| **ML: XGBOOST (Congelado)** | Estático (Sem Retreino) | +563,51% | 18,93% | 23,59% | 0,80 | -41,24% | -0,81% |
| **ML: LIGHTGBM (Congelado)** | Estático (Sem Retreino) | +828,28% | 22,64% | 22,75% | 1,00 | -37,54% | +2,90% |

---

### 3. Políticas de Portfólio: Hard Cut (Top 10) vs. Soft Weighting (Top 15)

| Modelo | Política de Ponderação | Retorno Acumulado (%) | CAGR (% a.a.) | Volatilidade (% a.a.) | Sharpe Ratio | Max Drawdown (%) | Alpha vs. Universo (% a.a.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (Universo)** | Equiponderado (1/N) | +615,00% | 19,74% | 24,26% | 0,81 | -37,16% | 0,00% |
| **ML: RIDGE** | Hard Cut (Top 10 1/N com Histerese) | +1.415,50% | 28,28% | 27,93% | 1,01 | -39,34% | +8,54% |
| **ML: RIDGE** | **Soft Weighting (Top 15 Rank Linear)** | **+1.568,85%** | **29,41%** | 28,81% | **1,02** | **-35,23%** | **+9,67%** |
| **ML: XGBOOST** | Hard Cut (Top 10 1/N com Histerese) | +827,44% | 22,63% | 29,17% | 0,78 | -45,89% | +2,89% |
| **ML: XGBOOST** | **Soft Weighting (Top 15 Rank Linear)** | **+862,40%** | **23,05%** | 28,59% | **0,81** | **-40,82%** | **+3,31%** |
| **ML: LIGHTGBM** | Hard Cut (Top 10 1/N com Histerese) | +777,54% | 22,01% | 29,27% | 0,75 | -41,76% | +2,27% |
| **ML: LIGHTGBM** | Soft Weighting (Top 15 Rank Linear) | +670,39% | 20,57% | 28,74% | 0,72 | -42,05% | +0,83% |

> **Vantagens da Ponderação Suave (*Soft Weighting*):**
> * **Redução do Risco Extremo:** O Max Drawdown do XGBoost cai de **-45,89% para -40,82%**, e o da Ridge cai de **-39,34% para -35,23%**.
> * **Eliminação do Efeito Falésia (*Cliff Edge*):** Em vez de atribuir 10% para o 10º colocado e 0% para o 11º, o peso decai suavemente ($12,5\% \to 11,7\% \to \dots \to 0,8\%$), neutralizando a sensibilidade a empates e à aleatoriedade estocástica dos nós das árvores.

---

## 🏗️ Estrutura do Repositório

```text
quant-ml-b3-portfolio/
├── app.py                                 # Terminal Web Interativo em Streamlit
├── README.md                              # Documentação executiva completa e guia de uso
├── pyproject.toml                         # Configurações de empacotamento moderno (PEP 621) e pytest
├── requirements.txt                       # Dependências reproduzíveis do projeto
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
│       └── plotting.py                    # Gráficos estáticos de alta resolução para relatórios
├── scripts/
│   ├── run_experiment.py                  # CLI para rodar o pipeline clássico 70/20/10 e Walk-Forward
│   ├── export_dashboard_data.py           # Pipeline que gera as tabelas completas para o Streamlit
│   └── update_market_data_to_today.py     # Script para sincronizar e atualizar dados de mercado
├── notebooks/
│   └── 01_ml_quant_strategies_b3.ipynb    # Jupyter Notebook executado com análise passo a passo
├── outputs/
│   ├── figures/                           # Visualizações estáticas em PNG de alta resolução
│   └── tables/                            # Séries temporais, carteiras e retornos em CSV
└── tests/                                 # Suíte automatizada de testes com pytest
    ├── test_features_and_splits.py        # Testa causalidade, purga e Z-score
    ├── test_models_and_metrics.py         # Testa convergência de modelos e métricas
    └── test_walk_forward.py               # Testa motor e histerese do Walk-Forward
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone https://github.com/pdromls/quant-ml-b3-portfolio.git
cd quant-ml-b3-portfolio
pip install -r requirements.txt
```

### 2. Rodar o Terminal Interativo em Streamlit
```bash
streamlit run app.py
```
O dashboard será aberto automaticamente no seu navegador em `http://localhost:8501`.

### 3. Rodar a Suíte de Testes Automatizados (`pytest`)
Garante imunidade a vazamento temporal, correção de dimensões matriciais e conformidade dos cálculos:
```bash
pytest -v
```

### 4. Re-executar os Pipelines de Modelagem e Dados (Opcional)
```bash
# Rodar experimentos completos de Machine Learning e Walk-Forward:
python scripts/run_experiment.py

# Exportar dados atualizados para o dashboard:
python scripts/export_dashboard_data.py
```

---

## 👤 Autor

**Pedro Magalhães**  
Ciência da Computação · Inteligência Artificial & Finanças Quantitativas  
GitHub: [@pdromls](https://github.com/pdromls)
