# Cross-Sectional Machine Learning e Backtesting Orientado a Eventos na B3 (IBrX 50)

Pipeline quantitativo em Python para seleção cross-sectional de ações na B3 (universo IBrX 50, 2012–2026), com um terminal analítico web em Streamlit e um relatório executivo em Jupyter Notebook.

**Acesse o terminal online:** https://quant-ml-b3.streamlit.app

O link abre o terminal completo no navegador, sem instalar nada. Se o app estiver hibernando por falta de acessos, clique em "Yes, get this app back up!" e aguarde cerca de um minuto.

---

## Como Usar

Há três formas de abrir o terminal. Todas mostram o mesmo app.

| Opção | Quando usar | Requisitos |
| :--- | :--- | :--- |
| Online (Streamlit Community Cloud) | Ver o projeto rapidamente ou compartilhar | Apenas um navegador |
| Docker | Rodar localmente com um comando, sem configurar Python | Docker instalado |
| Python local | Rodar o app, os testes, o notebook e os pipelines de modelagem | Python 3.10+ |

### Opção 1: Online

Acesse https://quant-ml-b3.streamlit.app.

### Opção 2: Docker

No diretório raiz do projeto:
```bash
docker compose up
```
O container instala as dependências, configura o ambiente Linux com suporte multithread para LightGBM/XGBoost e publica o terminal em `http://localhost:8501`.

### Opção 3: Python local

#### 1. Instalar as dependências
```bash
git clone https://github.com/pdromls/Quant-ML-B3-Portfolio.git
cd Quant-ML-B3-Portfolio
pip install -r requirements-dev.txt
```
`requirements.txt` contém apenas o necessário para o terminal web. `requirements-dev.txt` acrescenta pytest e Jupyter.

#### 2. Abrir o terminal web
```bash
streamlit run app.py
```
O terminal abre no navegador em `http://localhost:8501`.

#### 3. Abrir o relatório em Jupyter Notebook
```bash
jupyter notebook notebooks/01_ml_quant_strategies_b3.ipynb
```

#### 4. Executar os testes automatizados
```bash
pytest -v
```

#### 5. Reexecutar os pipelines de modelagem e exportação
```bash
# Experimentos de Machine Learning e simulação Walk-Forward:
python scripts/run_experiment.py

# Exportar tabelas atualizadas para o terminal:
python scripts/export_dashboard_data.py
```

---

## Guia do Terminal Web

### Tema claro e escuro
O terminal segue o tema do sistema operacional. Para trocar manualmente, abra o menu ⋮ no canto superior direito → Settings → Theme e escolha Light ou Dark.

### Ícone de informação
Todo gráfico e tabela tem um ícone de informação no canto superior direito. Ele abre uma janela explicando o que o gráfico mostra, como ler e de onde vêm os dados. Os gráficos principais também trazem um quadro "Como ler" logo abaixo.

### Barra lateral (controles globais)
- **Fonte de dados**
  - Base local: opção padrão. Carrega a base tratada de 41 ações do IBrX 50 (3.667 pregões auditados).
  - Yahoo Finance: sincroniza cotações recentes da B3. O botão "Atualizar cotações" baixa novos preços e "Resetar" volta para a base local.
- **Modelo de Machine Learning**
  - Regressão Ridge: modelo linear com regularização L2.
  - XGBoost: gradient boosted trees com divisão por profundidade (depth-wise).
  - LightGBM: gradient boosted trees com divisão ótima por folha (leaf-wise).
  - Comparar todos os modelos: sobrepõe as três estratégias.
- **Construção de carteira**
  - Soft Weighting (Top 15): pesos decrescentes por rank, de 12,50% para o 1º colocado até 0,83% para o 15º, reduzindo a sensibilidade a empates.
  - Hard Cut (Top 10): 10% fixos para as 10 melhores ações, com margem de tolerância no ranking (filtro anti-giro) para evitar trocas por variações mínimas de score.
- **Metodologia quant**: resumo dos parâmetros fixos (rebalanceamento mensal no 1º pregão, z-score cross-sectional, purga de 21 dias, custo de 0,15% por transação, alvo de excesso de retorno em 21 dias).

### Aba 1: Desempenho histórico (2015–2026)
Avalia a performance de longo prazo das estratégias contra o mercado.
- **Indicadores principais:** retorno total, CAGR, Sharpe, volatilidade, max drawdown e alpha líquido do modelo selecionado, cada um comparado ao Buy & Hold do universo.
- **Curva de patrimônio acumulado:** retorno composto do modelo contra o Buy & Hold 1/N e o Ibovespa (BOVA11), em escala linear ou logarítmica.
- **Drawdown:** gráfico separado mostrando quanto cada carteira está abaixo do seu pico anterior, com tabela de pior queda, período pico → fundo, tempo de recuperação e situação atual.
- **Tabela resumo comparativa:** retorno acumulado, CAGR, volatilidade anual, Sharpe, max drawdown, turnover médio mensal e alpha vs. universo para todas as combinações de modelo e política. A seleção atual fica destacada e o melhor valor de cada coluna aparece em verde.

### Aba 2: Previsão vs. realidade
Mostra exatamente o que cada modelo recomendou comprar e o que aconteceu no mercado. O modelo é escolhido no topo da aba (Ridge, XGBoost, LightGBM ou os três lado a lado). Há três modos de inspeção:

1. **Visão anual**
   - Seletor de ano (2015 a 2026, com o ano corrente por padrão).
   - Cartões com o resultado do ano de cada modelo, do Buy & Hold e do Ibovespa, além do alpha.
   - Evolução acumulada dentro do ano, retornos mês a mês em barras e tabela mensal com score previsto, retorno do modelo, benchmarks, alpha e a principal ação escolhida.

2. **Por pregão**
   - Calendário (DD/MM/AAAA) para escolher qualquer dia da série.
   - Se o dia escolhido não teve pregão (fim de semana ou feriado), o terminal avisa e usa o pregão imediatamente anterior.
   - Botões "Pregão anterior", "Próximo pregão" e "Mais recente" navegam apenas entre pregões reais.
   - Compara o preço de entrada no 1º pregão do mês (data da recomendação) com o fechamento do dia escolhido.
   - Gráfico de barras com cores semânticas (verde quando a ação superou o Buy & Hold do universo, vermelho quando ficou abaixo) e linhas de referência para Buy & Hold e Ibovespa, ordenável por rank do modelo ou por retorno.

3. **Mês a mês**
   - Seletor com os 142 meses da série e botões "Mês anterior", "Próximo mês" e "Mais recente".
   - Cartões, gráfico por ação e tabela da carteira recomendada: posição, ticker, score previsto, peso, preços de entrada e de saída, retorno, contribuição e se a ação bateu o mercado. A soma das contribuições aparece abaixo da tabela e corresponde ao retorno da carteira.

### Aba 3: Explicabilidade das features
Mostra os fundamentos econômicos das decisões dos modelos.
- Gráfico de barras horizontais por modelo:
  - Regressão Ridge: coeficientes beta padronizados em basis points (verde favorece a ação, vermelho penaliza).
  - XGBoost: ganho de informação (%) por divisão de nó.
  - LightGBM: ganho de informação (%) por folha.
- Os 12 indicadores técnicos, com nomes em português: volatilidade realizada (21 dias), posição nas Bandas de Bollinger, momentum trimestral (63 dias), retornos de 21, 10, 5 e 1 dia, RSI (14 dias), inclinação entre médias (SMA 20/50), distância das médias móveis de 50 e 20 dias e aceleração de volume (5d vs. 21d).
- Interpretação financeira dos fatores mais relevantes (`ret_63d`, `ret_1d`, `bb_pos` e `vol_21d`).

### Aba 4: Auditoria de qualidade de dados
- O botão "Executar auditoria de dados agora" roda as verificações formais sobre as matrizes de preço e volume.
- Mostra o status, as dimensões da base, a cobertura média de ações ativas e um checklist (cronologia estrita, ausência de datas duplicadas, de valores infinitos e de preços não positivos), com o relatório completo em texto.

---

## Sobre o Projeto

O foco metodológico do projeto é a integridade científica:
- Engenharia de atributos estritamente causal (sem viés de antecipação ou look-ahead bias).
- Padronização diária cross-sectional via Z-Score com winsorização [-4, +4], neutralizando o viés macroeconômico direcional e isolando o alpha relativo.
- Divisão temporal cronológica 70/20/10 com purga e embargo temporal de 21 dias úteis nas fronteiras de Treino, Validação e Teste, impedindo vazamento de dados futuros (data leakage).
- Simulação Walk-Forward ao longo de 12 anos (2015–2026, totalizando 142 rebalanceamentos mensais) para testar empiricamente o fenômeno de concept drift, confrontando modelos com retreino periódico anual contra modelos estáticos (congelados).
- Política de alocação suave (Soft Weighting Top 15), eliminando a sensibilidade a empates e o efeito falésia (cliff edge) inerente ao corte rígido tradicional (Hard Cut Top 10).
- Comparação sistemática com os dois benchmarks oficiais de mercado: o Buy & Hold do Universo (média equiponderada 1/N de todas as ações elegíveis) e o Ibovespa (ETF BOVA11).

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
├── .streamlit/
│   └── config.toml                        # Temas claro/escuro, fontes e paleta do terminal
├── Dockerfile                             # Imagem containerizada otimizada (Python 3.11 + libgomp1)
├── docker-compose.yml                     # Orquestrador para execução com 1 comando
├── .dockerignore                          # Filtro de arquivos para o build do container
├── README.md                              # Documentação técnica completa
├── pyproject.toml                         # Configurações de empacotamento moderno e pytest
├── requirements.txt                       # Dependências do terminal web (usadas no Docker e na nuvem)
├── requirements-dev.txt                   # requirements.txt + pytest e Jupyter
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
│       ├── plotting.py                    # Gráficos para relatórios
│       └── ui.py                          # Camada visual do terminal (cartões, painéis, modais, estilo dos gráficos)
├── scripts/
│   ├── run_experiment.py                  # CLI para rodar o pipeline clássico 70/20/10 e Walk-Forward
│   ├── export_dashboard_data.py           # Pipeline que gera as tabelas completas para o Streamlit
│   └── update_market_data_to_today.py     # Script para sincronizar dados de mercado até a data corrente
├── notebooks/
│   └── 01_ml_quant_strategies_b3.ipynb    # Jupyter Notebook executado com análise passo a passo
├── docs/
│   └── design/                            # Capturas e mockups do terminal (temas claro e escuro)
├── outputs/
│   ├── figures/                           # Visualizações estáticas em PNG
│   └── tables/                            # Séries temporais, carteiras e retornos em CSV
└── tests/                                 # Suíte automatizada de testes com pytest
    ├── test_features_and_splits.py        # Validação de causalidade, purga e Z-score
    ├── test_models_and_metrics.py         # Validação de convergência de modelos e métricas
    └── test_walk_forward.py               # Validação de motor e margem de tolerância do Walk-Forward
```


## Autor

Pedro Magalhães  
Ciência da Computação  
GitHub: @pdromls
