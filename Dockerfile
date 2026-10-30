# Imagem base oficial do Python enxuta
FROM python:3.11-slim

# Configurações de ambiente para Python e Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Instalar dependências nativas de sistema:
# - libgomp1: biblioteca OpenMP indispensável para execução multithread do LightGBM e XGBoost
# - curl: utilizado na validação de integridade (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar o restante da aplicação, modelos e tabelas pré-computadas
COPY . .

# Configurar o caminho do módulo do projeto no PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expor a porta padrão do Streamlit
EXPOSE 8501

# Verificação de saúde do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização do terminal Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
