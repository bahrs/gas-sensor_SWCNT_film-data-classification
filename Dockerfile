FROM python:3.13-slim

WORKDIR /app

# System deps (libgomp1 helps with OpenMP runtime used by some ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_HOST=0.0.0.0 \
    MLFLOW_PORT=5050 \
    MLFLOW_BACKEND_STORE_URI=sqlite:////app/mlflow.db \
    MLFLOW_ARTIFACT_ROOT=file:/app/mlruns

# Install Python deps first (better build caching)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create expected directories
RUN mkdir -p data/raw data/processed mlruns configs

# Create an entrypoint that starts MLflow server, then launches training
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5050

ENTRYPOINT ["/entrypoint.sh"]

# Default command (override in `docker run ... <cmd>`)
CMD ["python", "scripts/run_optimization_from_config.py", "configs/config_catboost_regression.yaml"]