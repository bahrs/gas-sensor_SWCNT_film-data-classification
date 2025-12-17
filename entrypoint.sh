#!/usr/bin/env sh
set -eu

echo "[entrypoint] Starting MLflow server on ${MLFLOW_HOST}:${MLFLOW_PORT}"
mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" &

echo "[entrypoint] Starting main command: $*"
exec "$@"