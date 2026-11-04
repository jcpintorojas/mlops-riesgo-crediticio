#!/usr/bin/env bash
# Levanta el servidor de MLflow (tracking + Model Registry + artifact store) en un contenedor.
# Se conecta a la red Docker "kind" para que los Pods del clúster lo encuentren como http://mlflow:5000.
# Interfaz web: http://localhost:5000
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p mlflow-data
if docker ps -a --format '{{.Names}}' | grep -qx mlflow; then
  docker start mlflow >/dev/null
else
  docker run -d --name mlflow --network kind -p 5000:5000 \
    -u "$(id -u):$(id -g)" -v "$PWD/mlflow-data:/mlflow" \
    riesgo-mlflow:3.14.0 \
    mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts --serve-artifacts \
      --allowed-hosts "localhost,localhost:5000,127.0.0.1:5000,mlflow,mlflow:5000" >/dev/null
fi
for i in $(seq 1 60); do curl -sf http://localhost:5000/health >/dev/null && break; sleep 1; done
echo "MLflow listo en http://localhost:5000 ($(curl -s http://localhost:5000/version))"
