#!/usr/bin/env bash
# Levanta el servidor de MLflow (tracking + Model Registry + artifact store) en un contenedor.
# Se conecta a la red Docker "kind" para que los Pods del clúster lo encuentren como http://mlflow:5000.
# Interfaz web: http://127.0.0.1:5000  (usar 127.0.0.1: "localhost" resuelve a IPv6 y la red de kind no lo publica)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p mlflow-data
if docker ps -a --format '{{.Names}}' | grep -qx mlflow; then
  docker update --restart unless-stopped mlflow >/dev/null   # contenedores creados antes de esta política
  docker start mlflow >/dev/null
else
  # --restart unless-stopped: MLflow vuelve solo tras reiniciar Docker o la PC (salvo que lo detengas a mano)
  # -p 127.0.0.1:5000:5000: la interfaz (sin autenticación) solo es visible desde esta máquina; los Pods usan la red "kind"
  docker run -d --name mlflow --restart unless-stopped --network kind -p 127.0.0.1:5000:5000 \
    -u "$(id -u):$(id -g)" -v "$PWD/mlflow-data:/mlflow" \
    riesgo-mlflow:3.14.0 \
    mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts --serve-artifacts \
      --allowed-hosts "localhost,localhost:5000,127.0.0.1:5000,mlflow,mlflow:5000" >/dev/null
fi
for i in $(seq 1 60); do curl -sf http://127.0.0.1:5000/health >/dev/null && break; sleep 1; done
echo "MLflow listo en http://127.0.0.1:5000 ($(curl -s http://127.0.0.1:5000/version))"
