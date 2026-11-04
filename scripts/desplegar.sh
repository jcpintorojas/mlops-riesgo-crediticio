#!/usr/bin/env bash
# Construye la imagen de la API, la carga en el clúster kind "mlops" y aplica todos los manifiestos.
# Requisitos: servidor MLflow arriba (scripts/iniciar_mlflow.sh) con el modelo registrado y alias @champion.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
docker build -t riesgo-api:2.0 api
kind load docker-image riesgo-api:2.0 --name mlops
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl create configmap riesgo-ui --from-file=ui/index.html --from-file=ui/default.conf --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/ui.yaml
kubectl rollout status deployment/riesgo-api --timeout=300s
kubectl rollout status deployment/riesgo-ui --timeout=120s
kubectl get pods -o wide
