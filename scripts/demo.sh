#!/usr/bin/env bash
# Levanta TODO para la presentación con un solo comando (el clúster y MLflow ya deben haberse creado una vez).
#   bash scripts/demo.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
echo "1) Clúster Kubernetes (kind 'mlops')"
docker start mlops-control-plane >/dev/null 2>&1 || true
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do sleep 2; done
kubectl get nodes
echo "2) Servidor MLflow"
bash scripts/iniciar_mlflow.sh
echo "3) Pods de la API y de la interfaz"
kubectl rollout restart deployment/riesgo-api >/dev/null   # que vuelvan a leer @champion del registro
kubectl rollout status deployment/riesgo-api --timeout=300s
kubectl rollout status deployment/riesgo-ui --timeout=120s
kubectl get pods -o wide
NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
echo
echo "Listo:"
echo "  MLflow (experimentos y Model Registry): http://localhost:5000"
echo "  Interfaz web (consume la API en K8s):   http://$NODO:30091"
echo "  API (Swagger):                         http://$NODO:30090/docs"
echo "  Prueba rápida de balanceo:"
echo "    for i in \$(seq 1 10); do curl -s -X POST http://$NODO:30090/predict -H 'Content-Type: application/json' -d @scripts/cliente_alto.json; echo; done"
command -v xdg-open >/dev/null && { xdg-open http://localhost:5000 >/dev/null 2>&1; xdg-open "http://$NODO:30091" >/dev/null 2>&1; } || true
