#!/usr/bin/env bash
# Construye la imagen de la API, la carga en el clúster kind "mlops" y aplica todos los manifiestos.
# Requisitos: servidor MLflow arriba (scripts/iniciar_mlflow.sh) con el modelo registrado y alias @champion.
# Detiene el script ante un error.
set -euo pipefail
# Trabaja desde la raíz del proyecto.
cd "$(dirname "$0")/.."
# Permite encontrar kind y kubectl.
export PATH=$HOME/.local/bin:$PATH
# Construye la imagen de la API.
docker build -t riesgo-api:2.0 api
# Copia la imagen al nodo de kind; no hay un registro externo.
kind load docker-image riesgo-api:2.0 --name mlops
# Crea o actualiza la API y su Service.
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
# Crea el ConfigMap con la página y la configuración de nginx.
kubectl create configmap riesgo-ui --from-file=ui/index.html --from-file=ui/default.conf --dry-run=client -o yaml | kubectl apply -f -
# Despliega la interfaz.
kubectl apply -f k8s/ui.yaml
# Espera a que las réplicas estén listas.
kubectl rollout status deployment/riesgo-api --timeout=300s
kubectl rollout status deployment/riesgo-ui --timeout=120s
# Muestra el resultado.
kubectl get pods -o wide
