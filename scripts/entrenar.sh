#!/usr/bin/env bash
# Ejecuta un script de src/ dentro del entorno de experimentación, conectado al servidor MLflow.
# USO: bash scripts/entrenar.sh [src/entrenar.py]
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm --network kind -u "$(id -u):$(id -g)" -e HOME=/tmp -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo sin-git)" \
  -e PYTHONPATH=/proyecto/src -v "$PWD:/proyecto" riesgo-mlflow:3.14.0 python "${@:-src/entrenar.py}"
