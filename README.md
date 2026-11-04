# MLOps de riesgo crediticio — del entrenamiento a Kubernetes con detección de drift

Proyecto final del módulo **ML-Ops y Puesta en Producción** (MIA-V1E4, UAGRM-SOE) · **Julio Cesar Pinto Rojas**

Modelo que predice si un cliente de tarjeta de crédito incumplirá su próximo pago (dataset UCI *Default of Credit Card Clients*, 30.000 clientes), llevado de punta a punta:

| Fase | Qué hay | Dónde |
|---|---|---|
| 1. MLflow | 6 runs comparables, modelo `riesgo-credito` con 2 versiones y alias `@champion` | `src/`, `docker/mlflow/` |
| 2. Docker | API FastAPI que carga `models:/riesgo-credito@champion` y responde con versión, run y Pod | `api/` |
| 3. Kubernetes | 3 réplicas, Service NodePort, probes, recursos; balanceo, autorreparación y escalado demostrados | `k8s/`, `scripts/demo_k8s.sh` |
| 4. Drift | Data drift (KS + PSI, pruebas pytest verde/rojo) y concept drift (AUC por lote, criterio de reentrenamiento) | `drift/` |
| Extra | Interfaz web servida por nginx en el clúster que consume la API | `ui/`, `k8s/ui.yaml` |

| Documento | Para qué |
|---|---|
| [`docs/OPERACION.md`](docs/OPERACION.md) | Arranque, funcionamiento y solución de problemas |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Decisiones, resultados y justificación |
| [`AGENTS.md`](AGENTS.md) | Guía para agentes de IA |
| [`docs/PENDIENTES.md`](docs/PENDIENTES.md) | Lo que falta y mejoras posibles |
| [`evidencias/`](evidencias/) | Salidas reales y capturas |

## Requisitos
Docker, [kind](https://kind.sigs.k8s.io/) y kubectl. Todo lo de Python corre dentro de contenedores (no hace falta instalar nada más).

## Puesta en marcha
```bash
kind create cluster --name mlops                           # una sola vez
docker build -t riesgo-mlflow:3.14.0 docker/mlflow
bash scripts/iniciar_mlflow.sh                            # http://127.0.0.1:5000
bash scripts/entrenar.sh src/descargar_datos.py
bash scripts/entrenar.sh src/entrenar.py
bash scripts/entrenar.sh src/registrar.py
bash scripts/desplegar.sh
```
Después, para la demo: `bash scripts/demo.sh` (levanta clúster, MLflow y Pods, e imprime las URLs).

> Usar `http://127.0.0.1:5000` y no `localhost:5000`: en algunas máquinas `localhost` resuelve a IPv6 y la red de kind publica el puerto solo en IPv4.
