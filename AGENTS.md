# AGENTS.md — Guía para agentes de IA y personas nuevas en el proyecto

Lee este archivo primero. Resume qué es el sistema, cómo arrancarlo y qué no romper.
Detalle: [`docs/OPERACION.md`](docs/OPERACION.md) (arranque y fallas) · [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) (decisiones y resultados).
Trabajo pendiente: [`docs/PENDIENTES.md`](docs/PENDIENTES.md).

## Qué es

Sistema MLOps de punta a punta que predice si un cliente de tarjeta de crédito incumplirá su próximo pago (dataset UCI *Default of Credit Card Clients*, 30.000 filas, 23 variables). Es un proyecto académico: se evalúa la ingeniería alrededor del modelo, no su exactitud.

## Flujo del sistema

```
data/credito.csv -> src/entrenar.py -> MLflow (runs) -> src/registrar.py -> Model Registry (riesgo-credito@champion)
                                                                                     |
                                                       cada Pod lo descarga al arrancar
                                                                                     v
navegador -> riesgo-ui (nginx, :30091) -> /api -> Service riesgo-api (:30090) -> 3 Pods FastAPI (:8000)

drift/data_drift.py y drift/concept_drift.py comparan datos y métricas contra el baseline de entrenamiento.
```

## Mapa del repositorio

| Ruta | Función |
|---|---|
| `docker/mlflow/` | Imagen `riesgo-mlflow:3.14.0`: servidor MLflow, entrenamiento y drift |
| `src/datos.py` | Carga, split 80/20 (semilla 42) y lista de variables. **Fuente única** para entrenamiento y drift |
| `src/descargar_datos.py` | Descarga el dataset a `data/credito.csv` (no se versiona) |
| `src/entrenar.py` | 6 runs en el experimento `riesgo-crediticio` |
| `src/registrar.py` | Registra dos versiones y asigna el alias `champion` a la mejor |
| `src/alias.py` | Mueve `@champion` a otra versión |
| `src/resumen_mlflow.py` | Imprime runs, versiones y alias (solo lectura) |
| `api/` | Servicio FastAPI e imagen `riesgo-api:2.0`. El modelo **no** va dentro de la imagen |
| `k8s/` | Deployment y Service de la API; Deployment y Service de la interfaz |
| `ui/` | `index.html` y configuración de nginx (se montan con el ConfigMap `riesgo-ui`) |
| `drift/` | Data drift (KS + PSI), pruebas pytest y simulación de concept drift |
| `scripts/` | `iniciar_mlflow.sh`, `entrenar.sh`, `desplegar.sh`, `demo.sh`, `demo_k8s.sh` |
| `evidencias/` | Salidas reales y capturas. No editar a mano: se regeneran ejecutando las pruebas |
| `mlflow-data/`, `data/`, `drift/salidas/` | Estado local, excluido de Git |

## Comandos

Requisitos: Docker, kind y kubectl. Python corre solo dentro de contenedores.

```bash
export PATH=$HOME/.local/bin:$PATH                 # si kind y kubectl están instalados ahí
bash scripts/demo.sh                               # sistema ya creado: levanta todo tras un reinicio
bash scripts/entrenar.sh src/resumen_mlflow.py     # ver qué versión está en producción
bash scripts/entrenar.sh -m pytest -v drift/test_drift.py   # pruebas (no necesitan MLflow; sí data/credito.csv)
kubectl get pods                                   # estado del clúster
```

La instalación desde cero está en `docs/OPERACION.md`.
`scripts/entrenar.sh <archivo.py>` ejecuta cualquier script dentro de la imagen de experimentación, en la red Docker `kind`.

## Reglas al modificar

- **Siempre** usa `http://127.0.0.1:5000` para MLflow desde la máquina: `localhost` puede resolver a IPv6 y fallar.
- Dentro de la red `kind`, MLflow es `http://mlflow:5000`. Su nombre de contenedor y `--allowed-hosts` deben coincidir con esa URL.
- La API carga el modelo solo por `models:/riesgo-credito@champion`. No agregues rutas de archivo ni `joblib` manual: el enunciado lo prohíbe.
- Cambiar variables exige actualizar juntos `src/datos.py` y la clase `Cliente` de `api/app.py`, y volver a entrenar.
- Las versiones de ML de `api/requirements.txt` deben coincidir con las de `docker/mlflow/requirements.txt`.
- Después de editar `api/`: `bash scripts/desplegar.sh`. Si mantienes el tag `2.0`, agrega `kubectl rollout restart deployment/riesgo-api`.
- Después de editar `ui/`: `bash scripts/desplegar.sh` y `kubectl rollout restart deployment/riesgo-ui` (los montajes con `subPath` no se actualizan solos).
- Después de mover el alias: `kubectl rollout restart deployment/riesgo-api` (el modelo se lee solo al arrancar).

## Operaciones con efectos: confirma antes

| Operación | Efecto |
|---|---|
| `src/entrenar.py` | Agrega 6 runs |
| `src/registrar.py` | Crea 2 versiones **nuevas** y mueve `@champion`; la documentación cita v2 y su run |
| `src/alias.py` | Cambia el modelo servido tras reiniciar los Pods; vuelve a dejar la versión 2 |
| `drift/concept_drift.py` | Agrega un run a `monitoreo-concept-drift` |
| `scripts/demo_k8s.sh` | Borra un Pod, escala, reinicia y sobrescribe `evidencias/k8s-*` |
| Borrar `mlflow-data/` | **Destruye** todos los runs, modelos y el registro; no está en Git |

## Limitaciones conocidas

- **MLflow no tiene política de reinicio.** Tras reiniciar la PC, el clúster vuelve solo pero MLflow no. Solución: `bash scripts/demo.sh`.
- **La carga del modelo bloquea el arranque** (`api/app.py`, evento `startup`). Si MLflow está caído, un Pod nuevo no llega a responder `/health`, la `startupProbe` lo mata a los 2 minutos y entra en `CrashLoopBackOff`. Los Pods que ya cargaron el modelo siguen atendiendo (medido: 20 de 20 peticiones con MLflow apagado).
- No hay CI/CD; MLflow usa SQLite y disco local.
- La instalación desde cero no se ha probado en una máquina limpia. Sí se probó un clon nuevo: descarga de datos y pruebas de drift.
