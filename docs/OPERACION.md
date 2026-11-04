# Operación: cómo arrancar y cómo funciona

## 1. Requisitos

| Herramienta | Versión usada | Para qué |
|---|---|---|
| Docker | 29.1.3 | Todos los contenedores |
| kind | 0.33.0 | Clúster Kubernetes local (1.37) |
| kubectl | 1.37.2 | Administrar el clúster |
| Git, curl | — | Clonar y probar |

No hace falta instalar Python: entrenamiento, drift y servidor corren dentro de contenedores. Se probó en Linux.

Puertos: `5000` (MLflow), `30090` (API) y `30091` (interfaz).

## 2. Arranque según el caso

### A. El sistema ya existe (por ejemplo, después de reiniciar la PC)

```bash
export PATH=$HOME/.local/bin:$PATH
bash scripts/demo.sh
```

Enciende el clúster y MLflow, reinicia los Pods para que lean `@champion`, espera a que estén listos e imprime las direcciones. Medido tras un reinicio real: 54 segundos.

### B. Instalación desde cero

Cada comando depende del anterior:

```bash
git clone https://github.com/jcpintorojas/mlops-riesgo-crediticio
cd mlops-riesgo-crediticio

kind create cluster --name mlops                   # también crea la red Docker "kind"
docker build -t riesgo-mlflow:3.14.0 docker/mlflow
bash scripts/iniciar_mlflow.sh                     # necesita la red "kind"

bash scripts/entrenar.sh src/descargar_datos.py    # crea data/credito.csv
bash scripts/entrenar.sh src/entrenar.py           # 6 runs (cerca de 1 minuto)
bash scripts/entrenar.sh src/registrar.py          # versiones 1 y 2, alias @champion

bash scripts/desplegar.sh                          # imagen, carga en kind, manifiestos e interfaz
```

Los identificadores de run serán distintos a los de `docs/ARQUITECTURA.md`, porque corresponden a la ejecución original. Consúltalos con `bash scripts/entrenar.sh src/resumen_mlflow.py`.

## 3. Comprobar que funciona

```bash
kubectl get pods                                   # riesgo-api x3 y riesgo-ui en 1/1 Running
curl -s http://127.0.0.1:5000/health               # OK
NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
curl -s http://$NODO:30090/info                    # versión y run del modelo servido
curl -s -X POST http://$NODO:30090/predict -H 'Content-Type: application/json' -d @scripts/cliente_alto.json
bash scripts/entrenar.sh -m pytest -v drift/test_drift.py    # 3 passed
```

| Interfaz | Dirección |
|---|---|
| MLflow | `http://127.0.0.1:5000` (en la barra lateral elige **Model training**) |
| Web | `http://<NODO>:30091` |
| Swagger de la API | `http://<NODO>:30090/docs` |

## 4. Cómo funciona

**Entrenamiento y registro.** `src/entrenar.py` usa el split de `src/datos.py` y registra en MLflow parámetros, métricas, md5 de los datos, commit de Git y el modelo empaquetado. `src/registrar.py` registra el mejor modelo lineal como versión 1 y el mejor AUC como versión 2, y le asigna `@champion`.

**Una petición.**
1. La web envía `POST /api/predict` a nginx.
2. nginx la reenvía al Service `riesgo-api` sin keepalive, así que cada petición abre una conexión nueva.
3. kube-proxy elige uno de los Pods listos. El reparto es aleatorio, no exactamente parejo.
4. Pydantic valida la entrada; si es inválida, responde 422.
5. El modelo devuelve `[P(cumple), P(incumple)]`, y la API responde la probabilidad con la versión, el run y el Pod.

**Arranque de un Pod.** Durante `startup`, la API pregunta a MLflow qué versión tiene `@champion` y la descarga. Solo entonces el servidor responde.

| Sonda | Ruta | Si falla |
|---|---|---|
| startupProbe | `/health` | Tras 24 intentos cada 5 s (2 min) reinicia el contenedor |
| readinessProbe | `/ready` | Saca al Pod del Service |
| livenessProbe | `/health` | Tras 3 fallos reinicia el contenedor |

**Cambio de modelo.** `bash scripts/entrenar.sh src/alias.py <versión>` y luego `kubectl rollout restart deployment/riesgo-api`. La imagen no cambia.

**Drift.** `drift/data_drift.py` compara cada variable con el baseline: KS con Bonferroni para las numéricas y PSI para las categóricas. `drift/concept_drift.py` simula un cambio en la relación entrada-salida y dispara la alarma si el AUC queda 3 lotes seguidos bajo referencia − 2σ. Los umbrales y su justificación están en `docs/ARQUITECTURA.md` §6.

## 5. Problemas conocidos

| Síntoma | Causa | Solución |
|---|---|---|
| Pods en `CrashLoopBackOff`, log detenido en `Waiting for application startup` | MLflow apagado: la carga del modelo bloquea el arranque y la startupProbe mata el Pod | `bash scripts/demo.sh` |
| `kubectl: command not found` | `~/.local/bin` no está en el PATH | `export PATH=$HOME/.local/bin:$PATH` |
| `connection reset` en `localhost:5000` | `localhost` resuelve a IPv6 | Usa `127.0.0.1:5000` |
| `network kind not found` | No se ha creado el clúster | `kind create cluster --name mlops` |
| `Failed to resolve 'mlflow'` en `entrenar.sh` | Contenedor `mlflow` apagado | `bash scripts/iniciar_mlflow.sh` |
| MLflow rechaza la petición por el nombre del host | Host fuera de `--allowed-hosts` | Revisa `scripts/iniciar_mlflow.sh` |
| Tras escalar, un Pod nuevo no recibe tráfico | kube-proxy tarda unos segundos en actualizarse | Espera 5 s |
| No se ven los runs en MLflow | La interfaz abre en modo GenAI | Elige **Model training** |

## 6. Apagar y limpiar

```bash
docker stop mlflow mlops-control-plane             # libera memoria; se recupera con scripts/demo.sh
kubectl delete -f k8s/ && kubectl delete configmap riesgo-ui   # quita solo los recursos del proyecto
kind delete cluster --name mlops                   # borra el clúster
```

`mlflow-data/` guarda todos los runs y modelos y no está en Git: si lo borras, debes repetir el entrenamiento y el registro.
