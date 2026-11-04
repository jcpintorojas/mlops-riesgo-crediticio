# Documento de arquitectura — MLOps de riesgo crediticio

**Proyecto final · Módulo 14: ML-Ops y Puesta en Producción · MIA-V1E4 (UAGRM-SOE)**
**Autor:** Julio Cesar Pinto Rojas (trabajo individual)

## 1. Problema y datos

| Aspecto | Decisión |
|---|---|
| Problema | Clasificación binaria: ¿el cliente **incumplirá el pago del próximo mes**? |
| Dataset | *Default of Credit Card Clients* (Yeh, 2009, UCI ML Repository, [doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H)): 30.000 clientes, 23 variables, 22,1 % de morosos |
| Variables | 14 numéricas (límite de crédito, edad, 6 facturas, 6 pagos) y 9 categóricas (sexo, educación, estado civil, 6 estados de pago mensuales) |
| División | 80 % entrenamiento (24.000) / 20 % prueba (6.000), estratificada, semilla 42 (`src/datos.py`) |
| Métrica principal | **ROC AUC**: con clases desbalanceadas la *accuracy* engaña (predecir "cumple" siempre da 78 %); el AUC mide si el modelo ordena bien a los clientes por riesgo, que es lo que usa el negocio para decidir |
| Reproducibilidad | `src/descargar_datos.py` descarga el archivo oficial y registra su md5 (`4acdad6b…`); cada run guarda en MLflow el md5 de los datos y el commit de Git |

## 2. Vista general

```
                 ┌────────────────────────── red Docker "kind" ───────────────────────────┐
  Desarrollador  │                                                                         │
  ─────────────► │  Contenedor riesgo-mlflow:3.14.0          Clúster kind "mlops" (K8s 1.37)│
  entrenar.sh    │  ┌──────────────────────────┐            ┌─────────────────────────────┐│
  registrar.py   │  │ MLflow server :5000      │◄── carga ──│ Deployment riesgo-api (x3)  ││
                 │  │  · Tracking (SQLite)     │  @champion │  Pod ─ Pod ─ Pod  :8000     ││
  Navegador ───► │  │  · Model Registry        │            │      ▲  Service NodePort 30090
  127.0.0.1:5000 │  │  · Artifact store        │            │      │                      ││
                 │  └──────────────────────────┘            │ Deployment riesgo-ui (nginx)││
  Navegador ─────┼──────────────────────────────────────────►│  /api → Service riesgo-api  ││
  <nodo>:30091   │                                          │  Service NodePort 30091     ││
                 │                                          └─────────────────────────────┘│
                 └─────────────────────────────────────────────────────────────────────────┘
```

| Componente | Tecnología | Archivo |
|---|---|---|
| Entorno de experimentación | Imagen `riesgo-mlflow:3.14.0` (Python 3.12, MLflow 3.14.0, scikit-learn 1.9.0) | `docker/mlflow/` |
| Servidor MLflow | `mlflow server` con backend SQLite y `--serve-artifacts` | `scripts/iniciar_mlflow.sh` |
| Entrenamiento y registro | 6 runs + registro de 2 versiones + alias | `src/entrenar.py`, `src/registrar.py` |
| Servicio de inferencia | FastAPI + Pydantic, imagen `riesgo-api:2.0` | `api/` |
| Orquestación | Kubernetes (kind): Deployment 3 réplicas + Service NodePort | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Monitoreo de drift | KS + PSI (data drift), AUC por lote (concept drift) | `drift/` |
| Interfaz (extra) | HTML + JavaScript servido por nginx dentro del clúster | `ui/`, `k8s/ui.yaml` |

## 3. Fase 1 — Experimentación y trazabilidad con MLflow

**Runs del experimento `riesgo-crediticio`** (evidencia: `evidencias/mlflow-01-runs-y-registro.txt`):

| Run | Algoritmo | Hiperparámetros | ROC AUC | F1 | Accuracy |
|---|---|---|---|---|---|
| **gb_lr-0.1_depth-3** | Gradient boosting | learning_rate 0,1 · 150 árboles · profundidad 3 | **0,7787** | 0,4726 | 0,8188 |
| gb_lr-0.05_depth-4 | Gradient boosting | learning_rate 0,05 · 300 árboles · profundidad 4 · subsample 0,8 | 0,7765 | 0,4678 | 0,8168 |
| rf_n-300_depth-10 | Random forest | 300 árboles · profundidad 10 · min_samples_leaf 5 | 0,7749 | 0,4612 | 0,8173 |
| rf_n-100_depth-6 | Random forest | 100 árboles · profundidad 6 | 0,7708 | 0,4556 | 0,8172 |
| logreg_C-1.0 | Regresión logística | C = 1,0 (con StandardScaler) | 0,7076 | 0,3553 | 0,8077 |
| logreg_C-0.01 | Regresión logística | C = 0,01 (con StandardScaler) | 0,7067 | 0,3480 | 0,8070 |

Cada run guarda parámetros, 5 métricas, etiquetas (`algoritmo`, `dataset`, `datos_md5`, `git_commit`, tamaños de train/test) y el modelo empaquetado como **MLflow Model** (`MLmodel`, `requirements.txt`, firma de entrada/salida, `input_example`). No se usa pickle/joblib a mano.

**Por qué se eligió el modelo desplegado:** el gradient boosting con learning rate 0,1 tiene el mejor AUC y también el mejor F1. Los modelos de árboles superan a la regresión logística en ~0,07 de AUC porque captan relaciones no lineales (por ejemplo, el efecto de los meses de atraso no es proporcional). La diferencia entre los 4 modelos de árboles es pequeña (< 0,008); se prefirió el de 150 árboles frente al de 300 porque es más liviano a igual rendimiento.

**Model Registry** (modelo `riesgo-credito`):

| Versión | Run de origen | run_id | Alias | Rol |
|---|---|---|---|---|
| 1 | logreg_C-1.0 | `bc5c120a34074e23b70c79daed8229c5` | — | Modelo base, simple e interpretable; queda como respaldo |
| **2** | **gb_lr-0.1_depth-3** | **`662559032df74600bfa1deab64b38922`** | **`@champion`** | **Modelo desplegado en el clúster** |

El servicio carga `models:/riesgo-credito@champion`, nunca una ruta de archivo. Se usan **alias** (no *stages*, que están deprecados en MLflow 3): mover `@champion` a otra versión cambia el modelo servido sin tocar el código ni la imagen. Esto se demostró moviendo el alias a la versión 1 y de regreso a la 2 (`evidencias/k8s-05-alias.txt`). Cada respuesta de la API incluye `version` y `run_id`, así que cualquier predicción se puede rastrear hasta el experimento exacto que la produjo.

## 4. Fase 2 — Contenerización

- **Dockerfile propio** (`api/Dockerfile`): base `python:3.12-slim`, dependencias primero (aprovecha la caché de capas), usuario no root (`appuser`, uid 10001), `.dockerignore`.
- **Versiones fijadas** (`api/requirements.txt`): `mlflow-skinny==3.14.0` (cliente de MLflow sin el servidor ni la interfaz web: el servicio no los necesita), `scikit-learn==1.9.0`, `skops==0.16.0`, `numpy==2.5.1`, `pandas==2.3.3`, `fastapi==0.115.6`, `uvicorn==0.32.1`, `pydantic==2.10.3`. Las versiones de ML coinciden con las de entrenamiento para evitar el *version mismatch*.
- **Arranque sin pasos manuales:** al iniciar, la API lee la variable `MLFLOW_TRACKING_URI`, pide al registro la versión con alias `champion` y la descarga. El modelo **no** se copia dentro de la imagen: así la misma imagen sirve cualquier versión y la trazabilidad queda en el registro.
- **Endpoints:** `POST /predict` (validación con Pydantic: tipos, rangos y valores permitidos; entrada inválida → HTTP 422), `GET /health` (liveness), `GET /ready` (readiness: 503 hasta que el modelo esté cargado), `GET /info`, `/docs` (Swagger).
- **Serialización segura:** MLflow 3.14 guarda los modelos de scikit-learn con *skops*; se declaró como confiable solo el tipo `sklearn.tree._tree.Tree`, porque los modelos se crean en el mismo proyecto.

## 5. Fase 3 — Kubernetes

| Decisión | Valor | Por qué |
|---|---|---|
| Réplicas | 3 | Tolera la caída de un Pod con margen; exigencia del proyecto |
| Nombre del Pod | Variable `POD_NAME` desde la *Downward API* | Demuestra qué réplica atendió cada petición |
| Service | NodePort 30090 → 8000 | Acceso desde fuera del clúster local (kind) |
| Recursos | requests = limits: CPU 250m, memoria 512 Mi | QoS *Guaranteed*; medido ~190 MiB en reposo, margen para cargar el modelo |
| startupProbe | `/health`, hasta 2 min | Da tiempo a descargar el modelo sin que la liveness lo reinicie |
| readinessProbe | `/ready` cada 5 s | Un Pod no recibe tráfico hasta tener el modelo cargado |
| livenessProbe | `/health` cada 10 s, 3 fallos | Reinicia un proceso colgado |
| Estrategia | RollingUpdate, `maxUnavailable: 0`, `maxSurge: 1` | Nunca hay menos de 3 réplicas listas durante una actualización |
| Conexión a MLflow | `http://mlflow:5000` (contenedor en la red Docker `kind`) | Los Pods resuelven el nombre por el DNS del clúster |

**Demostraciones** (registro de terminal en `evidencias/`):

| Prueba | Resultado |
|---|---|
| 3 réplicas en Running | `k8s-01`: `3/3`, cada Pod registra en su log "modelo riesgo-credito@champion -> versión 2" |
| Balanceo | `k8s-02`: 30 peticiones sucesivas atendidas por los 3 Pods |
| Autorreparación | `k8s-03`: se eliminó un Pod con carga continua; Kubernetes creó otro y el servicio respondió **todas** las peticiones (≈185 con HTTP 200, 0 errores) |
| Escalado | `k8s-04`: 3 → 5 réplicas (50 peticiones repartidas entre los 5 Pods) → 3 |
| Cambio de versión por alias | `k8s-05`: `@champion` → v1 + `rollout restart` ⇒ los Pods sirven la v1; de regreso a v2 |

## 6. Fase 4 — Drift

### 6.1 Data drift (`drift/data_drift.py`, `drift/test_drift.py`)

Se compara **cada variable** del lote nuevo contra el baseline (las 24.000 filas de entrenamiento):

| Tipo de variable | Prueba | Umbral de alerta | De dónde sale el umbral |
|---|---|---|---|
| Numérica continua (14) | **Kolmogorov-Smirnov** de dos muestras | p < 0,05 / 14 = **0,0036** | Nivel de significancia clásico 5 % con **corrección de Bonferroni**: al hacer 14 pruebas a la vez, sin corregir habría ~50 % de probabilidad de al menos una falsa alarma en un lote sano. KS se eligió porque compara la distribución completa sin suponer normalidad (los montos son muy asimétricos) |
| Categórica (9) | **PSI** (índice de estabilidad poblacional) | **≥ 0,20 rojo**; 0,10–0,20 amarillo | Convención de la industria vista en clase 7 (PSI 0,15 "alerta menor", > 0,2 "drift detectado"). PSI se eligió porque compara la proporción de cada categoría y es interpretable |

Un lote está en **ROJO** si al menos una variable está en rojo. Resultados (`evidencias/drift-01-data-drift.txt`):

- **Lote del mismo origen** (las 6.000 filas de prueba): **VERDE**, las 23 variables estables. Detalle útil: `AGE` tiene p = 0,023, que sin la corrección de Bonferroni habría sido una falsa alarma.
- **Lote derivado** (se inyectó drift: límites de crédito × 0,6, clientes 6 años más jóvenes, facturas × 1,4, 25 % con 2 meses de atraso, 30 % con educación secundaria): **ROJO**, exactamente en las 5 variables alteradas (`LIMIT_BAL`, `AGE`, `BILL_AMT1`, `PAY_1`, `EDUCATION`); las demás siguen en verde.
- **Prueba automatizada:** `pytest drift/test_drift.py` → 3 pruebas pasan (verde con el mismo origen, rojo con el derivado, variables no tocadas estables).

### 6.2 Concept drift (`drift/concept_drift.py`)

- **Escenario simulado:** una crisis de endeudamiento. Desde el lote 5, una fracción creciente de clientes (15 % → 90 %) pasa a regirse por una regla nueva: incumple quien usa más del 70 % de su límite (`BILL_AMT1 / LIMIT_BAL > 0,7`), aunque venga pagando al día. **Las entradas no cambian**: cambia P(Y|X).
- **Medición:** 12 lotes sucesivos de 500 clientes; se calcula el AUC del modelo en producción (`@champion`) en cada lote y se registra en MLflow (experimento `monitoreo-concept-drift`, una métrica por *step*). Gráfica: `evidencias/capturas/drift-concept-drift.png`.
- **Criterio de reentrenamiento:**
  - Umbral = AUC de referencia − 2σ = 0,7787 − 2 × 0,0232 = **0,732**. σ es la variación normal del AUC en lotes sanos de 500 clientes (200 remuestreos del conjunto de prueba): por debajo de 2σ ya no es mala suerte del lote.
  - Alarma si el AUC queda bajo el umbral en **3 lotes consecutivos**: evita reentrenar por un lote raro aislado.
  - Resultado: el AUC cae de 0,78 a 0,60; la alarma se dispara en el **lote 8**.
- **Observación clave:** el monitor de data drift estuvo en **VERDE en los 12 lotes**. El concept drift no se ve mirando solo las entradas; por eso se necesitan las etiquetas reales.

### 6.3 El problema del retraso de etiquetas

En crédito la etiqueta real ("¿pagó?") llega cuando vence la cuota, uno o varios meses después de la predicción. Mientras tanto el equipo haría:

1. **Vigilar lo que sí está disponible al instante:** data drift de las entradas (KS/PSI) y la distribución de las predicciones (probabilidad media, % de clientes clasificados de alto riesgo).
2. **Usar etiquetas tempranas (proxies):** por ejemplo, el atraso en el primer pago o la mora a 30 días, que llegan antes que el incumplimiento definitivo.
3. **Guardar cada predicción con su `run_id`, versión y datos de entrada**, para calcular el AUC real en cuanto lleguen las etiquetas (la API ya devuelve versión y run).
4. **Comparar la tasa de morosidad observada con la predicha** por cohortes, cuando empiezan a llegar etiquetas: en la simulación la tasa real subió de 0,22 a 0,33 mientras la probabilidad media predicha se quedó en ~0,22.
5. **Reentrenar también por calendario** (por ejemplo cada trimestre) como red de seguridad, y desplegar el nuevo modelo como `@challenger` antes de moverlo a `@champion`.

## 7. Cómo ejecutar

Guía completa de arranque y solución de problemas: [`OPERACION.md`](OPERACION.md). Guía para agentes de IA: [`../AGENTS.md`](../AGENTS.md).

```bash
bash scripts/iniciar_mlflow.sh                            # MLflow en http://127.0.0.1:5000
bash scripts/entrenar.sh src/descargar_datos.py            # datos
bash scripts/entrenar.sh src/entrenar.py                   # 6 runs
bash scripts/entrenar.sh src/registrar.py                  # registro + @champion
bash scripts/desplegar.sh                                  # imagen + Kubernetes + interfaz
bash scripts/demo_k8s.sh                                   # demostraciones de la fase 3
bash scripts/entrenar.sh drift/data_drift.py               # data drift
bash scripts/entrenar.sh -m pytest -v drift/test_drift.py  # pruebas automatizadas
bash scripts/entrenar.sh drift/concept_drift.py            # concept drift + gráfica
```

## 8. Reparto del trabajo

| Integrante | Componentes |
|---|---|
| Julio Cesar Pinto Rojas | Todos (trabajo individual): datos, MLflow, API y Docker, Kubernetes, drift, interfaz y documentación |

## 9. Limitaciones y siguientes pasos

- El servidor MLflow usa SQLite y disco local; en un equipo real se usaría PostgreSQL y un almacenamiento de objetos (S3/MinIO), como se vio en clase 5.
- Los Pods que ya cargaron el modelo siguen atendiendo si MLflow se cae (medido: 20 de 20 peticiones con HTTP 200). Pero la carga bloquea el arranque: un Pod nuevo no responde `/health` sin MLflow, la startupProbe lo reinicia a los 2 minutos y entra en `CrashLoopBackOff` hasta que MLflow vuelve. Mejora pendiente: cargar el modelo en segundo plano para que quede vivo pero no listo.
- El contenedor de MLflow no tiene política de reinicio: tras reiniciar la máquina hay que levantarlo con `scripts/demo.sh`.
- Un cambio de alias requiere `kubectl rollout restart` para que los Pods recarguen el modelo; se podría automatizar con un pipeline de CD (CD4ML, clase 6).
- Faltaría un pipeline de CI que corra `pytest`, construya la imagen y la publique en un registro de contenedores.
