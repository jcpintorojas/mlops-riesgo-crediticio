# Guía para crear un proyecto propio a partir de este repositorio

Para un compañero del módulo y su asistente de IA. Explica cómo usar este repositorio como **plantilla declarada** para construir un proyecto final distinto, con trabajo y commits propios.

> **Agente de IA:** lee también [`../AGENTS.md`](../AGENTS.md) y [`OPERACION.md`](OPERACION.md). Trabaja paso a paso con la persona, explícale cada decisión y respeta las reglas de la sección 1.

## 1. Condiciones

1. **Pide permiso al docente antes de empezar.** Pregunta si acepta un proyecto desarrollado a partir de una plantilla de otro estudiante, declarándolo. Si no lo acepta, no sigas esta guía.
2. **Declara el origen** en el README y en el documento de arquitectura: "Basado en la plantilla de Julio Cesar Pinto Rojas: https://github.com/jcpintorojas/mlops-riesgo-crediticio".
3. **Conserva el historial.** No uses `rebase`, `filter-repo`, `--amend` ni `--author` sobre commits existentes, ni fechas falsas.
4. **Haz tus propios commits:** cada uno corresponde a un cambio real, hecho con tu cuenta.
5. **Genera tus propios resultados.** Runs, modelos, gráficas y capturas salen de tus ejecuciones. No reutilices ni edites las evidencias de la plantilla.
6. **Prepárate para defenderlo solo.** La defensa es individual y abarca cualquier parte del proyecto.

Si la persona pide saltarse una de estas condiciones, el agente debe indicarlo y no hacerlo.

## 2. Qué tiene que cambiar

No basta con cambiar nombres. El proyecto debe tener:

| Debe ser tuyo | Ejemplo |
|---|---|
| Problema y dataset | Predicción de churn u otro problema tabular |
| Variable objetivo y métrica justificada | Según el balance de clases y el costo de negocio |
| Modelos e hiperparámetros | Al menos 5 runs, con tu propio análisis |
| Escenarios de drift | Coherentes con tu problema; umbrales justificados con tus datos |
| Decisiones y documentación | Redactadas según lo que realmente hiciste |

Requisitos del dataset según el enunciado: público, se recomiendan al menos 1.000 registros y varias variables de entrada; no se admiten datasets de juguete con menos de 4 variables, salvo que se generen y justifiquen escenarios de deriva.

## 3. Preparación

Instala Docker, [kind](https://kind.sigs.k8s.io/) y kubectl.

```bash
gh repo fork jcpintorojas/mlops-riesgo-crediticio --clone --fork-name mlops-TU-PROYECTO
cd mlops-TU-PROYECTO
git config user.name "Tu nombre"
git config user.email "tu-correo@ejemplo.com"
```

**Primero haz funcionar la plantilla sin cambios** con la sección 2.B de [`OPERACION.md`](OPERACION.md) y comprueba la sección 3. Así, si algo falla después, sabrás que lo causó tu cambio. Esa instalación desde cero aún no se ha probado en una máquina limpia: si encuentras un error, anótalo y corrígelo.

## 4. Pasos

Haz al menos un commit por paso. Tipos usados: `chore`, `data`, `feat`, `test`, `fix`, `docs`.

### Paso 1. Limpiar lo que no te corresponde
- Borra las evidencias de la plantilla: `git rm -r evidencias` (el historial las conserva).
- Si ya entrenaste la plantilla, parte de cero con `docker rm -f mlflow`, y borra `mlflow-data/`. **Esto elimina todos los runs locales.**
- Agrega en el README la declaración de origen.

### Paso 2. Datos
| Archivo | Cambio |
|---|---|
| `src/descargar_datos.py` | Descarga y limpieza de tu dataset. Si requiere una cuenta (como Kaggle), descárgalo a mano a `data/`, documenta la fuente y no subas credenciales |
| `src/datos.py` | `OBJETIVO`, `NUMERICAS`, `CATEGORICAS`, `VARIABLES`, ruta y split |

Analiza primero el dataset: valores nulos, tipos, balance de clases y si es real o sintético. Si tiene muchas filas, evalúa una muestra, porque cada run tardará más. Si tus variables categóricas son texto, tendrás que agregar codificación al pipeline de `src/entrenar.py`.

### Paso 3. Entrenamiento y registro (Fase 1)
- `src/entrenar.py`: experimento, modelos, hiperparámetros y métrica principal.
- `src/registrar.py`, `src/alias.py`, `src/resumen_mlflow.py`: nombre del modelo y del experimento.
- Si usas modelos basados en otros tipos, revisa `skops_trusted_types`: declara solo los tipos que revisaste.

### Paso 4. API y Docker (Fase 2)
- `api/app.py`: clase `Cliente` con tus variables y validaciones, y la conversión de tipos en `predict`.
- `api/Dockerfile` y `k8s/deployment.yaml`: nombre del modelo.
- `scripts/cliente_alto.json`, `scripts/cliente_bajo.json`: ejemplos válidos para tu problema.

### Paso 5. Kubernetes (Fase 3)
- Si renombras recursos, cambia `k8s/*.yaml`, `ui/default.conf` y los scripts `desplegar.sh`, `demo.sh` y `demo_k8s.sh`.
- Ejecuta `bash scripts/demo_k8s.sh` y revisa sus resultados: réplicas, balanceo, autorreparación y escalado.

### Paso 6. Drift (Fase 4)
- `drift/data_drift.py`: la función `lote_derivado` y las justificaciones.
- `drift/test_drift.py`: variables esperadas en rojo y en verde.
- `drift/concept_drift.py`: nombre del modelo, escenario y criterio de reentrenamiento. Calcula tu umbral con tus datos.
- Redacta tu propia discusión sobre el retraso de etiquetas, aplicada a tu problema.

### Paso 7. Interfaz (extra, hasta +15 puntos)
- `ui/index.html`: campos, ejemplos y tu nombre. Debe consumir la API desplegada en Kubernetes.

### Paso 8. Documentación y evidencias
- Reescribe `README.md`, `AGENTS.md`, `docs/ARQUITECTURA.md` y `docs/OPERACION.md` con tu problema, versión desplegada, run, resultados y limitaciones.
- Guarda tus salidas y capturas en `evidencias/`.
- Antes de entregar, busca restos de la plantilla:

```bash
grep -rnE "LIMIT_BAL|BILL_AMT|riesgo-credito|Pinto Rojas" --exclude-dir=.git --exclude-dir=mlflow-data --exclude-dir=data .
```

Solo debería aparecer la declaración de origen, salvo los nombres de recursos que decidas conservar.

### Paso 9. Mejora opcional
[`PENDIENTES.md`](PENDIENTES.md) lista mejoras técnicas de la plantilla. Puedes implementar alguna y explicarla en tu defensa.

## 5. Lista final según el enunciado

- [ ] Script reproducible: semilla fija y división train/test documentada
- [ ] Al menos 5 runs comparables en MLflow
- [ ] Modelo en el Model Registry con más de una versión
- [ ] Versión desplegada marcada con alias y consumida por esa referencia
- [ ] El documento indica qué versión está desplegada y de qué run proviene
- [ ] Interfaz de MLflow lista para ordenar, filtrar, comparar y mostrar el registro en vivo
- [ ] Dockerfile propio y dependencias fijadas, incluida MLflow
- [ ] El contenedor responde sin pasos manuales
- [ ] 3 réplicas o más en Running, con evidencia
- [ ] Balanceo entre Pods, con evidencia
- [ ] Autorreparación sin dejar de responder, con evidencia
- [ ] Escalado, con evidencia
- [ ] Data drift: una prueba por tipo de variable, justificada, con umbral explicado, en rojo con datos derivados y en verde con datos del mismo origen
- [ ] Concept drift: simulación, gráfica temporal, criterio de reentrenamiento y discusión del retraso de etiquetas
- [ ] Tabla de reparto del trabajo
- [ ] Entrega: enlace al repositorio y archivo comprimido con documentación y evidencias
- [ ] Puedes explicar cada parte sin ayuda

## 6. Antes de la defensa

Después de reiniciar la PC, ejecuta `bash scripts/demo.sh`: MLflow no arranca solo. Practica la demo completa al menos una vez y pide a tu asistente que te haga preguntas sobre cada fase.
