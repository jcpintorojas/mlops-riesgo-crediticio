# Pendientes del proyecto

Los requisitos del enunciado están cubiertos. Esta lista reúne lo que falta para la entrega y las mejoras opcionales.

## Entrega

| Pendiente | Responsable | Estado |
|---|---|---|
| Defensa oral | Julio | Pendiente |
| Subir a Moodle el enlace del repositorio y `T08-documentacion-y-evidencias.zip` (cierra 08/11/2026 23:59) | Julio | Pendiente |

## Mejoras técnicas

Ninguna es necesaria para cumplir el enunciado.

| Prioridad | Mejora | Problema actual | Archivos |
|---|---|---|---|
| Alta | Cargar el modelo en segundo plano | La carga bloquea el arranque: sin MLflow, un Pod nuevo entra en `CrashLoopBackOff` en vez de quedar vivo pero no listo | `api/app.py` |
| Alta | Política de reinicio de MLflow (`--restart unless-stopped`) | Tras reiniciar la PC, MLflow queda apagado | `scripts/iniciar_mlflow.sh` |
| Alta | Probar la instalación desde cero en una máquina limpia | Solo se probó un clon nuevo: descarga de datos y pruebas de drift | `docs/OPERACION.md` |
| Media | CI con GitHub Actions: `pytest` y construcción de imágenes | No hay integración continua | `.github/workflows/` (nuevo) |
| Media | Flujo `@challenger` antes de promover a `@champion` | La promoción es manual y sin comparación automática | `src/registrar.py`, `src/alias.py` |
| Media | Recarga del modelo sin reiniciar Pods | Cambiar el alias exige `rollout restart` | `api/app.py` |
| Media | Guardar predicciones con versión y run | La medición diferida se describe, pero no está implementada | `api/app.py` |
| Baja | MLflow con PostgreSQL y almacenamiento S3/MinIO | SQLite y disco local solo sirven para una persona | `scripts/iniciar_mlflow.sh` |
| Baja | Reporte HTML con Evidently | El drift se revisa en terminal | `drift/` |
| Baja | Métricas técnicas con Prometheus y Grafana | No hay métricas de latencia ni errores | `api/app.py`, `k8s/` |

Antes de cambiar `api/app.py` o `k8s/`, ejecuta `bash scripts/demo_k8s.sh` para comprobar que no se rompan las demostraciones. Esa prueba sobrescribe `evidencias/k8s-*`.
