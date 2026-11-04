# Servicio de inferencia de riesgo crediticio. El modelo NO va dentro de la imagen:
# se descarga al arrancar desde el Model Registry de MLflow (models:/riesgo-credito@champion).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN useradd --create-home --uid 10001 appuser
USER appuser
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 MODELO_NOMBRE=riesgo-credito MODELO_ALIAS=champion
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
