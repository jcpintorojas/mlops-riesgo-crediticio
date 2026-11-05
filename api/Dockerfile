# Servicio de inferencia de riesgo crediticio. El modelo NO va dentro de la imagen:
# se descarga al arrancar desde el Model Registry de MLflow (models:/riesgo-credito@champion).

# Parte de una imagen con Linux mínimo y Python 3.12.
FROM python:3.12-slim

# Evita archivos .pyc, muestra los logs al instante y no guarda la caché de pip.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Crea la carpeta /app y ejecuta desde ahí las instrucciones siguientes.
WORKDIR /app

# Copia la lista de librerías antes que el código para aprovechar la caché de capas.
COPY requirements.txt .

# Instala las librerías con sus versiones fijadas. Se ejecuta al construir la imagen.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la API. Si solo cambia este archivo, las librerías no se reinstalan.
COPY app.py .

# Crea un usuario sin privilegios de administrador.
RUN useradd --create-home --uid 10001 appuser

# Ejecuta el contenedor con ese usuario, por seguridad.
USER appuser

# Indica dónde está MLflow y qué modelo y alias debe cargar la API.
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 MODELO_NOMBRE=riesgo-credito MODELO_ALIAS=champion

# Documenta que la API usa el puerto 8000. Por sí solo no lo publica.
EXPOSE 8000

# Arranca la API cada vez que inicia el contenedor. 0.0.0.0 acepta conexiones desde fuera de él.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
