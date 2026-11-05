"""Servicio de inferencia de riesgo crediticio (Fase 2).

Al arrancar carga el modelo desde el Model Registry de MLflow por su alias
(models:/riesgo-credito@champion), no desde una ruta de archivo. Cada respuesta indica qué versión y qué run
del registro la produjo, y en qué Pod de Kubernetes se atendió (variable POD_NAME, Downward API).
"""
# os: lee variables de entorno como POD_NAME y MODELO_ALIAS.
import os
# socket: obtiene el nombre de la máquina si POD_NAME no existe.
import socket
# Literal: limita un campo a una lista cerrada de valores.
from typing import Literal

# mlflow: carga el modelo desde el Model Registry.
import mlflow
# numpy: convierte la salida del modelo en un arreglo.
import numpy as np
# pandas: el modelo recibe los datos como una tabla (DataFrame).
import pandas as pd
# FastAPI crea la API; HTTPException devuelve errores HTTP.
from fastapi import FastAPI, HTTPException
# MlflowClient consulta qué versión y qué run tiene un alias.
from mlflow import MlflowClient
# Pydantic valida los datos que llegan a la API.
from pydantic import BaseModel, Field

# Modelo a cargar. Viene del Deployment; si falta, usa este valor por defecto.
NOMBRE = os.getenv("MODELO_NOMBRE", "riesgo-credito")
# Alias del registro. Cambiarlo no exige tocar el código.
ALIAS = os.getenv("MODELO_ALIAS", "champion")
# Nombre del Pod, inyectado por Kubernetes con la Downward API.
POD = os.getenv("POD_NAME", socket.gethostname())
# Desde esta probabilidad, el cliente se clasifica como "incumple".
UMBRAL = float(os.getenv("UMBRAL_RIESGO", "0.5"))

# Estado del proceso: el modelo en memoria, su versión, su run y el último error.
estado = {"modelo": None, "version": None, "run_id": None, "error": None}


# Se ejecuta al arrancar el Pod.
def cargar_modelo():
    try:
        # Pregunta al registro qué versión y qué run tiene el alias, para la trazabilidad.
        version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS)
        # Descarga el modelo por su referencia, nunca por una ruta de archivo.
        estado["modelo"] = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@{ALIAS}")
        # Guarda la versión y el run para incluirlos en cada respuesta.
        estado.update(version=version.version, run_id=version.run_id, error=None)
        # Deja en los logs qué modelo cargó este Pod.
        print(f"[{POD}] modelo {NOMBRE}@{ALIAS} -> versión {version.version} (run {version.run_id})", flush=True)
    # Si la carga termina en error, /ready responde 503 y el Pod no recibe tráfico.
    except Exception as e:  # el Pod queda vivo pero NO listo: la readinessProbe le impide recibir tráfico
        estado["error"] = str(e)
        print(f"[{POD}] no se pudo cargar el modelo: {e}", flush=True)


# Crea la aplicación; también genera la documentación Swagger en /docs.
app = FastAPI(title="API de riesgo crediticio", version="2.0.0",
              description="Predice la probabilidad de que un cliente incumpla su próximo pago.")


# Carga el modelo antes de atender peticiones. Sin MLflow, esto bloquea el arranque del Pod.
@app.on_event("startup")
def al_iniciar():
    cargar_modelo()


# Valores válidos de PAY_*: -2 sin consumo, -1 al día, 0 pago mínimo, 1 a 9 meses de atraso.
Estado = Literal[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# Contrato de entrada: las 23 variables. Un dato inválido recibe HTTP 422 y no llega al modelo.
class Cliente(BaseModel):
    # gt=0: el límite de crédito debe ser mayor que cero.
    LIMIT_BAL: float = Field(gt=0, description="Límite de crédito (NT$)")
    # Solo acepta los códigos definidos por el dataset.
    SEX: Literal[1, 2] = Field(description="1 = hombre, 2 = mujer")
    # ge y le fijan el mínimo y el máximo permitidos.
    EDUCATION: int = Field(ge=0, le=6, description="1 posgrado, 2 universidad, 3 secundaria, 4-6 otros")
    # Estado civil codificado como número.
    MARRIAGE: int = Field(ge=0, le=3, description="1 casado, 2 soltero, 3 otros")
    # Edad entre 18 y 100 años.
    AGE: int = Field(ge=18, le=100)
    # Estado de pago de cada mes; PAY_1 es el más reciente.
    PAY_1: Estado = Field(description="Estado de pago del último mes: -1 al día, 1..9 meses de atraso")
    PAY_2: Estado
    PAY_3: Estado
    PAY_4: Estado
    PAY_5: Estado
    PAY_6: Estado
    # Montos facturados de los últimos 6 meses. Pueden ser negativos si hay saldo a favor.
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    # Montos pagados de los últimos 6 meses. ge=0: no pueden ser negativos.
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


# Usada por la livenessProbe y la startupProbe: indica si el proceso responde.
@app.get("/health")
def health():
    """Liveness: el proceso responde."""
    return {"estado": "vivo", "pod": POD}


# Usada por la readinessProbe: indica si este Pod puede recibir tráfico.
@app.get("/ready")
def ready():
    """Readiness: solo está listo si el modelo del registro se cargó."""
    # Sin modelo cargado, responde 503 y el Service deja de enviarle peticiones.
    if estado["modelo"] is None:
        raise HTTPException(503, f"modelo no cargado: {estado['error']}")
    return {"estado": "listo", "pod": POD, "version": estado["version"]}


# Muestra el modelo servido por este Pod. Sirve para comprobar el alias.
@app.get("/info")
def info():
    return {"pod": POD, "modelo": NOMBRE, "alias": ALIAS, "version": estado["version"], "run_id": estado["run_id"],
            "umbral": UMBRAL}


# Ruta principal. FastAPI valida el cuerpo con Cliente antes de entrar a esta función.
@app.post("/predict")
def predict(cliente: Cliente):
    # Protección por si llega una petición sin modelo cargado.
    if estado["modelo"] is None:
        raise HTTPException(503, "modelo no cargado")
    # Convierte el cliente en una tabla de una fila con tipos enteros, como en entrenamiento.
    datos = pd.DataFrame([cliente.model_dump()]).astype("int64")
    # Ejecuta el modelo cargado desde MLflow.
    salida = np.asarray(estado["modelo"].predict(datos))  # pyfunc usa predict_proba: columnas [P(cumple), P(incumple)]
    # Toma la columna 1, P(incumple); admite también una salida de una dimensión.
    proba = float(salida[0, 1] if salida.ndim == 2 else salida[0])
    return {
        # Resultado principal, redondeado a 4 decimales.
        "probabilidad_incumplimiento": round(proba, 4),
        # Aplica el umbral para decidir la clase.
        "prediccion": "incumple" if proba >= UMBRAL else "cumple",
        # Trazabilidad: esta predicción se puede seguir hasta su versión y su run.
        "modelo": {"nombre": NOMBRE, "alias": ALIAS, "version": estado["version"], "run_id": estado["run_id"]},
        # Muestra qué réplica atendió: así se demuestra el balanceo.
        "pod": POD,
    }
