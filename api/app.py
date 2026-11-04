"""Servicio de inferencia de riesgo crediticio (Fase 2).

Al arrancar carga el modelo desde el Model Registry de MLflow por su alias
(models:/riesgo-credito@champion), no desde una ruta de archivo. Cada respuesta indica qué versión y qué run
del registro la produjo, y en qué Pod de Kubernetes se atendió (variable POD_NAME, Downward API).
"""
import os
import socket
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

NOMBRE = os.getenv("MODELO_NOMBRE", "riesgo-credito")
ALIAS = os.getenv("MODELO_ALIAS", "champion")
POD = os.getenv("POD_NAME", socket.gethostname())
UMBRAL = float(os.getenv("UMBRAL_RIESGO", "0.5"))

estado = {"modelo": None, "version": None, "run_id": None, "error": None}


def cargar_modelo():
    try:
        version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS)
        estado["modelo"] = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@{ALIAS}")
        estado.update(version=version.version, run_id=version.run_id, error=None)
        print(f"[{POD}] modelo {NOMBRE}@{ALIAS} -> versión {version.version} (run {version.run_id})", flush=True)
    except Exception as e:  # el Pod queda vivo pero NO listo: la readinessProbe le impide recibir tráfico
        estado["error"] = str(e)
        print(f"[{POD}] no se pudo cargar el modelo: {e}", flush=True)


app = FastAPI(title="API de riesgo crediticio", version="2.0.0",
              description="Predice la probabilidad de que un cliente incumpla su próximo pago.")


@app.on_event("startup")
def al_iniciar():
    cargar_modelo()


Estado = Literal[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Cliente(BaseModel):
    LIMIT_BAL: float = Field(gt=0, description="Límite de crédito (NT$)")
    SEX: Literal[1, 2] = Field(description="1 = hombre, 2 = mujer")
    EDUCATION: int = Field(ge=0, le=6, description="1 posgrado, 2 universidad, 3 secundaria, 4-6 otros")
    MARRIAGE: int = Field(ge=0, le=3, description="1 casado, 2 soltero, 3 otros")
    AGE: int = Field(ge=18, le=100)
    PAY_1: Estado = Field(description="Estado de pago del último mes: -1 al día, 1..9 meses de atraso")
    PAY_2: Estado
    PAY_3: Estado
    PAY_4: Estado
    PAY_5: Estado
    PAY_6: Estado
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@app.get("/health")
def health():
    """Liveness: el proceso responde."""
    return {"estado": "vivo", "pod": POD}


@app.get("/ready")
def ready():
    """Readiness: solo está listo si el modelo del registro se cargó."""
    if estado["modelo"] is None:
        raise HTTPException(503, f"modelo no cargado: {estado['error']}")
    return {"estado": "listo", "pod": POD, "version": estado["version"]}


@app.get("/info")
def info():
    return {"pod": POD, "modelo": NOMBRE, "alias": ALIAS, "version": estado["version"], "run_id": estado["run_id"],
            "umbral": UMBRAL}


@app.post("/predict")
def predict(cliente: Cliente):
    if estado["modelo"] is None:
        raise HTTPException(503, "modelo no cargado")
    datos = pd.DataFrame([cliente.model_dump()]).astype("int64")
    salida = np.asarray(estado["modelo"].predict(datos))  # pyfunc usa predict_proba: columnas [P(cumple), P(incumple)]
    proba = float(salida[0, 1] if salida.ndim == 2 else salida[0])
    return {
        "probabilidad_incumplimiento": round(proba, 4),
        "prediccion": "incumple" if proba >= UMBRAL else "cumple",
        "modelo": {"nombre": NOMBRE, "alias": ALIAS, "version": estado["version"], "run_id": estado["run_id"]},
        "pod": POD,
    }
