"""Pruebas de la API sin MLflow ni Kubernetes: el modelo del registro se reemplaza por un doble de prueba.
USO (desde la raíz del repositorio):  cd api && python -m pytest -v
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as servicio

CLIENTE = json.loads((Path(__file__).parent.parent / "scripts" / "cliente_alto.json").read_text())


class ModeloFalso:
    """Imita al pyfunc de MLflow con predict_proba: devuelve columnas [P(cumple), P(incumple)]."""
    def __init__(self, p_incumple):
        self.p = p_incumple
        self.recibido = None

    def predict(self, datos):
        self.recibido = datos
        return np.array([[1 - self.p, self.p]] * len(datos))


@pytest.fixture(autouse=True)
def estado_limpio():
    servicio.estado.update(modelo=None, version=None, run_id=None, error=None)
    yield
    servicio.estado.update(modelo=None, version=None, run_id=None, error=None)


@pytest.fixture
def cliente_http():
    # Sin "with": no se dispara el lifespan, así que no se intenta contactar a MLflow.
    return TestClient(servicio.app)


def cargar_falso(p_incumple=0.8, version="2", run_id="abc123"):
    servicio.estado.update(modelo=ModeloFalso(p_incumple), version=version, run_id=run_id)


def test_health_responde_aunque_no_haya_modelo(cliente_http):
    r = cliente_http.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "vivo"


def test_ready_es_503_sin_modelo_y_200_con_modelo(cliente_http):
    servicio.estado["error"] = "MLflow no responde"
    r = cliente_http.get("/ready")
    assert r.status_code == 503 and "MLflow no responde" in r.json()["detail"]
    cargar_falso()
    r = cliente_http.get("/ready")
    assert r.status_code == 200 and r.json()["version"] == "2"


def test_predict_es_503_sin_modelo(cliente_http):
    assert cliente_http.post("/predict", json=CLIENTE).status_code == 503


def test_predict_devuelve_probabilidad_version_run_y_pod(cliente_http):
    cargar_falso(p_incumple=0.8)
    r = cliente_http.post("/predict", json=CLIENTE)
    assert r.status_code == 200
    d = r.json()
    assert d["probabilidad_incumplimiento"] == 0.8 and d["prediccion"] == "incumple"
    assert d["modelo"]["version"] == "2" and d["modelo"]["run_id"] == "abc123"
    assert d["pod"] == servicio.POD


def test_predict_respeta_el_umbral(cliente_http):
    cargar_falso(p_incumple=0.3)
    assert cliente_http.post("/predict", json=CLIENTE).json()["prediccion"] == "cumple"


def test_predict_envia_las_23_columnas_en_enteros(cliente_http):
    cargar_falso()
    cliente_http.post("/predict", json=CLIENTE)
    recibido = servicio.estado["modelo"].recibido
    assert list(recibido.columns) == list(servicio.Cliente.model_fields)
    assert len(recibido.columns) == 23
    assert (recibido.dtypes == "int64").all()  # la firma del modelo en MLflow espera enteros


@pytest.mark.parametrize("campo, valor", [("SEX", 3), ("AGE", 10), ("PAY_1", 10), ("LIMIT_BAL", 0), ("PAY_AMT1", -5)])
def test_predict_rechaza_valores_fuera_de_rango(cliente_http, campo, valor):
    cargar_falso()
    r = cliente_http.post("/predict", json={**CLIENTE, campo: valor})
    assert r.status_code == 422


def test_predict_rechaza_campos_faltantes(cliente_http):
    cargar_falso()
    incompleto = {k: v for k, v in CLIENTE.items() if k != "AGE"}
    assert cliente_http.post("/predict", json=incompleto).status_code == 422


def test_cargar_modelo_ok(monkeypatch):
    modelo = ModeloFalso(0.5)
    monkeypatch.setattr(servicio, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda n, a: SimpleNamespace(version="7", run_id="run7")))
    monkeypatch.setattr(servicio.mlflow.pyfunc, "load_model", lambda uri: modelo)
    assert servicio.cargar_modelo() is True
    assert servicio.estado["modelo"] is modelo and servicio.estado["version"] == "7" and servicio.estado["error"] is None


def test_cargar_modelo_falla_sin_romper_el_proceso(monkeypatch):
    def sin_mlflow():
        raise ConnectionError("mlflow:5000 inalcanzable")
    monkeypatch.setattr(servicio, "MlflowClient", sin_mlflow)
    assert servicio.cargar_modelo() is False
    assert servicio.estado["modelo"] is None and "inalcanzable" in servicio.estado["error"]


def test_reintentos_hasta_que_mlflow_vuelve(monkeypatch):
    intentos = []
    def a_veces():
        intentos.append(1)
        return len(intentos) >= 3
    monkeypatch.setattr(servicio, "cargar_modelo", a_veces)
    monkeypatch.setattr(servicio, "REINTENTO_S", 0.01)
    servicio.cargar_con_reintentos(threading.Event())
    assert len(intentos) == 3


def test_reintentos_se_detienen_al_apagar(monkeypatch):
    monkeypatch.setattr(servicio, "cargar_modelo", lambda: False)
    monkeypatch.setattr(servicio, "REINTENTO_S", 0.01)
    detener = threading.Event()
    hilo = threading.Thread(target=servicio.cargar_con_reintentos, args=(detener,))
    hilo.start()
    detener.set()
    hilo.join(timeout=2)
    assert not hilo.is_alive()


def test_cargar_modelo_si_mlflow_devuelve_none_no_queda_listo(monkeypatch):
    monkeypatch.setattr(servicio, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda n, a: SimpleNamespace(version="1", run_id="r")))
    monkeypatch.setattr(servicio.mlflow.pyfunc, "load_model", lambda uri: None)
    assert servicio.cargar_modelo() is False
    assert servicio.estado["modelo"] is None and "no devolvió" in servicio.estado["error"]
