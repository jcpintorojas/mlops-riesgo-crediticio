"""Prueba automatizada de data drift (pytest): verde con datos del mismo origen, rojo con datos derivados.
USO: bash scripts/entrenar.sh -m pytest -v drift/test_drift.py
"""
import datos
from data_drift import estado_global, evaluar, lote_derivado

X_TRAIN, X_TEST, _, _ = datos.cargar()


def test_mismo_origen_pasa_en_verde():
    tabla = evaluar(X_TRAIN, X_TEST)
    assert estado_global(tabla) == "VERDE", tabla[tabla.estado != "VERDE"]


def test_lote_derivado_falla_en_rojo():
    tabla = evaluar(X_TRAIN, lote_derivado(X_TEST))
    rojas = set(tabla[tabla.estado == "ROJO"].variable)
    assert estado_global(tabla) == "ROJO"
    assert {"LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_1", "EDUCATION"} <= rojas, rojas


def test_variables_no_tocadas_siguen_estables():
    tabla = evaluar(X_TRAIN, lote_derivado(X_TEST)).set_index("variable")
    assert tabla.loc["PAY_AMT1", "estado"] == "VERDE" and tabla.loc["SEX", "estado"] == "VERDE"
