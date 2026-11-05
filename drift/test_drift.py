"""Prueba automatizada de data drift (pytest): verde con datos del mismo origen, rojo con datos derivados.
USO: bash scripts/entrenar.sh -m pytest -v drift/test_drift.py
"""
# Carga la división de datos.
import datos
# Importa las funciones a probar.
from data_drift import estado_global, evaluar, lote_derivado

# X_TRAIN es el baseline; X_TEST representa datos del mismo origen.
X_TRAIN, X_TEST, _, _ = datos.cargar()


# Sin drift inyectado, no debe haber alarma.
def test_mismo_origen_pasa_en_verde():
    tabla = evaluar(X_TRAIN, X_TEST)
    assert estado_global(tabla) == "VERDE", tabla[tabla.estado != "VERDE"]


# Con drift inyectado, debe haber alarma.
def test_lote_derivado_falla_en_rojo():
    tabla = evaluar(X_TRAIN, lote_derivado(X_TEST))
    rojas = set(tabla[tabla.estado == "ROJO"].variable)
    assert estado_global(tabla) == "ROJO"
    # Exige detectar las 5 variables alteradas.
    assert {"LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_1", "EDUCATION"} <= rojas, rojas


# Las variables no alteradas no deben dar alarma.
def test_variables_no_tocadas_siguen_estables():
    tabla = evaluar(X_TRAIN, lote_derivado(X_TEST)).set_index("variable")
    assert tabla.loc["PAY_AMT1", "estado"] == "VERDE" and tabla.loc["SEX", "estado"] == "VERDE"
