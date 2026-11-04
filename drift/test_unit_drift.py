"""Pruebas unitarias de las funciones de drift con datos sintéticos: no requieren descargar el dataset de UCI,
así que corren en CI y en cualquier máquina.
USO:  PYTHONPATH=src:drift python -m pytest -v drift/test_unit_drift.py
"""
import numpy as np
import pandas as pd
import pytest

import datos
from data_drift import ALFA_BONF, PSI_ALERTA, PSI_DRIFT, estado_global, evaluar, lote_derivado, psi


def tabla_sintetica(n, semilla):
    """DataFrame con las 23 variables del proyecto y distribuciones simples pero realistas."""
    rng = np.random.default_rng(semilla)
    d = {}
    for v in datos.NUMERICAS:
        d[v] = rng.lognormal(10, 1, n).round() if v != "AGE" else rng.integers(21, 70, n)
    d["SEX"] = rng.choice([1, 2], n, p=[0.4, 0.6])
    d["EDUCATION"] = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02])
    d["MARRIAGE"] = rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02])
    for i in range(1, 7):
        d[f"PAY_{i}"] = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[0.1, 0.2, 0.5, 0.08, 0.1, 0.02])
    return pd.DataFrame(d)[datos.VARIABLES]


def test_psi_de_distribuciones_identicas_es_casi_cero():
    s = pd.Series(np.random.default_rng(1).choice([0, 1, 2], 5000, p=[0.5, 0.3, 0.2]))
    assert psi(s, s) == pytest.approx(0, abs=1e-6)


def test_psi_crece_con_el_cambio_de_proporciones():
    ref = pd.Series([0] * 500 + [1] * 500)
    leve = pd.Series([0] * 550 + [1] * 450)
    fuerte = pd.Series([0] * 800 + [1] * 200)
    assert psi(ref, leve) < PSI_ALERTA <= PSI_DRIFT < psi(ref, fuerte)


def test_psi_tolera_categorias_nuevas_o_ausentes():
    ref, act = pd.Series([1, 1, 2, 2]), pd.Series([1, 1, 3, 3])
    assert np.isfinite(psi(ref, act)) and psi(ref, act) > PSI_DRIFT


def test_umbral_de_bonferroni_es_alfa_entre_numero_de_pruebas():
    assert ALFA_BONF == pytest.approx(0.05 / len(datos.NUMERICAS)) and len(datos.NUMERICAS) == 14


def test_lote_del_mismo_origen_sale_verde():
    tabla = evaluar(tabla_sintetica(20000, 1), tabla_sintetica(5000, 2))
    assert estado_global(tabla) == "VERDE", tabla[tabla.estado != "VERDE"]
    assert len(tabla) == 23  # una prueba por cada variable de entrada


def test_lote_derivado_sale_rojo_en_las_variables_alteradas():
    base, lote = tabla_sintetica(20000, 1), tabla_sintetica(5000, 2)
    tabla = evaluar(base, lote_derivado(lote))
    rojas = set(tabla[tabla.estado == "ROJO"].variable)
    assert estado_global(tabla) == "ROJO"
    assert {"LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_1", "EDUCATION"} <= rojas, rojas


def test_variables_no_tocadas_no_se_marcan():
    base, lote = tabla_sintetica(20000, 1), tabla_sintetica(5000, 2)
    tabla = evaluar(base, lote_derivado(lote)).set_index("variable")
    assert tabla.loc["PAY_AMT1", "estado"] == "VERDE" and tabla.loc["SEX", "estado"] == "VERDE"


def test_se_usa_la_prueba_correcta_por_tipo_de_variable():
    tabla = evaluar(tabla_sintetica(5000, 1), tabla_sintetica(2000, 2)).set_index("variable")
    assert (tabla.loc[datos.NUMERICAS, "prueba"] == "KS").all()
    assert (tabla.loc[datos.CATEGORICAS, "prueba"] == "PSI").all()


def test_estado_global_prioriza_rojo_sobre_amarillo():
    t = pd.DataFrame({"estado": ["VERDE", "AMARILLO", "ROJO"]})
    assert estado_global(t) == "ROJO"
    assert estado_global(t.iloc[:2]) == "AMARILLO" and estado_global(t.iloc[:1]) == "VERDE"
