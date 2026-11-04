"""Fase 4.1 — Detección de data drift: compara cada variable de un lote nuevo contra el baseline de entrenamiento.

Pruebas por tipo de variable (clase 7, lámina 9):
- Numéricas continuas (LIMIT_BAL, AGE, BILL_AMT*, PAY_AMT*): Kolmogorov-Smirnov de dos muestras. Compara las
  distribuciones acumuladas completas sin suponer normalidad (los montos son muy asimétricos). Con 14 pruebas
  simultáneas se aplica la corrección de Bonferroni: alerta si p < 0,05 / 14 = 0,0036.
  Además se informa la magnitud del cambio (estadístico D) para no alarmar por diferencias ínfimas.
- Categóricas (SEX, EDUCATION, MARRIAGE, PAY_1..PAY_6): índice de estabilidad poblacional (PSI) por categoría.
  Umbrales de la industria (clase 7, láminas 9 y 11): PSI < 0,10 estable; 0,10–0,20 alerta menor; >= 0,20 drift.
Estado del lote: ROJO si al menos una variable está en rojo; AMARILLO si solo hay alertas menores; VERDE si no.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import datos

ALFA = 0.05
ALFA_BONF = ALFA / len(datos.NUMERICAS)
PSI_ALERTA, PSI_DRIFT = 0.10, 0.20
EPS = 1e-4


def psi(ref, act):
    cats = sorted(set(ref.unique()) | set(act.unique()))
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    q = act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def evaluar(referencia: pd.DataFrame, lote: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for v in datos.NUMERICAS:
        r = ks_2samp(referencia[v], lote[v])
        estado = "ROJO" if r.pvalue < ALFA_BONF else "VERDE"
        filas.append({"variable": v, "tipo": "numérica", "prueba": "KS", "estadistico": round(r.statistic, 4),
                      "p_valor": float(f"{r.pvalue:.3g}"), "umbral": f"p < {ALFA_BONF:.4f}", "estado": estado})
    for v in datos.CATEGORICAS:
        valor = psi(referencia[v], lote[v])
        estado = "ROJO" if valor >= PSI_DRIFT else "AMARILLO" if valor >= PSI_ALERTA else "VERDE"
        filas.append({"variable": v, "tipo": "categórica", "prueba": "PSI", "estadistico": round(valor, 4),
                      "p_valor": None, "umbral": f"PSI >= {PSI_DRIFT}", "estado": estado})
    return pd.DataFrame(filas)


def estado_global(tabla):
    return "ROJO" if (tabla.estado == "ROJO").any() else "AMARILLO" if (tabla.estado == "AMARILLO").any() else "VERDE"


def lote_derivado(lote: pd.DataFrame, semilla=7) -> pd.DataFrame:
    """Inyecta drift deliberado, simulando que el banco empieza a captar otro segmento de clientes."""
    rng = np.random.default_rng(semilla)
    d = lote.copy()
    d["LIMIT_BAL"] = (d["LIMIT_BAL"] * 0.6).round(-4).clip(lower=10000)        # límites más bajos (clientes nuevos)
    d["AGE"] = (d["AGE"] - 6).clip(lower=21)                                   # clientes más jóvenes
    d["BILL_AMT1"] = d["BILL_AMT1"] * 1.4                                     # facturas más altas (inflación)
    atraso = rng.random(len(d)) < 0.25                                        # 25 % pasa a tener 2 meses de atraso
    d.loc[atraso, "PAY_1"] = 2
    d.loc[rng.random(len(d)) < 0.30, "EDUCATION"] = 3                         # más clientes con secundaria
    return d


def main():
    X_train, X_test, _, _ = datos.cargar()
    salida = Path("drift/salidas"); salida.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for nombre, lote in [("mismo_origen", X_test), ("derivado", lote_derivado(X_test))]:
        tabla = evaluar(X_train, lote)
        tabla.to_csv(salida / f"data_drift_{nombre}.csv", index=False)
        resumen[nombre] = {"estado": estado_global(tabla), "variables_en_rojo": tabla[tabla.estado == "ROJO"].variable.tolist(),
                           "variables_en_amarillo": tabla[tabla.estado == "AMARILLO"].variable.tolist()}
        print(f"\n=== Lote '{nombre}' ({len(lote)} filas) vs baseline de entrenamiento ({len(X_train)} filas) ===")
        print(tabla.to_string(index=False))
        print(f"ESTADO DEL LOTE: {resumen[nombre]['estado']}  · en rojo: {resumen[nombre]['variables_en_rojo']}")
    (salida / "data_drift_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
