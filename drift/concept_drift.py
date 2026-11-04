"""Fase 4.2 — Simulación de concept drift y criterio de reentrenamiento.

Escenario: llega una crisis de endeudamiento. Desde el lote 5, una parte creciente de los clientes que usan
más del 70 % de su límite de crédito (BILL_AMT1 / LIMIT_BAL > 0,7) pasa a incumplir aunque venga pagando al día,
y los que usan poco dejan de incumplir. Las ENTRADAS no cambian (P(X) igual): cambia la relación
entrada -> salida, P(Y|X). Por eso el monitor de data drift sigue en verde y solo la métrica lo revela.

Se usa el modelo en producción, cargado del registro: models:/riesgo-credito@champion.
Criterio de reentrenamiento (clase 7, lámina 11, disparador por rendimiento):
  umbral = AUC de referencia - 2 desviaciones estándar del AUC en lotes sanos de 500 (variación normal por azar);
  alarma de reentrenamiento si el AUC queda bajo el umbral en 3 lotes CONSECUTIVOS (evita reaccionar a un lote raro).
Cada lote se registra en MLflow (experimento "monitoreo-concept-drift", métrica por step = lote).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

import datos
from data_drift import estado_global, evaluar

N_LOTES, TAM, INICIO_DRIFT, CONSECUTIVOS = 12, 500, 5, 3
INTENSIDAD = {5: 0.15, 6: 0.30, 7: 0.45, 8: 0.60, 9: 0.70, 10: 0.80, 11: 0.85, 12: 0.90}  # fracción de clientes afectados


def main():
    rng = np.random.default_rng(datos.SEMILLA)
    X_train, X_test, _, y_test = datos.cargar()
    modelo = mlflow.pyfunc.load_model("models:/riesgo-credito@champion")
    version = mlflow.MlflowClient().get_model_version_by_alias("riesgo-credito", "champion")
    proba = lambda X: np.asarray(modelo.predict(X))[:, 1]

    auc_ref = roc_auc_score(y_test, proba(X_test))
    boot = [roc_auc_score(y_test.iloc[i], proba(X_test.iloc[i]))
            for i in (rng.choice(len(X_test), TAM, replace=False) for _ in range(200))]
    sigma = float(np.std(boot)); umbral = auc_ref - 2 * sigma

    orden = rng.permutation(len(X_test))
    filas, seguidos, alarma = [], 0, None
    mlflow.set_experiment("monitoreo-concept-drift")
    with mlflow.start_run(run_name=f"simulacion-crisis-v{version.version}"):
        mlflow.log_params({"modelo": f"riesgo-credito v{version.version}", "run_modelo": version.run_id,
                           "auc_referencia": round(auc_ref, 4), "sigma_auc_lote": round(sigma, 4),
                           "umbral_auc": round(umbral, 4), "lotes_consecutivos": CONSECUTIVOS, "tam_lote": TAM})
        for k in range(1, N_LOTES + 1):
            idx = orden[(k - 1) * TAM:k * TAM]
            X, y = X_test.iloc[idx], y_test.iloc[idx].to_numpy().copy()
            f = INTENSIDAD.get(k, 0.0)
            if f:
                uso = (X["BILL_AMT1"] / X["LIMIT_BAL"]).to_numpy()
                nueva = (uso > 0.7).astype(int)                       # la nueva regla que genera la etiqueta
                afectado = rng.random(len(y)) < f
                y[afectado] = nueva[afectado]
            p = proba(X)
            auc = roc_auc_score(y, p)
            seguidos = seguidos + 1 if auc < umbral else 0
            if seguidos >= CONSECUTIVOS and alarma is None:
                alarma = k
            dd = estado_global(evaluar(X_train, X))
            filas.append({"lote": k, "fraccion_afectada": f, "auc": round(auc, 4), "bajo_umbral": auc < umbral,
                          "lotes_seguidos": seguidos, "tasa_real_morosos": round(float(y.mean()), 3),
                          "prob_media_predicha": round(float(p.mean()), 3), "data_drift": dd})
            mlflow.log_metrics({"auc_lote": auc, "tasa_real_morosos": y.mean(), "prob_media_predicha": p.mean()}, step=k)
            print(f"lote {k:2d}  afectados {f:4.0%}  AUC {auc:.4f}  {'BAJO UMBRAL' if auc < umbral else 'ok':11s}"
                  f"  seguidos {seguidos}  morosos reales {y.mean():.3f}  prob. media {p.mean():.3f}  data drift {dd}")
        mlflow.log_metric("lote_alarma", alarma or 0)

        sal = Path("drift/salidas"); sal.mkdir(parents=True, exist_ok=True)
        lotes = [r["lote"] for r in filas]; aucs = [r["auc"] for r in filas]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axvspan(INICIO_DRIFT - 0.5, N_LOTES + 0.5, color="#f4cccc", alpha=0.35, label="Concept drift activo")
        ax.plot(lotes, aucs, "o-", color="#1f3864", lw=2, label="ROC AUC por lote (modelo @champion)")
        ax.axhline(auc_ref, color="#6aa84f", ls="--", label=f"AUC de referencia = {auc_ref:.3f}")
        ax.axhline(umbral, color="#cc0000", ls="--", label=f"Umbral = referencia − 2σ = {umbral:.3f}")
        if alarma:
            ax.axvline(alarma, color="#cc0000", lw=2)
            ax.annotate(f"ALARMA: reentrenar\n({CONSECUTIVOS} lotes seguidos bajo el umbral)", (alarma, umbral),
                        xytext=(alarma + 0.4, auc_ref + 0.02), arrowprops=dict(arrowstyle="->", color="#cc0000"), color="#cc0000")
        ax.set(xlabel="Lote de producción (500 clientes cada uno, en orden temporal)", ylabel="ROC AUC",
               title="Degradación del modelo por concept drift", xticks=lotes)
        ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout(); fig.savefig(sal / "concept_drift.png", dpi=150)
        mlflow.log_artifact(str(sal / "concept_drift.png"))
        (sal / "concept_drift.json").write_text(json.dumps(
            {"auc_referencia": auc_ref, "sigma": sigma, "umbral": umbral, "lote_alarma": alarma, "lotes": filas},
            indent=2, ensure_ascii=False, default=bool))
        print(f"\nAUC referencia {auc_ref:.4f} · σ por lote {sigma:.4f} · umbral {umbral:.4f} · ALARMA DE REENTRENAMIENTO en el lote {alarma}")


if __name__ == "__main__":
    main()
