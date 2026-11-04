"""Fase 1 — Registra modelos en el Model Registry de MLflow y marca la versión que se despliega.

- Versión 1: el mejor run de regresión logística (modelo base, simple e interpretable).
- Versión 2: el mejor run de todo el experimento según ROC AUC.
- Alias "champion" -> versión 2. El servicio en Kubernetes carga models:/riesgo-credito@champion,
  nunca una ruta de archivo. Mover el alias a otra versión cambia el modelo servido sin tocar código.
USO: bash scripts/entrenar.sh src/registrar.py
"""
import mlflow
from mlflow import MlflowClient

NOMBRE = "riesgo-credito"
ALIAS = "champion"


def mejor_run(cliente, exp_id, filtro=""):
    return cliente.search_runs([exp_id], filter_string=filtro, order_by=["metrics.roc_auc DESC"], max_results=1)[0]


def main():
    cliente = MlflowClient()
    exp = cliente.get_experiment_by_name("riesgo-crediticio")
    base = mejor_run(cliente, exp.experiment_id, "tags.algoritmo = 'regresion-logistica'")
    mejor = mejor_run(cliente, exp.experiment_id)
    for run, descripcion in [(base, "Modelo base: regresión logística"), (mejor, "Mejor ROC AUC del experimento")]:
        v = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", NOMBRE)
        cliente.update_model_version(NOMBRE, v.version, f"{descripcion} — run {run.data.tags['mlflow.runName']} "
                                     f"(AUC {run.data.metrics['roc_auc']:.4f})")
        print(f"v{v.version} <- {run.data.tags['mlflow.runName']} run_id={run.info.run_id} AUC={run.data.metrics['roc_auc']:.4f}")
        ultima = v.version
    cliente.set_registered_model_alias(NOMBRE, ALIAS, ultima)
    cliente.update_registered_model(NOMBRE, "Clasificador de riesgo de incumplimiento (UCI Default of Credit Card Clients).")
    print(f"alias @{ALIAS} -> versión {cliente.get_model_version_by_alias(NOMBRE, ALIAS).version}")


if __name__ == "__main__":
    main()
