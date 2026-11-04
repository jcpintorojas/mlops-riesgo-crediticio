"""Mueve el alias @champion de riesgo-credito a la versión indicada (rollback/promoción del modelo).
USO: bash scripts/entrenar.sh src/alias.py <version>
"""
import sys

from mlflow import MlflowClient

cliente = MlflowClient()
cliente.set_registered_model_alias("riesgo-credito", "champion", sys.argv[1])
v = cliente.get_model_version_by_alias("riesgo-credito", "champion")
print(f"@champion -> versión {v.version} (run {v.run_id})")
