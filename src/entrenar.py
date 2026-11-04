"""Fase 1 — Entrena varios modelos de riesgo crediticio y registra cada ejecución en MLflow.

Cada combinación de algoritmo + hiperparámetros es un *run* del experimento "riesgo-crediticio".
Se registran parámetros, métricas, etiquetas, la firma del modelo y el modelo empaquetado (MLflow Model).
Métrica principal: ROC AUC (el dataset está desbalanceado: 22 % de morosos, así que accuracy engaña).
USO: MLFLOW_TRACKING_URI=http://mlflow:5000 python src/entrenar.py
"""
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import datos

EXPERIMENTO = "riesgo-crediticio"

CONFIGURACIONES = [
    ("regresion-logistica", "logreg_C-0.01", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000, random_state=datos.SEMILLA)), {"C": 0.01}),
    ("regresion-logistica", "logreg_C-1.0", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=datos.SEMILLA)), {"C": 1.0}),
    ("random-forest", "rf_n-100_depth-6", lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=datos.SEMILLA, n_jobs=-1), {"n_estimators": 100, "max_depth": 6}),
    ("random-forest", "rf_n-300_depth-10", lambda: RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=datos.SEMILLA, n_jobs=-1), {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5}),
    ("gradient-boosting", "gb_lr-0.1_depth-3", lambda: GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, random_state=datos.SEMILLA), {"learning_rate": 0.1, "n_estimators": 150, "max_depth": 3}),
    ("gradient-boosting", "gb_lr-0.05_depth-4", lambda: GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=4, subsample=0.8, random_state=datos.SEMILLA), {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 4, "subsample": 0.8}),
]


def main():
    np.random.seed(datos.SEMILLA)
    X_train, X_test, y_train, y_test = datos.cargar()
    mlflow.set_experiment(EXPERIMENTO)
    md5 = datos.md5_datos()
    for algoritmo, nombre, crear, params in CONFIGURACIONES:
        with mlflow.start_run(run_name=nombre) as run:
            modelo = crear().fit(X_train, y_train)
            proba = modelo.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metricas = {
                "roc_auc": roc_auc_score(y_test, proba),
                "f1": f1_score(y_test, pred),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
            }
            mlflow.log_params({**params, "semilla": datos.SEMILLA, "test_size": datos.TEST_SIZE})
            mlflow.log_metrics(metricas)
            mlflow.set_tags({"algoritmo": algoritmo, "dataset": "uci-default-credit-card", "datos_md5": md5,
                             "filas_train": len(X_train), "filas_test": len(X_test),
                             "git_commit": os.environ.get("GIT_COMMIT", "desconocido")})
            firma = infer_signature(X_train, modelo.predict_proba(X_train)[:, 1])
            mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma,
                                     input_example=X_train.head(3), pyfunc_predict_fn="predict_proba",
                                     # MLflow 3.14 serializa con skops: se declara el único tipo propio de los árboles (modelo creado aquí mismo)
                                     skops_trusted_types=["sklearn.tree._tree.Tree"] if algoritmo != "regresion-logistica" else None)
            print(f"{nombre:22s} run={run.info.run_id[:8]}  AUC={metricas['roc_auc']:.4f}  F1={metricas['f1']:.4f}  acc={metricas['accuracy']:.4f}")


if __name__ == "__main__":
    main()
