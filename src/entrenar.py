"""Fase 1 — Entrena varios modelos de riesgo crediticio y registra cada ejecución en MLflow.

Cada combinación de algoritmo + hiperparámetros es un *run* del experimento "riesgo-crediticio".
Se registran parámetros, métricas, etiquetas, la firma del modelo y el modelo empaquetado (MLflow Model).
Métrica principal: ROC AUC (el dataset está desbalanceado: 22 % de morosos, así que accuracy engaña).
USO: MLFLOW_TRACKING_URI=http://mlflow:5000 python src/entrenar.py
"""
# os lee el commit de Git desde una variable de entorno.
import os

# mlflow registra parámetros, métricas y modelos.
import mlflow
import mlflow.sklearn
import numpy as np
# infer_signature describe las entradas y salidas del modelo.
from mlflow.models import infer_signature
# Los dos algoritmos basados en árboles.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# El modelo lineal, usado como base.
from sklearn.linear_model import LogisticRegression
# Métricas para comparar los runs.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# make_pipeline encadena el escalado y el modelo.
from sklearn.pipeline import make_pipeline
# StandardScaler pone las variables en una escala comparable.
from sklearn.preprocessing import StandardScaler

# Nuestro módulo: misma división de datos que usan las pruebas de drift.
import datos

# Agrupa todos los runs de este objetivo en MLflow.
EXPERIMENTO = "riesgo-crediticio"

# Cada fila es un run: algoritmo, nombre, función que crea el modelo e hiperparámetros.
CONFIGURACIONES = [
    # Regresión logística: C pequeño significa más regularización. Necesita escalado.
    ("regresion-logistica", "logreg_C-0.01", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000, random_state=datos.SEMILLA)), {"C": 0.01}),
    ("regresion-logistica", "logreg_C-1.0", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=datos.SEMILLA)), {"C": 1.0}),
    # Random forest: muchos árboles que votan. No necesita escalado.
    ("random-forest", "rf_n-100_depth-6", lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=datos.SEMILLA, n_jobs=-1), {"n_estimators": 100, "max_depth": 6}),
    ("random-forest", "rf_n-300_depth-10", lambda: RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=datos.SEMILLA, n_jobs=-1), {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5}),
    # Gradient boosting: cada árbol corrige los errores del anterior. Este run ganó.
    ("gradient-boosting", "gb_lr-0.1_depth-3", lambda: GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, random_state=datos.SEMILLA), {"learning_rate": 0.1, "n_estimators": 150, "max_depth": 3}),
    ("gradient-boosting", "gb_lr-0.05_depth-4", lambda: GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=4, subsample=0.8, random_state=datos.SEMILLA), {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 4, "subsample": 0.8}),
]


def main():
    # Fija el azar de numpy para repetir resultados.
    np.random.seed(datos.SEMILLA)
    # Carga la división 80/20 estratificada.
    X_train, X_test, y_train, y_test = datos.cargar()
    # Selecciona el experimento; lo crea si no existe.
    mlflow.set_experiment(EXPERIMENTO)
    # Huella de los datos que se guardará en cada run.
    md5 = datos.md5_datos()
    # Un ciclo por configuración: un run por vuelta.
    for algoritmo, nombre, crear, params in CONFIGURACIONES:
        # Abre el run: todo lo registrado adentro queda asociado a él.
        with mlflow.start_run(run_name=nombre) as run:
            # Entrena solo con los datos de entrenamiento.
            modelo = crear().fit(X_train, y_train)
            # Probabilidad de incumplir en el conjunto de prueba.
            proba = modelo.predict_proba(X_test)[:, 1]
            # Convierte la probabilidad en clase con el umbral 0,5.
            pred = (proba >= 0.5).astype(int)
            # Se miden en datos que el modelo no vio.
            metricas = {
                # Métrica principal: qué tan bien ordena a los clientes por riesgo.
                "roc_auc": roc_auc_score(y_test, proba),
                # Equilibrio entre precision y recall.
                "f1": f1_score(y_test, pred),
                # Se registra para comparar, pero engaña con clases desbalanceadas.
                "accuracy": accuracy_score(y_test, pred),
                # De los marcados como morosos, cuántos lo eran.
                "precision": precision_score(y_test, pred),
                # De los morosos reales, cuántos detectó.
                "recall": recall_score(y_test, pred),
            }
            # Guarda la configuración del run.
            mlflow.log_params({**params, "semilla": datos.SEMILLA, "test_size": datos.TEST_SIZE})
            # Guarda los resultados del run.
            mlflow.log_metrics(metricas)
            # Etiquetas para filtrar y para reproducir: datos, tamaños y commit.
            mlflow.set_tags({"algoritmo": algoritmo, "dataset": "uci-default-credit-card", "datos_md5": md5,
                             "filas_train": len(X_train), "filas_test": len(X_test),
                             "git_commit": os.environ.get("GIT_COMMIT", "desconocido")})
            # Firma: documenta el formato de entrada y salida del modelo.
            firma = infer_signature(X_train, modelo.predict_proba(X_train)[:, 1])
            # Empaqueta el modelo con su entorno (MLflow Model). pyfunc usará predict_proba.
            mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma,
                                     input_example=X_train.head(3), pyfunc_predict_fn="predict_proba",
                                     # MLflow 3.14 serializa con skops: se declara el único tipo propio de los árboles (modelo creado aquí mismo)
                                     skops_trusted_types=["sklearn.tree._tree.Tree"] if algoritmo != "regresion-logistica" else None)
            # Resumen del run en la terminal.
            print(f"{nombre:22s} run={run.info.run_id[:8]}  AUC={metricas['roc_auc']:.4f}  F1={metricas['f1']:.4f}  acc={metricas['accuracy']:.4f}")


if __name__ == "__main__":
    main()
