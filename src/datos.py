"""Carga del dataset y división train/test reproducible (compartida por entrenamiento y pruebas de drift)."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RUTA = Path("data/credito.csv")
SEMILLA = 42
TEST_SIZE = 0.20  # 80 % entrenamiento (24.000) / 20 % prueba (6.000), estratificado por la etiqueta
OBJETIVO = "default"

# Tipos de variable (determinan la prueba de drift que se aplica a cada una)
CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERICAS = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
VARIABLES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] \
    + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


def md5_datos():
    return hashlib.md5(RUTA.read_bytes()).hexdigest()


def cargar():
    df = pd.read_csv(RUTA)
    X, y = df[VARIABLES], df[OBJETIVO]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)
