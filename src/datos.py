"""Carga del dataset y división train/test reproducible (compartida por entrenamiento y pruebas de drift)."""
# hashlib calcula el md5, la huella que identifica el archivo de datos.
import hashlib
# Path maneja rutas de archivos.
from pathlib import Path

# pandas lee el CSV.
import pandas as pd
# train_test_split divide los datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split

# Archivo creado por descargar_datos.py; no se versiona en Git.
RUTA = Path("data/credito.csv")
# Semilla fija: la misma división en cada ejecución.
SEMILLA = 42
# Proporción de datos reservada para la prueba.
TEST_SIZE = 0.20  # 80 % entrenamiento (24.000) / 20 % prueba (6.000), estratificado por la etiqueta
# Variable a predecir: 1 incumplió, 0 pagó.
OBJETIVO = "default"

# Tipos de variable (determinan la prueba de drift que se aplica a cada una)
# Se evalúan con PSI en las pruebas de drift.
CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Se evalúan con Kolmogorov-Smirnov en las pruebas de drift.
NUMERICAS = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
# Las 23 entradas en orden fijo; el modelo las espera así.
VARIABLES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] \
    + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


# Se guarda en cada run para saber con qué datos se entrenó.
def md5_datos():
    return hashlib.md5(RUTA.read_bytes()).hexdigest()


# Fuente única de datos: entrenamiento y drift obtienen la misma división.
def cargar():
    # Lee el dataset completo.
    df = pd.read_csv(RUTA)
    # Separa las entradas (X) de la etiqueta (y).
    X, y = df[VARIABLES], df[OBJETIVO]
    # stratify=y mantiene el 22 % de morosos en ambos conjuntos.
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)
