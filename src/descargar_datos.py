"""Descarga el dataset público "Default of Credit Card Clients" (UCI, id 350) y lo deja en data/credito.csv.

Fuente: Yeh, I. C. (2009). UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
30.000 clientes de tarjetas de crédito de Taiwán (abr–sep 2005), 23 variables de entrada y la etiqueta
"default payment next month" (1 = incumplió el pago del mes siguiente).
USO: python src/descargar_datos.py
"""
# hashlib calcula las huellas de los archivos.
import hashlib
# io permite leer el zip desde la memoria, sin guardarlo en disco.
import io
# urllib descarga el archivo.
import urllib.request
# zipfile abre el zip descargado.
import zipfile
# Path maneja rutas de archivos.
from pathlib import Path

# pandas lee el Excel y escribe el CSV.
import pandas as pd

# Dirección oficial del dataset en UCI.
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
# Archivo generado; está excluido de Git.
DESTINO = Path("data/credito.csv")


def main():
    # Crea la carpeta data/ si no existe.
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    # Descarga el zip a memoria.
    contenido = urllib.request.urlopen(URL, timeout=120).read()
    # Muestra el tamaño y la huella para comprobar la descarga.
    print(f"Descargado: {len(contenido):,} bytes  sha256={hashlib.sha256(contenido).hexdigest()[:16]}…")
    # Abre el zip sin extraerlo a disco.
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        # Busca el archivo Excel dentro del zip.
        nombre = next(n for n in z.namelist() if n.endswith(".xls"))
        # header=1: los nombres de columna están en la segunda fila.
        df = pd.read_excel(io.BytesIO(z.read(nombre)), header=1)
    # Quita ID porque no predice nada; PAY_0 pasa a PAY_1 y la etiqueta a "default".
    df = df.drop(columns=["ID"]).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
    # Guarda el resultado en CSV.
    df.to_csv(DESTINO, index=False)
    # Huella del CSV: identifica esta versión de los datos.
    md5 = hashlib.md5(DESTINO.read_bytes()).hexdigest()
    print(f"{DESTINO}: {df.shape[0]:,} filas x {df.shape[1]} columnas · tasa de morosidad {df['default'].mean():.4f} · md5 {md5}")


if __name__ == "__main__":
    main()
