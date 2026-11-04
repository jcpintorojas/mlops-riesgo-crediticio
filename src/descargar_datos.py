"""Descarga el dataset público "Default of Credit Card Clients" (UCI, id 350) y lo deja en data/credito.csv.

Fuente: Yeh, I. C. (2009). UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
30.000 clientes de tarjetas de crédito de Taiwán (abr–sep 2005), 23 variables de entrada y la etiqueta
"default payment next month" (1 = incumplió el pago del mes siguiente).
USO: python src/descargar_datos.py
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DESTINO = Path("data/credito.csv")


def main():
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    contenido = urllib.request.urlopen(URL, timeout=120).read()
    print(f"Descargado: {len(contenido):,} bytes  sha256={hashlib.sha256(contenido).hexdigest()[:16]}…")
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        nombre = next(n for n in z.namelist() if n.endswith(".xls"))
        df = pd.read_excel(io.BytesIO(z.read(nombre)), header=1)
    df = df.drop(columns=["ID"]).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
    df.to_csv(DESTINO, index=False)
    md5 = hashlib.md5(DESTINO.read_bytes()).hexdigest()
    print(f"{DESTINO}: {df.shape[0]:,} filas x {df.shape[1]} columnas · tasa de morosidad {df['default'].mean():.4f} · md5 {md5}")


if __name__ == "__main__":
    main()
