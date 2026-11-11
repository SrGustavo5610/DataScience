"""Usa el modelo entrenado para puntuar un CSV nuevo de hogares."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["AREA_DESC", "DPTO_NOMBRE", "TOTAL", "EDAD_JEFE", "SEXO_JEFE_DESC",
            "ANIOS_ESTUDIO_JEFE", "CAT_OCUP_JEFE_DESC", "IPCM"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="CSV con las 8 variables predictoras")
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "output" / "modelo_acceso_internet.joblib")
    p.add_argument("--output", type=Path, default=Path("predicciones_nuevos_hogares.csv"))
    args = p.parse_args()
    df = pd.read_csv(args.input)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    model = joblib.load(args.model)
    X = df[FEATURES].copy()
    df["PROBABILIDAD_ESTIMADA_INTERNET"] = model.predict_proba(X)[:, 1]
    df["CLASIFICACION_UMBRAL_0_5"] = model.predict(X)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en: {args.output.resolve()}")

if __name__ == "__main__":
    main()
