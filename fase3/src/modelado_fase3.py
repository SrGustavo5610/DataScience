"""Entrenamiento reproducible para la Fase 3: acceso a internet en hogares paraguayos.

Unidad de análisis: hogar. El modelo estima TIENE_INTERNET a partir de variables
socioeconómicas y demográficas disponibles antes de observar el resultado.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

RANDOM_STATE = 42
TARGET = "TIENE_INTERNET"
FEATURES = ["AREA_DESC", "DPTO_NOMBRE", "TOTAL", "EDAD_JEFE",
            "SEXO_JEFE_DESC", "ANIOS_ESTUDIO_JEFE", "CAT_OCUP_JEFE_DESC", "IPCM"]
CATEGORICAL = ["AREA_DESC", "DPTO_NOMBRE", "SEXO_JEFE_DESC", "CAT_OCUP_JEFE_DESC"]
NUMERIC = ["TOTAL", "EDAD_JEFE", "ANIOS_ESTUDIO_JEFE"]
INCOME = ["IPCM"]


def project_root() -> Path:
    """Localiza la carpeta fase3 aunque se ejecute desde src/ o notebooks/."""
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent, *here.parents]:
        if (candidate / "data" / "dataset_limpio_internet_hogares.csv").exists():
            return candidate
    raise FileNotFoundError("No se encontró fase3/data/dataset_limpio_internet_hogares.csv")


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, low_memory=False)
    required = FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
    # La variable objetivo debe ser conocida; los predictores faltantes se imputan
    # dentro de cada pipeline para evitar filtración entre particiones.
    df = df.loc[df[TARGET].notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.isin([0, 1])
    X = df.loc[valid, FEATURES].copy()
    y = y.loc[valid].astype(int)
    for col in NUMERIC + INCOME:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # Ingresos negativos no son interpretables para log1p: se convierten en faltantes.
    X.loc[X["IPCM"] < 0, "IPCM"] = np.nan
    for col in CATEGORICAL:
        # object + np.nan mantiene compatibilidad con SimpleImputer incluso si
        # el CSV de inferencia contiene categorías faltantes.
        X[col] = X[col].astype(object)
        X.loc[pd.isna(X[col]), col] = np.nan
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    income_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipe, CATEGORICAL),
        ("numeric", numeric_pipe, NUMERIC),
        ("income", income_pipe, INCOME),
    ], remainder="drop", verbose_feature_names_out=False)


def make_candidates() -> dict:
    return {
        "Regresión logística": (
            LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"model__max_depth": [None, 15], "model__min_samples_leaf": [2, 5]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
             "model__max_iter": [100]},
        ),
    }


def score_row(name: str, estimator, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    pred = estimator.predict(X)
    proba = estimator.predict_proba(X)[:, 1]
    return {"modelo": name, "particion": split,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "auc_roc": roc_auc_score(y, proba)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa modelos de acceso a internet en hogares.")
    parser.add_argument("--data", type=Path, default=None, help="CSV limpio de hogares")
    parser.add_argument("--output", type=Path, default=None, help="Carpeta para artefactos")
    parser.add_argument("--quick", action="store_true", help="Reduce CV a 2 folds para pruebas rápidas")
    args = parser.parse_args()
    root = project_root()
    data_path = (args.data or root / "data" / "dataset_limpio_internet_hogares.csv").resolve()
    out = (args.output or root / "output").resolve()
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    folds = 2 if args.quick else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    dummy = Pipeline([("preprocess", build_preprocessor()),
                      ("model", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    results_cv = []
    results_val = [score_row("Baseline mayoritaria", dummy, X_val, y_val, "validacion")]
    best_by_model = {}
    candidates = make_candidates()
    for name, (model, grid) in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=2,
                              refit=True, return_train_score=False, verbose=0)
        search.fit(X_train, y_train)
        best_by_model[name] = search.best_estimator_
        results_cv.append({"modelo": name, "f1_cv": search.best_score_,
                           "mejores_parametros": json.dumps(search.best_params_, ensure_ascii=False)})
        results_val.append(score_row(name, search.best_estimator_, X_val, y_val, "validacion"))
        print(f"{name}: F1 CV={search.best_score_:.4f}; validación={results_val[-1]['f1']:.4f}", flush=True)

    cv_df = pd.DataFrame(results_cv).sort_values("f1_cv", ascending=False)
    val_df = pd.DataFrame(results_val).sort_values(["f1", "auc_roc"], ascending=False)
    # Selección principal por F1 de validación; AUC-ROC desempata.
    winner_name = val_df.loc[val_df["modelo"] != "Baseline mayoritaria"].sort_values(
        ["f1", "auc_roc"], ascending=False).iloc[0]["modelo"]
    winner = best_by_model[winner_name]
    test_metrics = score_row(winner_name, winner, X_test, y_test, "prueba")
    baseline_test_metrics = score_row("Baseline mayoritaria", dummy, X_test, y_test, "prueba")
    pd.DataFrame([test_metrics, baseline_test_metrics]).to_csv(
        out / "comparacion_prueba.csv", index=False)

    # Métricas de entrenamiento/validación/prueba con el mismo modelo final.
    split_metrics = pd.DataFrame([
        score_row(winner_name, winner, X_train, y_train, "entrenamiento"),
        score_row(winner_name, winner, X_val, y_val, "validacion"),
        test_metrics,
    ])
    split_metrics.to_csv(out / "metricas_particiones.csv", index=False)
    cv_df.to_csv(out / "comparacion_validacion_cruzada.csv", index=False)
    val_df.to_csv(out / "comparacion_validacion.csv", index=False)

    y_pred = winner.predict(X_test)
    y_prob = winner.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Real_sin_internet", "Real_con_internet"],
                 columns=["Pred_sin_internet", "Pred_con_internet"]).to_csv(out / "matriz_confusion.csv")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(out / "reporte_clasificacion.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    test_detail = X_test.copy()
    test_detail.insert(0, "TIENE_INTERNET_REAL", y_test.values)
    test_detail["TIENE_INTERNET_PREDICHO"] = y_pred
    test_detail["PROBABILIDAD_INTERNET"] = y_prob
    test_detail["ERROR"] = test_detail["TIENE_INTERNET_REAL"] != test_detail["TIENE_INTERNET_PREDICHO"]
    test_detail.loc[test_detail["ERROR"]].to_csv(out / "predicciones_errores_prueba.csv", index=False)

    # Auditoría agregada de errores por clase real y área, sin exponer filas individuales.
    error_area = (test_detail.groupby(["TIENE_INTERNET_REAL", "AREA_DESC"], dropna=False)
                  .agg(casos=("ERROR", "size"), errores=("ERROR", "sum"))
                  .reset_index())
    error_area["tasa_error"] = error_area["errores"] / error_area["casos"]
    error_area.to_csv(out / "diagnostico_errores_por_area.csv", index=False)
    fp = int(((y_test.values == 0) & (y_pred == 1)).sum())
    fn = int(((y_test.values == 1) & (y_pred == 0)).sum())
    hard_fp = int(((y_test.values == 0) & (y_pred == 1) & (y_prob >= 0.90)).sum())
    hard_fn = int(((y_test.values == 1) & (y_pred == 0) & (y_prob <= 0.10)).sum())
    with open(out / "resumen_errores.json", "w", encoding="utf-8") as f:
        json.dump({"errores_totales": int(test_detail["ERROR"].sum()),
                   "falsos_positivos": fp, "falsos_negativos": fn,
                   "falsos_positivos_confianza_ge_0_90": hard_fp,
                   "falsos_negativos_confianza_le_0_10": hard_fn,
                   "nota": "Diagnóstico interno no ponderado del conjunto de prueba."},
                  f, ensure_ascii=False, indent=2)

    # Importancia por permutación en validación; mide cambio en F1, no causalidad.
    perm = permutation_importance(winner, X_val, y_val, scoring="f1", n_repeats=5,
                                  random_state=RANDOM_STATE, n_jobs=2)
    imp = pd.DataFrame({"variable": FEATURES, "importancia_media": perm.importances_mean,
                        "desvio_estandar": perm.importances_std}).sort_values(
                            "importancia_media", ascending=False)
    imp.to_csv(out / "importancia_permutacion.csv", index=False)

    # Figuras.
    plt.figure(figsize=(6.5, 5.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión - conjunto de prueba")
    plt.colorbar()
    plt.xticks([0, 1], ["Sin internet", "Con internet"], rotation=15)
    plt.yticks([0, 1], ["Sin internet", "Con internet"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("Clase real"); plt.xlabel("Clase predicha")
    plt.tight_layout(); plt.savefig(out / "matriz_confusion.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    RocCurveDisplay.from_estimator(winner, X_test, y_test, ax=ax)
    ax.set_title("Curva ROC - conjunto de prueba")
    fig.tight_layout(); fig.savefig(out / "curva_roc.png", dpi=160); plt.close(fig)

    plt.figure(figsize=(7.5, 5.2))
    show = imp.sort_values("importancia_media", ascending=True)
    plt.barh(show["variable"], show["importancia_media"], xerr=show["desvio_estandar"])
    plt.xlabel("Disminución media de F1 al permutar")
    plt.title("Importancia por permutación - validación")
    plt.tight_layout(); plt.savefig(out / "importancia_permutacion.png", dpi=160); plt.close()

    joblib.dump(winner, out / "modelo_acceso_internet.joblib")
    # Dos ejemplos ilustrativos, no registros reales identificables.
    examples = pd.DataFrame([
        {"AREA_DESC": "Urbana", "DPTO_NOMBRE": "Central", "TOTAL": 4,
         "EDAD_JEFE": 42, "SEXO_JEFE_DESC": "Mujer", "ANIOS_ESTUDIO_JEFE": 12,
         "CAT_OCUP_JEFE_DESC": "Empleado/a privado/a", "IPCM": 2500000},
        {"AREA_DESC": "Rural", "DPTO_NOMBRE": "San Pedro", "TOTAL": 5,
         "EDAD_JEFE": 55, "SEXO_JEFE_DESC": "Hombre", "ANIOS_ESTUDIO_JEFE": 6,
         "CAT_OCUP_JEFE_DESC": "Cuenta propia", "IPCM": 1200000},
    ])[FEATURES]
    ex_prob = winner.predict_proba(examples)[:, 1]
    ex = examples.copy(); ex["PROBABILIDAD_ESTIMADA_INTERNET"] = ex_prob
    ex["CLASIFICACION_UMBRAL_0_5"] = winner.predict(examples)
    ex.to_csv(out / "ejemplos_prediccion.csv", index=False)
    examples.to_csv(out / "entrada_ejemplo.csv", index=False)

    summary = {
        "tema": "Factores asociados al acceso a internet en hogares paraguayos",
        "fuente": "EPHC Anual 2025 - INE Paraguay (según documentación del proyecto)",
        "unidad_analisis": "Hogar",
        "filas_modelado": int(len(X)), "predictores": FEATURES,
        "variable_objetivo": TARGET, "distribucion_objetivo": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "particiones": {"entrenamiento": int(len(X_train)), "validacion": int(len(X_val)), "prueba": int(len(X_test))},
        "semilla": RANDOM_STATE, "folds_cv": folds,
        "modelo_seleccionado": winner_name,
        "criterio_seleccion": "Mayor F1 en validación; AUC-ROC como desempate",
        "metricas_prueba": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
        "advertencias": ["No se incorporó el diseño muestral complejo ni FACTOR como ponderador.",
                         "Los resultados describen capacidad predictiva y no causalidad.",
                         "Evaluación interna en EPHC 2025; no existe validación externa en otra encuesta/año."]
    }
    with open(out / "resumen_modelo.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    versions = {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    with open(out / "versiones_entorno.json", "w", encoding="utf-8") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)
    print("\nModelo seleccionado:", winner_name)
    print(split_metrics.to_string(index=False))
    print("Matriz de confusión:\n", cm)
    print("Artefactos guardados en:", out)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
